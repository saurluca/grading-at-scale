import os
import math
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import mlflow
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config
from optimisation.common import LossLoggingCallback  # reuse logging callback


def _infer_default_lora_targets(model_type: str) -> list[str]:
    # Sensible defaults by architecture
    mt = (model_type or "").lower()
    if mt in {"llama"}:
        return ["q_proj", "v_proj"]
    if mt in {"distilbert"}:  # includes distilroberta which uses DistilBERT blocks
        return ["q_lin", "v_lin"]
    if mt in {"roberta", "bert"}:
        return ["query", "value"]
    # Fallback
    return ["q_proj", "v_proj"]


def _resolve_target_modules_for_model(cfg, base_model) -> list[str]:
    configured = list(getattr(cfg.lora, "target_modules", []))
    model_type = str(getattr(getattr(base_model, "config", object()), "model_type", ""))
    default_targets = _infer_default_lora_targets(model_type)

    # If user configured values, prefer them if they match any module names
    if configured:
        module_names = "\n".join(dict(base_model.named_modules()).keys())
        if any(t in module_names for t in configured):
            return configured
        print(
            f"Warning: None of the configured LoRA target_modules {configured} were found in model modules; "
            f"falling back to defaults for model_type={model_type}: {default_targets}"
        )
    return default_targets


def load_typst_dataset(
    cache_dir: str | None,
    seed: int = 42,
    typst_only: bool = True,
    test_size: float = 0.1,
) -> DatasetDict:
    """Load TechxGenus/Typst-Train and split into train/test.

    The dataset includes Typst sources and some Markdown. If typst_only is True,
    filter to rows whose 'language' equals 'typst' when available.
    """
    print("Loading TechxGenus/Typst-Train dataset...")
    ds = load_dataset("TechxGenus/Typst-Train", split="train", cache_dir=cache_dir)

    # Filter by language if present and requested
    if typst_only and "language" in ds.column_names:
        unique_langs = set(ds["language"])  # for debug
        print(f"Languages in dataset: {sorted(unique_langs)}")
        ds = ds.filter(lambda x: (x.get("language") or "").lower() == "typst")

    # Keep only non-empty content
    if "content" not in ds.column_names:
        raise ValueError("Dataset does not contain 'content' column")
    ds = ds.filter(
        lambda x: isinstance(x.get("content"), str) and len(x["content"].strip()) > 0
    )

    # Shuffle and split
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=float(test_size), seed=seed)
    print(f"Train size: {len(split['train'])}, Test size: {len(split['test'])}")
    return DatasetDict(train=split["train"], test=split["test"])


def setup_tokenizer(model_name: str, cache_dir: str | None):
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True
    )
    # Ensure pad token exists; for causal LMs, use eos as pad
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_and_group(
    dataset: DatasetDict,
    tokenizer,
    block_size: int = 1024,
    cache_dir: str | None = None,
    model_name: str | None = None,
    num_proc: int | None = None,
) -> DatasetDict:
    print("Tokenizing and grouping text into blocks...")

    # Determine cache location keyed by model and block size
    safe_model = (model_name or "model").replace("/", "_").replace(":", "_")
    base_cache = Path(cache_dir) if cache_dir else (_PROJECT_ROOT / ".hf_cache")
    token_cache = base_cache / "typst_tokenized" / safe_model / f"bs{block_size}"

    # If cached tokenized+grouped datasets exist, load and return
    train_cache_dir = token_cache / "train"
    test_cache_dir = token_cache / "test"
    if train_cache_dir.exists() and test_cache_dir.exists():
        try:
            print(f"Loading tokenized dataset from cache: {token_cache}")
            return DatasetDict(
                train=load_from_disk(str(train_cache_dir)),
                test=load_from_disk(str(test_cache_dir)),
            )
        except Exception as e:
            print(f"Warning: failed to load tokenized cache, recomputing: {e}")

    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["content"], return_special_tokens_mask=False)

    # Drop original text columns to avoid mixing strings into grouping
    effective_procs = num_proc or max(1, (os.cpu_count() or 1) - 1)
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=list(dataset["train"].column_names),
        num_proc=effective_procs,
        desc="Tokenizing",
    )

    # Concatenate texts and split into blocks of block_size tokens
    def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
        token_keys = [
            k
            for k in examples.keys()
            if k in {"input_ids", "attention_mask", "token_type_ids"}
        ]
        concatenated = {k: sum(examples[k], []) for k in token_keys}
        total_length = (
            len(concatenated["input_ids"]) if concatenated["input_ids"] else 0
        )
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(
        group_texts, batched=True, num_proc=effective_procs, desc="Grouping"
    )

    # Save grouped datasets to disk cache
    try:
        (token_cache / "train").mkdir(parents=True, exist_ok=True)
        (token_cache / "test").mkdir(parents=True, exist_ok=True)
        grouped["train"].save_to_disk(str(train_cache_dir))
        grouped["test"].save_to_disk(str(test_cache_dir))
    except Exception as e:
        print(f"Warning: could not save tokenized cache: {e}")

    return grouped


def setup_lora_causal_lm(model_name: str, cfg, tokenizer, cache_dir: str | None):
    print(f"Loading base CausalLM model from {model_name} and applying LoRA...")

    # Encoder-only models (DistilBERT/DistilRoBERTa/BERT/Roberta) lack native CLM heads.
    # Use a generation-capable checkpoint per web guidance (distilroberta-base-finetuned-wikitext2).
    # Ref: fxis.ai guidance on using distilroberta-base-finetuned-wikitext2 for generation.
    resolved_model_name = model_name
    lower_name = model_name.lower()
    if (
        any(s in lower_name for s in ["distilroberta", "distilbert", "roberta", "bert"])
        and "gpt" not in lower_name
    ):
        print(
            "Info: encoder-only backbone detected; switching to 'distilroberta-base-finetuned-wikitext2' for causal generation."
        )
        resolved_model_name = "distilroberta-base-finetuned-wikitext2"

    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_model_name, cache_dir=cache_dir
    )

    # If we added a new pad token, resize embeddings
    if hasattr(base_model, "resize_token_embeddings"):
        base_model.resize_token_embeddings(len(tokenizer))

    target_modules = _resolve_target_modules_for_model(cfg, base_model)

    lora_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.lora_alpha),
        lora_dropout=float(cfg.lora.lora_dropout),
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model


def setup_training_args(cfg, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg.training.num_train_epochs),
        per_device_train_batch_size=int(cfg.training.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.training.per_device_eval_batch_size),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        eval_strategy=str(cfg.training.eval_strategy),
        save_strategy=str(getattr(cfg.training, "save_strategy", "epoch")),
        logging_steps=int(getattr(cfg.training, "logging_steps", 10)),
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
        seed=int(getattr(cfg, "seed", 42)),
        save_total_limit=2,
    )


def train_and_evaluate(
    model, tokenizer, tokenized_data: DatasetDict, training_args: TrainingArguments
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("Setting up Trainer...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loss_callback = LossLoggingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics and math.isfinite(eval_metrics["eval_loss"]):
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])  # type: ignore[index]

    # Collect final epoch losses from callback if available
    aux_metrics: Dict[str, Any] = {}
    if loss_callback.train_losses:
        aux_metrics["final_train_loss"] = loss_callback.train_losses[-1]
    if loss_callback.eval_losses:
        aux_metrics["final_eval_loss"] = loss_callback.eval_losses[-1]

    return eval_metrics, aux_metrics


def main() -> None:
    print("Loading config peft_lora...")
    cfg = load_config("peft_lora_typst")

    model_name: str = str(cfg.model_name)
    output_dir = str(Path(cfg.output_dir) / "typst_lora")
    cache_dir: str | None = str(cfg.hf_cache_dir) if "hf_cache_dir" in cfg else None
    seed: int = int(getattr(cfg, "seed", 42))
    block_size: int = int(getattr(cfg, "block_size", 1024))
    typst_only: bool = bool(getattr(cfg, "typst_only", True))
    test_size: float = float(getattr(cfg, "test_size", 0.1))
    num_proc: int = int(getattr(cfg, "num_proc", max(1, (os.cpu_count() or 1) - 1)))

    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    mlflow.set_experiment("peft_lora_typst_training")

    with mlflow.start_run(run_name=f"lora_typst_{model_name.split('/')[-1]}"):
        # Log configuration
        mlflow.log_params(
            {
                "dataset": "TechxGenus/Typst-Train",
                "model_name": model_name,
                "output_dir": output_dir,
                "lora_r": int(cfg.lora.r),
                "lora_alpha": int(cfg.lora.lora_alpha),
                "lora_dropout": float(cfg.lora.lora_dropout),
                "target_modules": str(list(cfg.lora.target_modules)),
                "num_train_epochs": float(cfg.training.num_train_epochs),
                "per_device_train_batch_size": int(
                    cfg.training.per_device_train_batch_size
                ),
                "per_device_eval_batch_size": int(
                    cfg.training.per_device_eval_batch_size
                ),
                "learning_rate": float(cfg.training.learning_rate),
                "weight_decay": float(cfg.training.weight_decay),
                "eval_strategy": str(cfg.training.eval_strategy),
                "seed": seed,
                "block_size": block_size,
                "typst_only": typst_only,
                "test_size": test_size,
                "num_proc": num_proc,
            }
        )

        # Data
        raw_data = load_typst_dataset(
            cache_dir=cache_dir,
            seed=seed,
            typst_only=typst_only,
            test_size=test_size,
        )

        # Tokenizer and Model
        tokenizer = setup_tokenizer(model_name, cache_dir)
        tokenized = tokenize_and_group(
            raw_data,
            tokenizer,
            block_size=block_size,
            cache_dir=cache_dir,
            model_name=model_name,
            num_proc=num_proc,
        )
        model = setup_lora_causal_lm(model_name, cfg, tokenizer, cache_dir)

        # Trainer
        training_args = setup_training_args(cfg, output_dir)
        eval_metrics, aux_metrics = train_and_evaluate(
            model, tokenizer, tokenized, training_args
        )

        # Log metrics
        mlflow.log_metrics(eval_metrics)
        mlflow.log_metrics(aux_metrics)

        # Optionally save adapter and tokenizer
        adapter_dir = Path(output_dir) / "adapter"
        try:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            # model.save_pretrained(adapter_dir)  # Uncomment to save adapter
            # tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Warning: could not save adapter: {e}")

        print("Evaluation metrics:")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    print("Starting Typst LoRA finetuning...")
    main()
