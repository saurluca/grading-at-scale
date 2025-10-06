# %%
import os
import sys
from pathlib import Path
from typing import Dict, List

from datasets import DatasetDict, load_dataset
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# from synthetic_data.evaluate_answers import plot_confusion_matrix
from src.common import (  # noqa: E402
    compute_metrics,
    detailed_evaluation,
    setup_model_and_tokenizer,
    tokenize_dataset,
    map_labels,
)


def main() -> None:
    print("Evaluating Classifier")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    eval_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "evaluation.yaml")
    cfg = OmegaConf.merge(base_cfg, eval_cfg)

    # Label maps (fixed order)
    label_order: List[str] = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Load dataset exclusively from CSV
    csv_path_cfg = getattr(cfg.classifier_eval.dataset, "csv_path", None)
    if not csv_path_cfg:
        raise ValueError("classifier_eval.dataset.csv_path must be set to a CSV file")

    csv_path = os.path.normpath(os.path.join(PROJECT_ROOT, str(csv_path_cfg)))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    print(f"Loading evaluation data from CSV: {csv_path}")

    # Use semicolon as separator to match existing data generation unless changed later
    ds = load_dataset(
        "csv",
        data_files={"test": csv_path},
        cache_dir=str(
            getattr(
                cfg,
                "hf_cache_dir",
                getattr(cfg.paths, "hf_cache_dir", ".hf_cache"),
            )
        ),
        sep=";",
    )["test"]

    # Map labels to class indices
    ds = ds.map(lambda x: map_labels(x, label2id))

    # Build a DatasetDict expected by downstream code; we evaluate on the provided CSV
    raw: DatasetDict = DatasetDict({"test": ds})

    # Print a sample example
    print("An example from eval split:")
    if len(ds) > 0:
        print(ds[0])

    # Load tokenizer and base model, then optionally attach LoRA adapter
    base_model_name = str(cfg.classifier_eval.base_model)
    cache_dir = str(
        getattr(
            cfg,
            "hf_cache_dir",
            getattr(cfg.paths, "hf_cache_dir", ".hf_cache"),
        )
    )
    # Ensure cache directory is at project root
    cache_path = os.path.join(PROJECT_ROOT, cache_dir)
    output_dir = os.path.normpath(
        os.path.join(
            PROJECT_ROOT,
            str(getattr(cfg.classifier_eval, "output_dir", "data/scientsbank_eval")),
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, base_model = setup_model_and_tokenizer(
        base_model_name, label2id, id2label, cache_path
    )

    # Load LoRA adapter (support both new and legacy config format)
    model = base_model
    adapter_config = getattr(cfg.classifier_eval, "adapter", None)

    # New configuration format
    adapter_source = getattr(adapter_config, "source", "local")
    adapter_path_cfg = getattr(adapter_config, "path", None)

    if adapter_source == "hub":
        # Load from Hugging Face Hub
        print(f"Loading LoRA adapter from Hugging Face Hub: {adapter_path_cfg}")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path_cfg)
            print(f"Successfully loaded adapter from Hub: {adapter_path_cfg}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter from Hugging Face Hub '{adapter_path_cfg}': {e}"
            )
    elif adapter_source == "local":
        # Load from local path
        adapter_path = os.path.normpath(
            os.path.join(PROJECT_ROOT, str(adapter_path_cfg))
        )
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter directory not found at '{adapter_path}'. Exiting."
            )
        print(f"Loading LoRA adapter from local path: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("Successfully loaded adapter from local path")
    else:
        raise ValueError(
            f"Invalid adapter source '{adapter_source}'. Must be 'local' or 'hub'."
        )

    # Tokenize
    include_ref_ans = bool(getattr(cfg.tokenization, "include_reference_answer", False))
    include_chunk = bool(getattr(cfg.tokenization, "include_chunk_text", False))
    tokenized = tokenize_dataset(raw, tokenizer, include_ref_ans, include_chunk)

    # Trainer setup for evaluation only
    per_device_eval_batch_size = int(getattr(cfg.classifier_eval, "batch_size", 32))
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
        report_to=[],
        logging_strategy="no",
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    detailed_evaluation(trainer, tokenized["test"], label_order)


if __name__ == "__main__":
    main()
