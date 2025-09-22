import os
from pathlib import Path
from typing import Dict, Any
import sys

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config


def main() -> None:
    print("Loading config...")
    cfg = load_config("peft_lora")

    dataset_csv: str = str(cfg.dataset_csv)
    model_name: str = str(cfg.model_name)
    output_dir: str = str(cfg.output_dir)
    cache_dir: str | None = str(cfg.hf_cache_dir) if "hf_cache_dir" in cfg else None

    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Load CSV dataset and split
    print("Loading dataset...")
    raw = load_dataset(
        "csv",
        data_files={"data": dataset_csv},
        cache_dir=cache_dir,
    )["data"].train_test_split(test_size=0.1, seed=int(getattr(cfg, "seed", 42)))

    # Labels mapping (order matters)
    label_order = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    def map_labels(example: Dict[str, Any]) -> Dict[str, Any]:
        print("Mapping labels...")
        label_val = str(example.get("intended_label", "")).strip().lower()
        example["labels"] = label2id.get(label_val, 0)
        return example

    raw = raw.map(map_labels)

    # Text fields -> simple concat: "Question: ...\nAnswer: ..."
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(
                batch.get("question", [""] * len(batch["labels"])),
                batch.get("student_answer", [""] * len(batch["labels"])),
            )
        ]
        return tokenizer(texts, truncation=True)

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in raw["train"].column_names if c not in {"labels"}],
    )

    # Base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
    )

    # LoRA config (DistilBERT uses q_lin/v_lin; adjust target_modules if you change model)
    lora_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.lora_alpha),
        lora_dropout=float(cfg.lora.lora_dropout),
        target_modules=list(cfg.lora.target_modules),
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_cfg)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg.training.num_train_epochs),
        per_device_train_batch_size=int(cfg.training.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.training.per_device_eval_batch_size),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        eval_strategy=str(cfg.training.eval_strategy),
        save_strategy=str(getattr(cfg.training, "save_strategy", "epoch")),
        logging_steps=int(getattr(cfg.training, "logging_steps", 10)),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
        seed=int(getattr(cfg, "seed", 42)),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).astype(np.float32).mean().item()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save adapter and tokenizer1
    model.save_pretrained(Path(output_dir) / "adapter")
    tokenizer.save_pretrained(output_dir)

    print("Training complete. Eval metrics:", metrics)
    print(f"Adapter saved to: {Path(output_dir) / 'adapter'}")


if __name__ == "__main__":
    print("Starting...")
    main()
