# %%
import os
import sys
from pathlib import Path
from typing import Dict, List

from datasets import DatasetDict
from peft import PeftModel
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Ensure project root is on sys.path for absolute imports (works in scripts and notebooks)
if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config

# from synthetic_data.evaluate_answers import plot_confusion_matrix
from optimisation.common import (
    setup_model_and_tokenizer,
    tokenize_dataset,
    detailed_evaluation,
    compute_metrics,
)


def main() -> None:
    print("Evaluating SciEntsBank classifier")
    cfg = load_config("synthetic_data")

    dataset_dir = os.path.normpath(
        os.path.join(_PROJECT_ROOT, cfg.scientsbank.dataset_dir)
    )
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"SciEntsBank dataset directory not found: {dataset_dir}"
        )

    # Optional warnings about reference/topic usage flags (dataset lacks these semantically)
    if getattr(cfg, "eval_pass_reference", False):
        print(
            "Warning: 'eval_pass_reference' is True, but SciEntsBank has no reference text; using empty strings."
        )

    print(f"Loading SciEntsBank from: {dataset_dir}")
    ds_dict: DatasetDict = DatasetDict.load_from_disk(dataset_dir)

    # Build DatasetDict with at least 'test' split for evaluation and a dummy 'train'
    eval_split_name = str(getattr(cfg.classifier_eval, "eval_split", "test"))
    if eval_split_name not in ds_dict:
        raise ValueError(
            f"Requested eval_split '{eval_split_name}' not found in dataset: {list(ds_dict.keys())}"
        )

    eval_split = ds_dict[eval_split_name]
    train_split = ds_dict.get("train", eval_split)
    raw = DatasetDict({"train": train_split, "test": eval_split})

    # Print first 3 examples from the eval split
    print("An examples from eval split:")
    for i in range(min(1, len(eval_split))):
        print(eval_split[i])

    # Label maps
    label_order: List[str] = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Load tokenizer and base model, then optionally attach LoRA adapter
    base_model_name = str(cfg.classifier_eval.base_model_name)
    cache_dir = str(
        getattr(
            cfg.classifier_eval,
            "hf_cache_dir",
            getattr(cfg, "hf_cache_dir", ".hf_cache"),
        )
    )
    output_dir = os.path.normpath(
        os.path.join(
            _PROJECT_ROOT,
            str(getattr(cfg.classifier_eval, "output_dir", "data/scientsbank_eval")),
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, base_model = setup_model_and_tokenizer(
        base_model_name, label2id, id2label, cache_dir
    )

    lora_adapter_dir = getattr(cfg.classifier_eval, "lora_adapter_dir", None)
    model = base_model
    if lora_adapter_dir:
        print(f"Loading LoRA adapter from: {lora_adapter_dir}")
        adapter_path = os.path.normpath(
            os.path.join(_PROJECT_ROOT, str(lora_adapter_dir))
        )
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter directory not found at '{adapter_path}'. Exiting."
            )
        else:
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)

    # Tokenize
    tokenized = tokenize_dataset(raw, tokenizer)

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
        # train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Evaluate and detailed metrics
    print("\n" + "=" * 50)
    print("EVALUATING CLASSIFIER ON SCIENTSBANK")
    print("=" * 50)
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    detailed_evaluation(trainer, tokenized["test"], label_order)

    # # Sample a few rows
    # print("\n" + "=" * 50)
    # print("SAMPLE RESULTS")
    # print("=" * 50)
    # sample_cols = [
    #     c
    #     for c in [
    #         "question",
    #         "student_answer",
    #         "predicted_label_name",
    #         "label",
    #     ]
    #     if c in df.columns
    # ]
    # print_df = df[sample_cols].sample(n=min(5, len(df)))
    # for _, row in print_df.iterrows():
    # print(
    # f"Question: {row.get('question', '')}\n"
    # f"Student Answer: {row.get('student_answer', '')}\n"
    # f"Predicted Label: {row.get('predicted_label_name', '')}\n"
    # f"True Label: {row.get('label', '')}\n" + ("-" * 40)
    # )


if __name__ == "__main__":
    main()
