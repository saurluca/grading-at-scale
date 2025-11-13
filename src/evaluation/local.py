# %%
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Load config early to check for CPU enforcement before torch imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Check if CPU enforcement is needed (must be before torch imports)
base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
eval_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "evaluation.yaml")
cfg = OmegaConf.merge(base_cfg, eval_cfg)

enforce_cpu = bool(getattr(cfg.classifier_eval, "enforce_cpu", False))

# CRITICAL: Enforce CPU-only execution before any torch imports if needed
if enforce_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import DatasetDict, load_dataset
from peft import PeftModel
import torch
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Verify CPU enforcement if enabled
if enforce_cpu:
    if torch.cuda.is_available():
        print("WARNING: CUDA is available but CPU-only mode is enforced.")
        print("CUDA_VISIBLE_DEVICES is set to empty string.")
    print(f"Using device: CPU (torch.cuda.is_available()={torch.cuda.is_available()})")

from src.common import (  # noqa: E402
    compute_metrics,
    detailed_evaluation,
    setup_model_and_tokenizer,
    tokenize_dataset,
    map_labels,
    sample_dataset,
)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot a confusion matrix using seaborn's heatmap.
    
    Parameters:
    - y_true: List or array of true labels
    - y_pred: List or array of predicted labels
    - save_path: Optional path to save the plot
    """
    # Only use valid labels (0, 1, 2) for this task
    valid_labels = [0, 1, 2]
    
    # Filter to only include valid labels
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Create confusion matrix with only valid labels
    cm = confusion_matrix(y_true_array, y_pred_array, labels=valid_labels)
    
    label_names = {0: "Incorrect", 1: "Partially Correct", 2: "Correct"}
    label_display_names = [label_names[label] for label in valid_labels]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_display_names,
        yticklabels=label_display_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Grader Performance")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to: {save_path}")
    
    plt.close()  # Close the figure to free memory


def main() -> None:
    # Config already loaded at module level for CPU enforcement check
    device_mode = "CPU-only" if enforce_cpu else "GPU/CPU"
    print("=" * 60)
    print(f"EVALUATING CLASSIFIER ({device_mode})")
    print("=" * 60)

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

    # Ensure cache directory is at project root
    cache_dir = str(cfg.paths.hf_cache_dir)
    cache_path = os.path.join(PROJECT_ROOT, cache_dir)

    ds = load_dataset(
        "csv",
        data_files={"test": csv_path},
        cache_dir=cache_path,
        sep=";",
    )["test"]

    # Map labels to class indices
    ds = ds.map(lambda x: map_labels(x, label2id))

    # Apply data sampling if configured
    sample_fraction = float(
        getattr(cfg.classifier_eval.dataset, "sample_fraction", 1.0)
    )
    sample_seed = int(getattr(cfg.classifier_eval.dataset, "sample_seed", 42))
    ds = sample_dataset(ds, sample_fraction, sample_seed)

    # Build a DatasetDict expected by downstream code; we evaluate on the provided CSV
    raw: DatasetDict = DatasetDict({"test": ds})

    # Print a sample example
    print("An example from eval split:")
    if len(ds) > 0:
        print(ds[0])

    # Load tokenizer and base model, then optionally attach LoRA adapter
    base_model_name = str(cfg.classifier_eval.base_model)
    device_info = " (CPU-only)" if enforce_cpu else ""
    print(f"\nLoading base model: {base_model_name}{device_info}")

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
    
    # Explicitly move to CPU if CPU enforcement is enabled
    if enforce_cpu:
        base_model = base_model.to("cpu")
        print("Base model loaded and moved to CPU")

    # Load LoRA adapter config
    adapter_source = getattr(cfg.classifier_eval.adapter, "source", "local")
    adapter_path_cfg = getattr(cfg.classifier_eval.adapter, "path", None)

    if adapter_source == "none":
        # Use base model without any adapter
        print("\nUsing base model without adapter")
        model = base_model
    elif adapter_source == "hub":
        # Load from Hugging Face Hub
        print(f"\nLoading LoRA adapter from Hugging Face Hub: {adapter_path_cfg}")
        try:
            # Use device_map="cpu" if CPU enforcement is enabled
            load_kwargs = {}
            if enforce_cpu:
                load_kwargs["device_map"] = "cpu"
            model = PeftModel.from_pretrained(base_model, adapter_path_cfg, **load_kwargs)
            if enforce_cpu:
                model = model.to("cpu")  # Double-check CPU placement
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
        print(f"\nLoading LoRA adapter from local path: {adapter_path}")
        # Use device_map="cpu" if CPU enforcement is enabled
        load_kwargs = {}
        if enforce_cpu:
            load_kwargs["device_map"] = "cpu"
        model = PeftModel.from_pretrained(base_model, adapter_path, **load_kwargs)
        if enforce_cpu:
            model = model.to("cpu")  # Double-check CPU placement
        print("Successfully loaded adapter from local path")
    else:
        raise ValueError(
            f"Invalid adapter source '{adapter_source}'. Must be 'local', 'hub', or 'none'."
        )
    
    # Verify model is on correct device if CPU enforcement is enabled
    if enforce_cpu:
        print(f"\nModel device check:")
        for name, param in model.named_parameters():
            if param.device.type != "cpu":
                print(f"WARNING: Parameter {name} is on {param.device}, expected CPU")
            break  # Just check first parameter
        print("Model is on CPU (verified)")

    # Tokenize
    include_ref_ans = bool(getattr(cfg.tokenization, "include_reference_answer", False))
    if enforce_cpu:
        print(f"\nTokenizing dataset (include_reference_answer={include_ref_ans})...")
    tokenized = tokenize_dataset(raw, tokenizer, include_ref_ans)

    # Trainer setup for evaluation only
    # Default batch size: 8 for CPU (CPU-friendly), 32 for GPU
    default_batch_size = 8 if enforce_cpu else 32
    per_device_eval_batch_size = int(
        getattr(cfg.classifier_eval, "batch_size", default_batch_size)
    )
    
    # MLflow reporting: configurable, default to "mlflow" for backward compatibility
    report_to = str(getattr(cfg.classifier_eval, "report_to", "mlflow"))
    
    if enforce_cpu:
        print(f"\nSetting up Trainer with batch_size={per_device_eval_batch_size} (CPU-friendly)")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
        report_to=report_to,
        logging_strategy="no",
        fp16=False if enforce_cpu else None,  # Disable mixed precision for CPU
        bf16=False if enforce_cpu else None,  # Disable bfloat16 for CPU
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

    # Run detailed evaluation with optional timing
    print("\n" + "=" * 60)
    print("RUNNING DETAILED EVALUATION")
    print("=" * 60)
    
    # Optional timing metrics
    timing_enabled = bool(getattr(cfg.classifier_eval, "timing", False))
    num_examples = len(tokenized["test"])
    
    if timing_enabled:
        print(f"Evaluating {num_examples} examples{' on CPU' if enforce_cpu else ''}...")
        eval_start_time = time.time()
    
    detailed_evaluation(trainer, tokenized["test"], label_order)
    
    if timing_enabled:
        eval_end_time = time.time()
        total_eval_time = eval_end_time - eval_start_time
        examples_per_minute = (num_examples / total_eval_time) * 60 if total_eval_time > 0 else 0
        avg_time_per_example = total_eval_time / num_examples if num_examples > 0 else 0
        
        print("\n" + "=" * 60)
        print("EVALUATION TIMING METRICS")
        print("=" * 60)
        print(f"Total evaluation time: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)")
        print(f"Number of examples evaluated: {num_examples}")
        print(f"Examples per minute: {examples_per_minute:.2f}")
        print(f"Average time per example: {avg_time_per_example:.4f} seconds ({avg_time_per_example*1000:.2f} ms)")
        print("=" * 60)
    
    # Get predictions for confusion matrix
    print("\nGenerating confusion matrix...")
    predictions = trainer.predict(tokenized["test"])
    
    # Handle case where predictions.predictions might be a tuple/list
    logits = predictions.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    
    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids
    
    # Create and save confusion matrix
    base_model_short = base_model_name.split("/")[-1]
    adapter_short = ""
    if adapter_source == "hub" and adapter_path_cfg:
        adapter_short = f"_{adapter_path_cfg.split('/')[-1]}"
    elif adapter_source == "local" and adapter_path_cfg:
        # Extract adapter name from path
        adapter_path_local = os.path.normpath(
            os.path.join(PROJECT_ROOT, str(adapter_path_cfg))
        )
        adapter_short = f"_{os.path.basename(adapter_path_local)}"
    elif adapter_source == "none":
        adapter_short = "_base"
    
    confusion_matrix_filename = f"confusion_matrix_{base_model_short}{adapter_short}.png"
    confusion_matrix_path = os.path.join(output_dir, confusion_matrix_filename)
    
    plot_confusion_matrix(y_true, y_pred, save_path=confusion_matrix_path)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
