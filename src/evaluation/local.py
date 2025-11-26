# %%
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
eval_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "evaluation_local.yaml")
cfg = OmegaConf.merge(base_cfg, eval_cfg)

enforce_cpu = bool(getattr(cfg.classifier_eval, "enforce_cpu", False))

# CRITICAL: Enforce CPU-only execution before any torch imports if needed
if enforce_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import DatasetDict, load_dataset
from peft import PeftModel
import torch
import mlflow
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
from src.mlflow_config import setup_mlflow  # noqa: E402


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for adapter path inference.
    Removes org prefix and special characters.

    Examples:
        "Qwen/Qwen3-0.6B" -> "Qwen3-0.6B"
        "google/flan-t5-large" -> "flan-t5-large"
    """
    # Remove org prefix (everything before the last '/')
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    return model_name


def extract_dataset_name(csv_path: str, explicit_name: str = None) -> str:
    """
    Extract dataset name from CSV path or use explicit name if provided.

    Examples:
        "data/gras/test.csv" -> "gras"
        "data/SciEntsBank_3way/test_ud.csv" -> "SciEntsBank_3way"
    """
    if explicit_name:
        return explicit_name

    # Extract dataset name from path
    # Assumes format: data/{dataset_name}/... or data/{dataset_name}.csv
    path_parts = Path(csv_path).parts
    if len(path_parts) >= 2 and path_parts[0] == "data":
        # Get the directory name or filename without extension
        dataset_part = path_parts[1]
        if dataset_part.endswith(".csv"):
            return dataset_part[:-4]  # Remove .csv extension
        return dataset_part

    # Fallback: use filename without extension
    return Path(csv_path).stem


def infer_adapter_path(
    model_name: str,
    dataset_name: str,
    adapter_source: str,
    hf_username: str = None,
    project_root: Path = None,
    paths_output_dir: str = None,
) -> Optional[str]:
    """
    Infer adapter path based on model name, dataset name, and adapter source.

    Parameters:
    - model_name: Full model name (e.g., "Qwen/Qwen3-0.6B")
    - dataset_name: Dataset name (e.g., "gras")
    - adapter_source: "hub", "local", or "none"
    - hf_username: HuggingFace username (required for "hub")
    - project_root: Project root path (required for "local")
    - paths_output_dir: Output directory from paths config (e.g., "results/") (required for "local")

    Returns:
    - Adapter path string, or None if adapter_source is "none"
    """
    if adapter_source == "none":
        return None

    sanitized_model = sanitize_model_name(model_name)

    if adapter_source == "hub":
        if not hf_username:
            raise ValueError(
                "huggingface_username is required when adapter_source is 'hub'"
            )
        return f"{hf_username}/{sanitized_model}-lora-{dataset_name}"

    elif adapter_source == "local":
        if not project_root or not paths_output_dir:
            raise ValueError(
                "project_root and paths_output_dir are required when adapter_source is 'local'"
            )
        adapter_path = os.path.join(
            project_root,
            paths_output_dir,
            "peft_output",
            f"adapter-{sanitized_model}-{dataset_name}",
        )
        return adapter_path

    else:
        raise ValueError(
            f"Invalid adapter_source: {adapter_source}. Must be 'hub', 'local', or 'none'."
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

    # Setup MLflow tracking URI from config
    setup_mlflow(cfg, PROJECT_ROOT)

    # Set MLflow experiment
    experiment_name = getattr(cfg.mlflow, "experiment_name", "classifier_evaluation")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name}")

    # Validate configuration
    models_list = getattr(cfg.classifier_eval, "models", None)
    if not models_list:
        raise ValueError("classifier_eval.models must be set to a list of model names")

    # Convert OmegaConf ListConfig to Python list if needed
    try:
        models_list = list(models_list)
    except (TypeError, ValueError):
        raise ValueError("classifier_eval.models must be a list of model names")

    if len(models_list) == 0:
        raise ValueError("classifier_eval.models must be a non-empty list")

    # Get adapter configuration
    adapter_source = getattr(cfg.classifier_eval.adapter, "source", "local")
    if adapter_source not in ["hub", "local", "none"]:
        raise ValueError(
            f"Invalid adapter_source: {adapter_source}. Must be 'hub', 'local', or 'none'."
        )

    hf_username = getattr(cfg.classifier_eval.adapter, "huggingface_username", None)
    if adapter_source == "hub" and not hf_username:
        raise ValueError(
            "huggingface_username is required when adapter_source is 'hub'"
        )

    # Get dataset name the model was trained on (for adapter inference)
    dataset_trained_on_name = getattr(
        cfg.classifier_eval.adapter, "dataset_trained_on_name", None
    )
    if adapter_source != "none" and not dataset_trained_on_name:
        raise ValueError(
            "adapter.dataset_trained_on_name is required when adapter_source is not 'none'"
        )

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

    # Infer dataset name being tested on from CSV path
    dataset_test_on_name = extract_dataset_name(csv_path, None)
    print(f"Dataset trained on: {dataset_trained_on_name}")
    print(f"Dataset tested on: {dataset_test_on_name}")

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

    # Tokenize dataset once (shared across all models)
    include_ref_ans = bool(getattr(cfg.tokenization, "include_reference_answer", False))
    print(f"\nTokenizing dataset (include_reference_answer={include_ref_ans})...")
    # We'll tokenize per model since tokenizers may differ, but prepare the raw dataset here

    # Setup output directory - infer from test dataset name
    paths_output_dir = str(cfg.paths.output_dir)
    output_dir = os.path.normpath(
        os.path.join(PROJECT_ROOT, paths_output_dir, dataset_test_on_name)
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Trainer setup configuration (shared across models)
    default_batch_size = 8 if enforce_cpu else 32
    per_device_eval_batch_size = int(
        getattr(cfg.classifier_eval, "batch_size", default_batch_size)
    )
    report_to = str(getattr(cfg.classifier_eval, "report_to", "mlflow"))
    timing_enabled = bool(getattr(cfg.classifier_eval, "timing", False))
    num_examples = len(ds)

    # Loop over each model
    print(f"\n{'=' * 60}")
    print(f"EVALUATING {len(models_list)} MODEL(S)")
    print(f"{'=' * 60}\n")

    for model_idx, base_model_name in enumerate(models_list, 1):
        base_model_name = str(base_model_name)
        device_info = " (CPU-only)" if enforce_cpu else ""
        print(f"\n{'=' * 60}")
        print(f"MODEL {model_idx}/{len(models_list)}: {base_model_name}{device_info}")
        print(f"{'=' * 60}")

        # Infer adapter path using dataset_trained_on_name
        adapter_path = None
        if adapter_source != "none":
            paths_output_dir = str(cfg.paths.output_dir)
            adapter_path = infer_adapter_path(
                model_name=base_model_name,
                dataset_name=dataset_trained_on_name,
                adapter_source=adapter_source,
                hf_username=hf_username,
                project_root=PROJECT_ROOT,
                paths_output_dir=paths_output_dir,
            )
            print(f"Inferred adapter path: {adapter_path}")

            # For local adapters, check if path exists before proceeding
            if adapter_source == "local":
                adapter_path_local = os.path.normpath(adapter_path)
                if not os.path.exists(adapter_path_local):
                    print(
                        f"WARNING: LoRA adapter directory not found at '{adapter_path_local}'. Skipping model {base_model_name}."
                    )
                    continue

        # Create MLflow run name
        sanitized_model = sanitize_model_name(base_model_name)
        if adapter_source == "hub" and adapter_path:
            adapter_suffix = f"_{adapter_path.split('/')[-1]}"
        elif adapter_source == "local" and adapter_path:
            adapter_suffix = f"_{os.path.basename(adapter_path)}"
        elif adapter_source == "none":
            adapter_suffix = "_base"
        else:
            adapter_suffix = ""
        run_name = f"eval_{sanitized_model}{adapter_suffix}"

        # Start MLflow run for this model evaluation
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(
                {
                    "model_name": base_model_name,
                    "adapter_source": adapter_source,
                    "adapter_path": str(adapter_path) if adapter_path else "none",
                    "dataset_trained_on_name": dataset_trained_on_name
                    if dataset_trained_on_name
                    else "none",
                    "dataset_test_on_name": dataset_test_on_name,
                    "dataset_csv_path": csv_path,
                    "batch_size": per_device_eval_batch_size,
                    "sample_fraction": sample_fraction,
                    "sample_seed": sample_seed,
                    "include_reference_answer": include_ref_ans,
                    "enforce_cpu": enforce_cpu,
                    "num_examples": num_examples,
                }
            )

            # Load tokenizer and base model
            tokenizer, base_model = setup_model_and_tokenizer(
                base_model_name, label2id, id2label, cache_path
            )

            # Explicitly move to CPU if CPU enforcement is enabled
            if enforce_cpu:
                base_model = base_model.to("cpu")
                print("Base model loaded and moved to CPU")

            # Load adapter if needed
            if adapter_source == "none":
                print("\nUsing base model without adapter")
                model = base_model
            elif adapter_source == "hub":
                print(f"\nLoading LoRA adapter from Hugging Face Hub: {adapter_path}")
                try:
                    load_kwargs = {}
                    if enforce_cpu:
                        load_kwargs["device_map"] = "cpu"
                    model = PeftModel.from_pretrained(
                        base_model, adapter_path, **load_kwargs
                    )
                    if enforce_cpu:
                        model = model.to("cpu")
                    print(f"Successfully loaded adapter from Hub: {adapter_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load adapter from Hugging Face Hub '{adapter_path}': {e}"
                    )
            elif adapter_source == "local":
                # Path existence already checked above
                adapter_path_local = os.path.normpath(adapter_path)
                print(f"\nLoading LoRA adapter from local path: {adapter_path_local}")
                load_kwargs = {}
                if enforce_cpu:
                    load_kwargs["device_map"] = "cpu"
                model = PeftModel.from_pretrained(
                    base_model, adapter_path_local, **load_kwargs
                )
                if enforce_cpu:
                    model = model.to("cpu")
                print("Successfully loaded adapter from local path")

            # Verify model is on correct device if CPU enforcement is enabled
            if enforce_cpu:
                print("\nModel device check:")
                for name, param in model.named_parameters():
                    if param.device.type != "cpu":
                        print(
                            f"WARNING: Parameter {name} is on {param.device}, expected CPU"
                        )
                    break  # Just check first parameter
                print("Model is on CPU (verified)")

            # Tokenize dataset with this model's tokenizer
            tokenized = tokenize_dataset(raw, tokenizer, include_ref_ans)

            # Create trainer for this model
            model_output_dir = os.path.join(
                output_dir, sanitize_model_name(base_model_name)
            )
            os.makedirs(model_output_dir, exist_ok=True)

            if enforce_cpu:
                print(
                    f"\nSetting up Trainer with batch_size={per_device_eval_batch_size} (CPU-friendly)"
                )

            training_args = TrainingArguments(
                output_dir=model_output_dir,
                per_device_eval_batch_size=per_device_eval_batch_size,
                do_train=False,
                do_eval=True,
                report_to=report_to,
                logging_strategy="no",
                fp16=False if enforce_cpu else None,
                bf16=False if enforce_cpu else None,
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
            print(f"RUNNING DETAILED EVALUATION - {base_model_name}")
            print("=" * 60)

            if timing_enabled:
                print(
                    f"Evaluating {num_examples} examples{' on CPU' if enforce_cpu else ''}..."
                )
                eval_start_time = time.time()

            # Run detailed evaluation and capture metrics
            evaluation_metrics = detailed_evaluation(
                trainer, tokenized["test"], label_order
            )

            # Log evaluation metrics to MLflow
            mlflow.log_metrics(evaluation_metrics)

            if timing_enabled:
                eval_end_time = time.time()
                total_eval_time = eval_end_time - eval_start_time
                examples_per_minute = (
                    (num_examples / total_eval_time) * 60 if total_eval_time > 0 else 0
                )
                avg_time_per_example = (
                    total_eval_time / num_examples if num_examples > 0 else 0
                )

                # Log timing metrics to MLflow
                mlflow.log_metrics(
                    {
                        "total_eval_time_seconds": total_eval_time,
                        "examples_per_minute": examples_per_minute,
                        "avg_time_per_example_seconds": avg_time_per_example,
                    }
                )

                print("\n" + "=" * 60)
                print("EVALUATION TIMING METRICS")
                print("=" * 60)
                print(
                    f"Total evaluation time: {total_eval_time:.2f} seconds ({total_eval_time / 60:.2f} minutes)"
                )
                print(f"Number of examples evaluated: {num_examples}")
                print(f"Examples per minute: {examples_per_minute:.2f}")
                print(
                    f"Average time per example: {avg_time_per_example:.4f} seconds ({avg_time_per_example * 1000:.2f} ms)"
                )
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

            # Create and save confusion matrix with model-specific naming
            confusion_matrix_filename = (
                f"confusion_matrix_{sanitized_model}{adapter_suffix}.png"
            )
            confusion_matrix_path = os.path.join(
                model_output_dir, confusion_matrix_filename
            )

            plot_confusion_matrix(y_true, y_pred, save_path=confusion_matrix_path)

            # Log confusion matrix as artifact
            mlflow.log_artifact(confusion_matrix_path, "confusion_matrix")

            print("\n" + "=" * 60)
            print(f"EVALUATION COMPLETE - {base_model_name}")
            print("=" * 60)
            print(f"Results saved to: {model_output_dir}")
            print(f"Confusion matrix saved to: {confusion_matrix_path}")

            # Clean up model from memory before loading next one
            del model, base_model, tokenizer, trainer
            if torch.cuda.is_available() and not enforce_cpu:
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL MODELS EVALUATED")
    print("=" * 60)
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
