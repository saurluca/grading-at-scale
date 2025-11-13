# %%
import os
from pathlib import Path
import sys
import mlflow
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from datasets import load_dataset, Value, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.common import (
    setup_training_args,
    setup_trainer,
)
from src.mlflow_config import setup_mlflow  # noqa: E402


def load_and_preprocess_data_regression(
    dataset_csv: str,
    cache_dir: str | None,
    seed: int = 42,
    test_size: float = 0.5,
    use_split_files: bool = False,
    train_csv: str | None = None,
    val_csv: str | None = None,
    test_csv: str | None = None,
):
    """Load dataset for regression task using normalized_grade as target.
    
    Args:
        dataset_csv: Path to single CSV file (used when use_split_files=False)
        cache_dir: Cache directory for datasets
        seed: Random seed for reproducibility
        test_size: Fraction of data for test set (only used when use_split_files=False)
        use_split_files: If True, use separate train/val/test files instead of splitting
        train_csv: Path to train.csv (used when use_split_files=True)
        val_csv: Path to val.csv (used when use_split_files=True)
        test_csv: Path to test.csv (used when use_split_files=True)
    
    Returns:
        DatasetDict with 'train', 'val' (if use_split_files), and 'test' splits
    """
    
    # For regression, convert labels to normalized floats
    # Handle two cases:
    # 1. ASAG2024: has normalized_grade column (already normalized 0-1)
    # 2. GRAS: has labels column (0, 1, 2) -> convert to (0.0, 0.5, 1.0)
    def prepare_regression_labels(example):
        if "normalized_grade" in example and example.get("normalized_grade") is not None:
            # Use normalized_grade if available (ASAG2024 dataset)
            grade = float(example["normalized_grade"])
        elif "labels" in example:
            # Convert integer labels (0, 1, 2) to normalized floats (0.0, 0.5, 1.0) for regression
            label_int = int(example["labels"])
            grade = float(label_int) / 2.0  # Maps 0->0.0, 1->0.5, 2->1.0
        else:
            raise ValueError("Neither 'normalized_grade' nor 'labels' column found in dataset")
        example["labels"] = grade
        return example
    
    def add_weights_if_missing(dataset):
        """Add weight column if it doesn't exist."""
        if "weight" not in dataset.column_names:
            dataset = dataset.map(lambda x: {"weight": 1.0})
            dataset = dataset.cast_column("weight", Value("float32"))
        return dataset

    if use_split_files:
        if train_csv is None or val_csv is None or test_csv is None:
            raise ValueError(
                "train_csv, val_csv, and test_csv must all be provided when use_split_files=True"
            )

        print(f"Loading pre-split datasets from {train_csv}, {val_csv}, and {test_csv} for regression...")

        # Load separate train, validation, and test files
        train_dataset = load_dataset(
            "csv",
            data_files={"data": train_csv},
            cache_dir=cache_dir,
            sep=";",
        )["data"]

        val_dataset = load_dataset(
            "csv",
            data_files={"data": val_csv},
            cache_dir=cache_dir,
            sep=";",
        )["data"]

        test_dataset = load_dataset(
            "csv",
            data_files={"data": test_csv},
            cache_dir=cache_dir,
            sep=";",
        )["data"]

        print(f"Loaded train dataset: {len(train_dataset)} samples")
        print(f"Loaded validation dataset: {len(val_dataset)} samples")
        print(f"Loaded test dataset: {len(test_dataset)} samples")

        # Prepare labels for all splits
        train_dataset = train_dataset.map(prepare_regression_labels)
        val_dataset = val_dataset.map(prepare_regression_labels)
        test_dataset = test_dataset.map(prepare_regression_labels)
        
        # Cast labels to float32
        train_dataset = train_dataset.cast_column("labels", Value("float32"))
        val_dataset = val_dataset.cast_column("labels", Value("float32"))
        test_dataset = test_dataset.cast_column("labels", Value("float32"))
        
        # Add weights if missing
        train_dataset = add_weights_if_missing(train_dataset)
        val_dataset = add_weights_if_missing(val_dataset)
        test_dataset = add_weights_if_missing(test_dataset)

        raw = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        })
    else:
        print(f"Loading dataset from {dataset_csv} for regression...")
        full_dataset = load_dataset(
            "csv",
            data_files={"data": dataset_csv},
            cache_dir=cache_dir,
            sep=";",
        )["data"]

        full_dataset = full_dataset.map(prepare_regression_labels)
        
        # Cast labels to float32 explicitly for regression (required for PyTorch)
        full_dataset = full_dataset.cast_column("labels", Value("float32"))
        
        # Add weight column if it doesn't exist (default to 1.0 for all samples)
        full_dataset = add_weights_if_missing(full_dataset)

        # Split the dataset
        print("Splitting dataset (random split for regression)...")
        raw = full_dataset.train_test_split(test_size=test_size, seed=seed)

    print(f"Number of training samples: {len(raw['train'])}")
    if "val" in raw:
        print(f"Number of validation samples: {len(raw['val'])}")
    print(f"Number of test samples: {len(raw['test'])}")
    total_samples = len(raw['train']) + len(raw['test'])
    if "val" in raw:
        total_samples += len(raw['val'])
    print(f"Total samples: {total_samples}")

    # Show statistics of labels
    train_labels = np.array(raw["train"]["labels"])
    test_labels = np.array(raw["test"]["labels"])
    print(
        f"Train labels - mean: {train_labels.mean():.3f}, std: {train_labels.std():.3f}"
    )
    print(f"Test labels - mean: {test_labels.mean():.3f}, std: {test_labels.std():.3f}")
    if "val" in raw:
        val_labels = np.array(raw["val"]["labels"])
        print(f"Val labels - mean: {val_labels.mean():.3f}, std: {val_labels.std():.3f}")

    return raw


def setup_model_and_tokenizer_regression(
    model_name: str,
    cache_dir: str | None,
):
    """Setup model with num_labels=1 for regression."""
    print(f"Loading tokenizer and model from {model_name} for regression...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model with num_labels=1 for regression
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        cache_dir=cache_dir,
    )

    # Set problem type to regression
    base_model.config.problem_type = "regression"

    # Resize embeddings if new tokens were added
    if hasattr(base_model, "resize_token_embeddings"):
        base_model.resize_token_embeddings(len(tokenizer))
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, base_model


def compute_metrics_regression_with_weights(eval_dataset):
    """Return a compute_metrics function that has access to weights for wRMSE calculation.
    
    Handles both datasets with and without weights. If weights don't exist or are all 1.0,
    wRMSE will equal RMSE.
    """

    def compute_metrics(eval_pred):
        """Compute regression metrics: RMSE and wRMSE (if weights available)."""
        if hasattr(eval_pred, "predictions"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred

        # Predictions from regression head are shape (batch_size, 1), squeeze to (batch_size,)
        if predictions.ndim > 1:
            predictions = predictions.squeeze()

        # Ensure labels are float (they should be, but double-check)
        labels = labels.astype(np.float32)

        # Compute RMSE (used for training)
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)

        # Compute wRMSE (weighted RMSE) for evaluation if weights are available
        # Check if weight column exists and has non-uniform weights
        has_weights = "weight" in eval_dataset.column_names
        if has_weights:
            weights = np.array(eval_dataset["weight"], dtype=np.float32)
            # Only compute wRMSE if weights are not all equal (i.e., actually weighted)
            if not np.allclose(weights, weights[0]):
                squared_errors = (predictions - labels) ** 2
                weighted_mse = np.sum(weights * squared_errors) / np.sum(weights)
                wrmse = np.sqrt(weighted_mse)
            else:
                # All weights are equal, wRMSE = RMSE
                wrmse = rmse
        else:
            # No weights available, wRMSE = RMSE
            wrmse = rmse

        return {
            "rmse": float(rmse),
            "wrmse": float(wrmse),
        }

    return compute_metrics


def detailed_evaluation_regression_as_classification(trainer, test_dataset):
    """Evaluate regression model as classification problem.
    
    Converts regression predictions (0.0-1.0) back to classification labels (0, 1, 2)
    and computes classification metrics including weighted_f1.
    
    Args:
        trainer: Trained trainer with regression model
        test_dataset: Test dataset with regression labels (0.0, 0.5, 1.0)
    
    Returns:
        Dictionary of classification metrics for MLflow logging
    """
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION EVALUATION (from regression predictions)")
    print("=" * 60)
    
    # Get predictions from regression model
    predictions = trainer.predict(test_dataset)
    
    # Extract regression predictions (continuous values 0.0-1.0)
    pred_regression = predictions.predictions
    if pred_regression.ndim > 1:
        pred_regression = pred_regression.squeeze()
    pred_regression = pred_regression.astype(np.float32)
    
    # Get true labels (regression format: 0.0, 0.5, 1.0)
    labels_regression = predictions.label_ids.astype(np.float32)
    
    # Convert regression predictions to classification labels (0, 1, 2)
    # Round to nearest: 0.0->0, 0.5->1, 1.0->2
    # Method: round(pred * 2) and clip to [0, 2]
    pred_classification = np.round(pred_regression * 2).astype(int)
    pred_classification = np.clip(pred_classification, 0, 2)
    
    # Convert true regression labels back to classification labels
    labels_classification = np.round(labels_regression * 2).astype(int)
    labels_classification = np.clip(labels_classification, 0, 2)
    
    # Label names for reporting
    label_order = ["incorrect", "partial", "correct"]
    
    # Compute classification metrics
    accuracy = accuracy_score(labels_classification, pred_classification)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_classification, pred_classification, average=None, zero_division=0
    )
    
    # Compute weighted F1 (the main metric requested)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels_classification, pred_classification, average="weighted", zero_division=0
    )
    
    # Compute macro F1
    macro_f1 = np.mean(f1)
    
    # Print results
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print("-" * 50)
    print(
        classification_report(
            labels_classification, pred_classification, 
            target_names=label_order, zero_division=0
        )
    )
    
    # Confusion Matrix
    cm = confusion_matrix(labels_classification, pred_classification)
    print("Confusion Matrix:")
    print("-" * 30)
    print("Predicted ->")
    print("Actual", end="")
    for label in label_order:
        print(f"{label:>10}", end="")
    print()
    for i, label in enumerate(label_order):
        print(f"{label:>6}", end="")
        for j in range(len(label_order)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    # Return metrics for MLflow logging
    evaluation_metrics = {
        "test_accuracy": float(accuracy),
        "test_weighted_f1": float(weighted_f1),
        "test_macro_f1": float(macro_f1),
    }
    
    # Add per-class metrics
    for i, label in enumerate(label_order):
        evaluation_metrics.update(
            {
                f"test_{label}_precision": float(precision[i]),
                f"test_{label}_recall": float(recall[i]),
                f"test_{label}_f1": float(f1[i]),
                f"test_{label}_support": int(support[i]),
            }
        )
    
    return evaluation_metrics


def main() -> None:
    print("Loading config...")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    training_cfg_path = os.environ.get(
        "TRAINING_CONFIG_PATH",
        str(PROJECT_ROOT / "configs" / "training.yaml"),
    )
    training_cfg = OmegaConf.load(training_cfg_path)
    cfg = OmegaConf.merge(base_cfg, training_cfg)

    # Determine if using split files or single file
    use_split_files = bool(getattr(cfg.dataset, "use_split_files", False))

    if use_split_files:
        # Use separate train/val/test files
        dataset_base_path = PROJECT_ROOT / "data" / cfg.dataset.dataset_name
        train_csv = str(
            dataset_base_path / getattr(cfg.dataset, "train_file", "train.csv")
        )
        val_csv = str(dataset_base_path / getattr(cfg.dataset, "val_file", "val.csv"))
        test_csv = str(
            dataset_base_path / getattr(cfg.dataset, "test_file", "test.csv")
        )
        dataset_csv = train_csv  # For logging purposes
        print(f"Using split files - train: {train_csv}, val: {val_csv}, test: {test_csv}")
    else:
        # Use single file to split at runtime
        dataset_csv = str(PROJECT_ROOT / cfg.dataset.csv_path)
        train_csv = None
        val_csv = None
        test_csv = None
        print(f"Using single file to split at runtime: {dataset_csv}")

    model_name: str = str(cfg.model.base)
    output_dir: str = str(PROJECT_ROOT / cfg.output.dir / "regression")
    cache_dir: str | None = str(cfg.paths.hf_cache_dir) if "paths" in cfg else None

    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        cache_path = (
            os.path.join(PROJECT_ROOT, cache_dir)
            if not os.path.isabs(cache_dir)
            else cache_dir
        )
        os.makedirs(cache_path, exist_ok=True)

    # Setup MLflow tracking URI from config
    setup_mlflow(cfg, PROJECT_ROOT)

    # Start MLflow experiment
    experiment_name = getattr(cfg.mlflow, "experiment_name", "peft_lora_regression")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"lora_regression_{model_name.split('/')[-1]}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_name": str(cfg.dataset.dataset_name),
                "dataset_csv": dataset_csv,
                "use_split_files": use_split_files,
                "output_dir": output_dir,
                "task": "regression",
                "num_labels": 1,
                "lora_r": int(cfg.lora.r),
                "lora_alpha": int(cfg.lora.alpha),
                "lora_dropout": float(cfg.lora.dropout),
                "target_modules": str(list(cfg.lora.target_modules)),
                "num_train_epochs": float(cfg.training.num_epochs),
                "per_device_train_batch_size": int(cfg.training.batch_size.train),
                "per_device_eval_batch_size": int(cfg.training.batch_size.eval),
                "learning_rate": float(cfg.training.learning_rate),
                "weight_decay": float(cfg.training.weight_decay),
                "gradient_accumulation_steps": int(
                    cfg.training.gradient_accumulation_steps
                ),
                "seed": int(getattr(cfg.project, "seed", 42)),
                "include_reference_answer": bool(
                    getattr(cfg.tokenization, "include_reference_answer", False)
                ),
            }
        )

        # Load and preprocess data for regression
        raw_data = load_and_preprocess_data_regression(
            dataset_csv,
            cache_dir,
            int(getattr(cfg.project, "seed", 42)),
            test_size=cfg.dataset.test_size,
            use_split_files=use_split_files,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
        )

        # Log dataset info
        dataset_info = {
            "train_size": len(raw_data["train"]),
            "test_size": len(raw_data["test"]),
            "total_size": len(raw_data["train"]) + len(raw_data["test"]),
        }
        if "val" in raw_data:
            dataset_info["val_size"] = len(raw_data["val"])
            dataset_info["total_size"] = len(raw_data["train"]) + len(raw_data["val"]) + len(raw_data["test"])
        mlflow.log_params(dataset_info)

        # Setup model and tokenizer for regression
        tokenizer, base_model = setup_model_and_tokenizer_regression(
            model_name, cache_path
        )

        # Setup LoRA model
        lora_cfg = LoraConfig(
            r=int(cfg.lora.r),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
            target_modules=cfg.lora.target_modules,
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()

        # Save raw train, val, and test datasets
        raw_data["train"].to_pandas().to_csv(
            f"{output_dir}/train.csv", index=False, sep=";"
        )
        if "val" in raw_data:
            raw_data["val"].to_pandas().to_csv(
                f"{output_dir}/val.csv", index=False, sep=";"
            )
        raw_data["test"].to_pandas().to_csv(
            f"{output_dir}/test.csv", index=False, sep=";"
        )

        # Tokenize dataset (keeping weight column for wRMSE calculation)
        include_ref_ans = bool(
            getattr(cfg.tokenization, "include_reference_answer", False)
        )

        # Custom tokenization that preserves weight column
        print("Tokenizing dataset...")

        def tokenize_batch(batch):
            from src.common import tokenize_fn

            return tokenize_fn(batch, tokenizer, include_ref_ans)

        # Get column names from train split
        # Preserve labels and weight (if it exists) columns
        columns_to_preserve = {"labels"}
        if "weight" in raw_data["train"].column_names:
            columns_to_preserve.add("weight")
        columns_to_remove = [
            c for c in raw_data["train"].column_names if c not in columns_to_preserve
        ]

        tokenized_data = raw_data.map(
            tokenize_batch,
            batched=True,
            remove_columns=columns_to_remove,
        )

        # Setup training arguments and trainer
        training_args = setup_training_args(cfg, output_dir)
        # Override metric settings for regression
        training_args.metric_for_best_model = "rmse"
        training_args.greater_is_better = False  # Lower RMSE is better

        # Log eval_strategy and eval_steps after setup (since they may be overridden)
        eval_strategy_to_log = training_args.eval_strategy
        # Convert enum to string if needed (TrainingArguments uses IntervalStrategy enum)
        if hasattr(eval_strategy_to_log, 'value'):
            eval_strategy_to_log = eval_strategy_to_log.value
        else:
            eval_strategy_to_log = str(eval_strategy_to_log).replace("IntervalStrategy.", "").lower()
        mlflow.log_param("eval_strategy", eval_strategy_to_log)
        if eval_strategy_to_log == "steps":
            mlflow.log_param("eval_steps", training_args.eval_steps)

        trainer = setup_trainer(
            model, training_args, tokenized_data, tokenizer, cfg
        )
        # Override compute_metrics for regression with weights for wRMSE
        # Use the same eval_dataset that trainer uses (val if available, otherwise test)
        # This ensures we validate on validation set during training, not test set
        eval_dataset_for_training = trainer.eval_dataset
        trainer.compute_metrics = compute_metrics_regression_with_weights(
            eval_dataset_for_training
        )
        
        # Log which dataset is used for validation during training
        eval_set_name = "validation" if "val" in tokenized_data else "test"
        print(f"Using {eval_set_name} set for validation during training")

        # Training
        print("Starting training...")
        # Evaluate on validation set before training (if available, otherwise test set)
        print(f"Evaluating on {eval_set_name} set before training...")
        initial_metrics = trainer.evaluate()
        print(f"Metrics before training: {initial_metrics}")
        mlflow.log_metrics({f"initial_{k}": v for k, v in initial_metrics.items()})

        trainer.train()

        # The best model is automatically loaded at the end of training (load_best_model_at_end=True)
        # Final evaluation metrics from validation set (used during training)
        final_eval_metrics = trainer.evaluate()
        print(f"\nFinal evaluation metrics (on {eval_set_name} set used during training):")
        print(f"RMSE: {final_eval_metrics.get('eval_rmse', 'N/A'):.4f}")
        print(f"wRMSE: {final_eval_metrics.get('eval_wrmse', 'N/A'):.4f}")
        mlflow.log_metrics({f"final_eval_{k}": v for k, v in final_eval_metrics.items()})
        
        # Now evaluate on test set (only at the end, not during training)
        print("\nPerforming final evaluation on test dataset...")
        # Update compute_metrics to use test set for final evaluation
        trainer.compute_metrics = compute_metrics_regression_with_weights(
            tokenized_data["test"]
        )
        test_metrics = trainer.evaluate(eval_dataset=tokenized_data["test"])
        print(f"\nTest set regression metrics:")
        print(f"RMSE: {test_metrics.get('eval_rmse', 'N/A'):.4f}")
        print(f"wRMSE: {test_metrics.get('eval_wrmse', 'N/A'):.4f}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        
        # Perform detailed classification evaluation on test set (treating regression as classification)
        print("\nPerforming detailed classification evaluation on test dataset...")
        classification_metrics = detailed_evaluation_regression_as_classification(
            trainer, tokenized_data["test"]
        )
        
        # Log classification metrics to MLflow
        mlflow.log_metrics(classification_metrics)

        # Save adapter locally
        if cfg.output.save_model_locally:
            adapter_path = (
                Path(output_dir)
                / f"adapter-{model_name.split('/')[-1]}-{cfg.dataset.dataset_name}-regression"
            )
            model.save_pretrained(adapter_path)
            print(f"Adapter saved to: {adapter_path}")
            mlflow.log_artifacts(str(adapter_path), "adapter")
        else:
            adapter_path = (
                Path(output_dir)
                / f"adapter-{model_name.split('/')[-1]}-{cfg.dataset.dataset_name}-regression"
            )
            print("LoRA adapter saving to MLflow skipped (save_model_locally=false in config)")

        # Push adapter to Hugging Face Hub
        if cfg.output.push_to_hub:
            print("\nSaving model to huggingface")
            try:
                repo_name = f"{model_name.split('/')[-1]}-lora-{cfg.dataset.dataset_name}-regression"
                model.push_to_hub(repo_name)
                print(f"Adapter successfully pushed to Hugging Face Hub: {repo_name}")
                mlflow.log_param("hf_hub_repo", repo_name)
            except Exception as e:
                print(f"Warning: Could not push adapter to Hugging Face Hub: {e}")
                mlflow.log_param("hf_hub_error", str(e))

        # Log the full training configuration as an artifact
        mlflow.log_artifact(PROJECT_ROOT / "configs" / "training.yaml", "config")

        print("\n\nTraining completed")
        if cfg.output.save_model_locally:
            print(f"Adapter saved to: {adapter_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting regression fine-tuning...")
    main()
