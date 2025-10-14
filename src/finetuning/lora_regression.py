# %%
import os
from pathlib import Path
import sys
import mlflow
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.common import (
    setup_training_args,
    setup_trainer,
)


def load_and_preprocess_data_regression(
    dataset_csv: str,
    cache_dir: str | None,
    seed: int = 42,
    test_size: float = 0.5,
):
    """Load dataset for regression task using normalized_grade as target."""
    print(f"Loading dataset from {dataset_csv} for regression...")
    full_dataset = load_dataset(
        "csv",
        data_files={"data": dataset_csv},
        cache_dir=cache_dir,
        sep=";",
    )["data"]

    # For regression, we use normalized_grade as the label
    def prepare_regression_labels(example):
        # Ensure normalized_grade is a float between 0 and 1
        grade = float(example.get("normalized_grade", 0.0))
        example["labels"] = grade
        return example

    full_dataset = full_dataset.map(prepare_regression_labels)

    # Split the dataset
    print("Splitting dataset (random split for regression)...")
    raw = full_dataset.train_test_split(test_size=test_size, seed=seed)

    print(f"Number of training samples: {len(raw['train'])}")
    print(f"Number of test samples: {len(raw['test'])}")
    print(f"Total samples: {len(raw['train']) + len(raw['test'])}")

    # Show statistics of labels
    train_labels = np.array(raw["train"]["labels"])
    test_labels = np.array(raw["test"]["labels"])
    print(
        f"Train labels - mean: {train_labels.mean():.3f}, std: {train_labels.std():.3f}"
    )
    print(f"Test labels - mean: {test_labels.mean():.3f}, std: {test_labels.std():.3f}")

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
    """Return a compute_metrics function that has access to weights for wRMSE calculation."""

    def compute_metrics(eval_pred):
        """Compute regression metrics: RMSE and wRMSE."""
        if hasattr(eval_pred, "predictions"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred

        # Predictions from regression head are shape (batch_size, 1), squeeze to (batch_size,)
        if predictions.ndim > 1:
            predictions = predictions.squeeze()

        # Compute RMSE (used for training)
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)

        # Compute wRMSE (weighted RMSE) for evaluation
        # Get weights from the evaluation dataset
        weights = np.array(eval_dataset["weight"])

        # Weighted squared errors
        squared_errors = (predictions - labels) ** 2
        weighted_mse = np.sum(weights * squared_errors) / np.sum(weights)
        wrmse = np.sqrt(weighted_mse)

        return {
            "rmse": float(rmse),
            "wrmse": float(wrmse),
        }

    return compute_metrics


def main() -> None:
    print("Loading config...")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    training_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "training.yaml")
    cfg = OmegaConf.merge(base_cfg, training_cfg)

    dataset_csv: str = str(PROJECT_ROOT / cfg.dataset.csv_path)
    print(f"dataset_csv: {dataset_csv}")
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
                "eval_strategy": str(cfg.training.eval_strategy),
                "seed": int(getattr(cfg.project, "seed", 42)),
                "include_reference_answer": bool(
                    getattr(cfg.tokenization, "include_reference_answer", False)
                ),
                "load_in_4bit": bool(getattr(cfg.quantization, "load_in_4bit", False)),
            }
        )

        # Load and preprocess data for regression
        raw_data = load_and_preprocess_data_regression(
            dataset_csv,
            cache_dir,
            int(getattr(cfg.project, "seed", 42)),
            test_size=cfg.dataset.test_size,
        )

        # Log dataset info
        mlflow.log_params(
            {
                "train_size": len(raw_data["train"]),
                "test_size": len(raw_data["test"]),
                "total_size": len(raw_data["train"]) + len(raw_data["test"]),
            }
        )

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
            init_lora_weights=str(cfg.lora.init_weights),
        )
        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()

        # Save raw train and test datasets
        raw_data["train"].to_pandas().to_csv(
            f"{output_dir}/train.csv", index=False, sep=";"
        )
        raw_data["test"].to_pandas().to_csv(
            f"{output_dir}/test.csv", index=False, sep=";"
        )

        # Tokenize dataset (keeping weight column for wRMSE calculation)
        include_ref_ans = bool(
            getattr(cfg.tokenization, "include_reference_answer", False)
        )
        include_chunk = bool(getattr(cfg.tokenization, "include_chunk_text", False))

        # Custom tokenization that preserves weight column
        print("Tokenizing dataset...")

        def tokenize_batch(batch):
            from src.common import tokenize_fn

            return tokenize_fn(batch, tokenizer, include_ref_ans, include_chunk)

        # Get column names from train split
        columns_to_remove = [
            c for c in raw_data["train"].column_names if c not in {"labels", "weight"}
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

        trainer, loss_callback = setup_trainer(
            model, training_args, tokenized_data, tokenizer
        )
        # Override compute_metrics for regression with weights for wRMSE
        trainer.compute_metrics = compute_metrics_regression_with_weights(
            tokenized_data["test"]
        )

        # Training
        print("Starting training...")
        trainer.train()
        metrics = trainer.evaluate()

        # Log final loss information
        if loss_callback.train_losses:
            print(f"\nFinal Training Loss: {loss_callback.train_losses[-1]:.4f}")
            mlflow.log_metric("final_train_loss", loss_callback.train_losses[-1])

        if loss_callback.eval_losses:
            print(f"Final Evaluation Loss: {loss_callback.eval_losses[-1]:.4f}")
            mlflow.log_metric("final_eval_loss", loss_callback.eval_losses[-1])

        # Log final metrics
        mlflow.log_metrics(metrics)

        # Print regression metrics
        print("\n\nRegression Evaluation Metrics:")
        print(f"RMSE (training metric): {metrics.get('eval_rmse', 'N/A'):.4f}")
        print(f"wRMSE (evaluation metric): {metrics.get('eval_wrmse', 'N/A'):.4f}")

        # Save adapter locally
        adapter_path = (
            Path(output_dir)
            / f"adapter-{model_name.split('/')[-1]}-{cfg.dataset.dataset_name}-regression"
        )
        model.save_pretrained(adapter_path)

        # Push adapter to Hugging Face Hub (optional)
        save_model = bool(getattr(cfg.output, "save_model", False))
        if save_model:
            print("\nTrying to save model to Hugging Face...")
            try:
                repo_name = f"{model_name.split('/')[-1]}-lora-{cfg.dataset.dataset_name}-regression"
                model.push_to_hub(repo_name)
                print(f"Adapter successfully pushed to Hugging Face Hub: {repo_name}")
                mlflow.log_param("hf_hub_repo", repo_name)
            except Exception as e:
                print(f"Warning: Could not push adapter to Hugging Face Hub: {e}")
                mlflow.log_param("hf_hub_error", str(e))

        # Log model artifacts
        mlflow.log_artifacts(str(adapter_path), "adapter")

        # Log the full training configuration as an artifact
        mlflow.log_artifact(PROJECT_ROOT / "configs" / "training.yaml", "config")

        print("\n\nTraining completed")
        print(f"Adapter saved to: {adapter_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting regression fine-tuning...")
    main()
