import os
from pathlib import Path
from typing import Dict, Any
import sys

import numpy as np
import mlflow
import mlflow.transformers
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType

if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config


class LossLoggingCallback(TrainerCallback):
    """Custom callback to log training and validation metrics per step and epoch."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log metrics at each logging step."""
        if state.log_history:
            latest_log = state.log_history[-1]

            # Log training metrics per step
            if "train_loss" in latest_log:
                train_loss = latest_log["train_loss"]
                self.train_losses.append(train_loss)
                mlflow.log_metric("train_loss_step", train_loss, step=state.global_step)

            if "train_accuracy" in latest_log:
                train_acc = latest_log["train_accuracy"]
                mlflow.log_metric(
                    "train_accuracy_step", train_acc, step=state.global_step
                )

    def on_epoch_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Log loss at the end of each epoch."""
        if state.log_history:
            # Get the latest log entry
            latest_log = state.log_history[-1]

            # Extract training loss if available
            if "train_loss" in latest_log:
                train_loss = latest_log["train_loss"]
                self.train_losses.append(train_loss)
                print(f"Epoch {state.epoch}: Training Loss = {train_loss:.4f}")
                mlflow.log_metric("train_loss", train_loss, step=int(state.epoch))

            # Extract evaluation loss if available
            if "eval_loss" in latest_log:
                eval_loss = latest_log["eval_loss"]
                self.eval_losses.append(eval_loss)
                print(f"Epoch {state.epoch}: Evaluation Loss = {eval_loss:.4f}")
                mlflow.log_metric("eval_loss", eval_loss, step=int(state.epoch))

            # Log other metrics if available
            if "eval_accuracy" in latest_log:
                eval_acc = latest_log["eval_accuracy"]
                print(f"Epoch {state.epoch}: Evaluation Accuracy = {eval_acc:.4f}")
                mlflow.log_metric("eval_accuracy", eval_acc, step=int(state.epoch))


def map_labels(example: Dict[str, Any], label2id: Dict[str, int]) -> Dict[str, Any]:
    """Map string labels to integer IDs."""
    label_val = str(example.get("intended_label", "")).strip().lower()
    example["labels"] = label2id.get(label_val, 0)
    return example


def tokenize_fn(batch: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Tokenize text data for model input."""
    texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(
            batch.get("question", [""] * len(batch["labels"])),
            batch.get("student_answer", [""] * len(batch["labels"])),
        )
    ]
    return tokenizer(texts, truncation=True)


def compute_metrics(eval_pred):
    """Compute accuracy using logits and labels from EvalPrediction or tuple."""
    # Support both (logits, labels) tuple and EvalPrediction object
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


def load_and_preprocess_data(dataset_csv: str, cache_dir: str | None, seed: int = 42):
    """Load and preprocess the dataset."""
    print("Loading dataset...")
    raw = load_dataset(
        "csv",
        data_files={"data": dataset_csv},
        cache_dir=cache_dir,
    )["data"].train_test_split(test_size=0.1, seed=seed)

    print(f"Number of training samples: {len(raw['train'])}")
    print(f"Number of test samples: {len(raw['test'])}")
    print(f"Total samples: {len(raw['train']) + len(raw['test'])}")

    # Labels mapping (order matters)
    label_order = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Map labels
    raw = raw.map(lambda x: map_labels(x, label2id))

    return raw, label_order, label2id, id2label


def setup_model_and_tokenizer(
    model_name: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    cache_dir: str | None,
):
    """Setup the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
    )

    return tokenizer, base_model


def setup_lora_model(base_model, cfg):
    """Setup LoRA configuration and apply to base model."""
    lora_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.lora_alpha),
        lora_dropout=float(cfg.lora.lora_dropout),
        target_modules=list(cfg.lora.target_modules),
        task_type=TaskType.SEQ_CLS,
    )
    return get_peft_model(base_model, lora_cfg)


def setup_training_args(cfg, output_dir: str):
    """Setup training arguments."""
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
        logging_strategy="steps",  # Log per step for training accuracy
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
        seed=int(getattr(cfg, "seed", 42)),
        # Enable evaluation and logging
        save_total_limit=2,
    )


def setup_trainer(model, training_args, tokenized_data, tokenizer):
    """Setup the trainer with model, data, and configuration."""
    data_collator = DataCollatorWithPadding(tokenizer)

    # Create the loss logging callback
    loss_callback = LossLoggingCallback()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback],
    ), loss_callback


def tokenize_dataset(raw_data, tokenizer):
    """Tokenize the dataset for training."""

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenize_fn(batch, tokenizer)

    return raw_data.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            c for c in raw_data["train"].column_names if c not in {"labels"}
        ],
    )


def detailed_evaluation(trainer, test_dataset, label_order):
    """
    Perform detailed evaluation on the test dataset and calculate comprehensive metrics.
    """
    print("\n" + "=" * 60)
    print("DETAILED EVALUATION")
    print("=" * 60)

    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    # Print detailed results
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print()

    # Per-class metrics
    print("Per-class Metrics:")
    print("-" * 50)
    for i, label in enumerate(label_order):
        print(
            f"{label:>10}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}"
        )

    print()
    print("Classification Report:")
    print("-" * 50)
    print(
        classification_report(y_true, y_pred, target_names=label_order, zero_division=0)
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
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
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    # Add per-class metrics
    for i, label in enumerate(label_order):
        evaluation_metrics.update(
            {
                f"{label}_precision": precision[i],
                f"{label}_recall": recall[i],
                f"{label}_f1": f1[i],
                f"{label}_support": int(support[i]),
            }
        )

    return evaluation_metrics


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

    # Start MLflow experiment
    experiment_name = "peft_lora_training"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"lora_training_{model_name.split('/')[-1]}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_csv": dataset_csv,
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
                "seed": int(getattr(cfg, "seed", 42)),
            }
        )

        # Load and preprocess data
        raw_data, label_order, label2id, id2label = load_and_preprocess_data(
            dataset_csv, cache_dir, int(getattr(cfg, "seed", 42))
        )

        # Log dataset info
        mlflow.log_params(
            {
                "train_size": len(raw_data["train"]),
                "test_size": len(raw_data["test"]),
                "total_size": len(raw_data["train"]) + len(raw_data["test"]),
            }
        )

        # Setup model and tokenizer
        tokenizer, base_model = setup_model_and_tokenizer(
            model_name, label2id, id2label, cache_dir
        )

        # Setup LoRA model
        model = setup_lora_model(base_model, cfg)

        model.print_trainable_parameters()

        # Tokenize dataset
        tokenized_data = tokenize_dataset(raw_data, tokenizer)

        # Setup training arguments and trainer
        training_args = setup_training_args(cfg, output_dir)
        trainer, loss_callback = setup_trainer(
            model, training_args, tokenized_data, tokenizer
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

        # Perform detailed evaluation
        print("\nPerforming detailed evaluation on test dataset...")
        detailed_metrics = detailed_evaluation(
            trainer, tokenized_data["test"], label_order
        )

        # Log detailed evaluation metrics to MLflow
        mlflow.log_metrics(detailed_metrics)

        # Save adapter and tokenizer
        # model.save_pretrained(Path(output_dir) / "adapter")
        # tokenizer.save_pretrained(output_dir)

        # Log model artifacts
        # mlflow.log_artifacts(output_dir, "model_outputs")

        # Log the trained model using MLflow transformers integration
        # try:
        #     mlflow.transformers.log_model(
        #         transformers_model={
        #             "model": model,
        #             "tokenizer": tokenizer,
        #         },
        #         artifact_path="model",
        #         task="text-classification",
        #     )
        #     print("Model logged to MLflow successfully")
        # except Exception as e:
        #     print(f"Warning: Could not log model to MLflow: {e}")

        # print("Training complete. Eval metrics:", metrics)
        # print(f"Adapter saved to: {Path(output_dir) / 'adapter'}")
        # print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting...")
    main()
