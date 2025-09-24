import numpy as np
import mlflow
from typing import Dict, Any

from datasets import load_dataset, ClassLabel
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
    print(f"Loading dataset from {dataset_csv}...")
    full_dataset = load_dataset(
        "csv",
        data_files={"data": dataset_csv},
        cache_dir=cache_dir,
    )["data"]

    # Labels mapping (order matters)
    label_order = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Map labels on the full dataset (before splitting)
    full_dataset = full_dataset.map(lambda x: map_labels(x, label2id))

    # Ensure 'labels' is a ClassLabel feature to support stratified splitting
    full_dataset = full_dataset.cast_column("labels", ClassLabel(names=label_order))

    # use stratified split
    raw = full_dataset.train_test_split(
        test_size=0.2, seed=seed, stratify_by_column="labels"
    )

    print(f"Number of training samples: {len(raw['train'])}")
    print(f"Number of test samples: {len(raw['test'])}")
    print(f"Total samples: {len(raw['train']) + len(raw['test'])}")

    # Show per-class counts in test set for verification
    test_labels = raw["test"]["labels"]
    counts = {name: 0 for name in label_order}
    for v in test_labels:
        counts[id2label[int(v)]] += 1
    print("Test set per-class counts (stratified):", counts)

    return raw, label_order, label2id, id2label


def setup_model_and_tokenizer(
    model_name: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    cache_dir: str | None,
):
    print(f"Loading tokenizer and model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Ensure tokenizer has a pad token (required for batching/padding)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
    )

    # Resize embeddings if new tokens were added and align pad_token_id
    if hasattr(base_model, "resize_token_embeddings"):
        base_model.resize_token_embeddings(len(tokenizer))
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, base_model


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
    print("Setting up trainer...")
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
    print("Tokenizing dataset...")

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

    # Baselines
    # Naive majority-class classifier
    majority_class = np.bincount(y_true).argmax()
    y_pred_naive = np.full_like(y_true, fill_value=majority_class)
    _, _, f1_naive, _ = precision_recall_fscore_support(
        y_true, y_pred_naive, average=None, zero_division=0
    )
    macro_f1_naive = np.mean(f1_naive)

    # Random classifier proportional to label frequencies
    class_counts = np.bincount(y_true, minlength=len(label_order))
    class_probs = (
        class_counts / class_counts.sum()
        if class_counts.sum() > 0
        else np.ones(len(label_order)) / len(label_order)
    )
    rng = np.random.default_rng(42)
    y_pred_random = rng.choice(len(label_order), size=len(y_true), p=class_probs)
    _, _, f1_random, _ = precision_recall_fscore_support(
        y_true, y_pred_random, average=None, zero_division=0
    )
    macro_f1_random = np.mean(f1_random)

    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    # Print detailed results
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Macro F1 (naive majority): {macro_f1_naive:.4f}")
    print(f"Macro F1 (random label-proportional): {macro_f1_random:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

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
        "macro_f1_naive": macro_f1_naive,
        "macro_f1_random": macro_f1_random,
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
