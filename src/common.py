import numpy as np
import mlflow
from typing import Dict, Any

from datasets import load_dataset, ClassLabel, DatasetDict
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
    ApertusForCausalLM,
)

# AutoConfig.register("new-model", NewModelConfig)
# AutoModel.register(NewModelConfig, NewModel)


class LossLoggingCallback(TrainerCallback):
    """Log training and evaluation metrics per epoch only."""

    def __init__(self):
        self.train_losses = []  # historical epoch means
        self.eval_losses = []  # historical epoch eval losses
        self._current_epoch_train_losses = []

    def on_epoch_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        # Reset per-epoch accumulators
        self._current_epoch_train_losses = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Collect training loss during the epoch; do not log per-step."""
        if not state.log_history:
            return
        latest_log = state.log_history[-1]

        # HF Trainer emits per-step training loss under key 'loss'
        if "loss" in latest_log:
            loss_value = latest_log["loss"]
            try:
                # Some logs include NaNs during warmup; filter them out
                if loss_value is not None and np.isfinite(float(loss_value)):
                    self._current_epoch_train_losses.append(float(loss_value))
            except Exception:
                pass

    def on_epoch_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Aggregate and log metrics at the end of each epoch."""
        epoch_index = int(state.epoch) if state.epoch is not None else 0

        # 1) Training loss (epoch average of step losses)
        if self._current_epoch_train_losses:
            train_loss_epoch = float(np.mean(self._current_epoch_train_losses))
            self.train_losses.append(train_loss_epoch)
            print(f"Epoch {epoch_index}: Training Loss = {train_loss_epoch:.4f}")
            mlflow.log_metric("train_loss", train_loss_epoch, step=epoch_index)

        # 2) Evaluation metrics from the most recent eval of this epoch
        eval_log = None
        if state.log_history:
            # Iterate in reverse to find the last eval for this epoch
            for log_entry in reversed(state.log_history):
                if "eval_loss" in log_entry:
                    # If 'epoch' is present, prefer entries matching this epoch
                    entry_epoch = log_entry.get("epoch")
                    if entry_epoch is None or int(entry_epoch) == epoch_index:
                        eval_log = log_entry
                        break

        if eval_log is not None:
            if "eval_loss" in eval_log:
                eval_loss = float(eval_log["eval_loss"])  # type: ignore[arg-type]
                self.eval_losses.append(eval_loss)
                print(f"Epoch {epoch_index}: Evaluation Loss = {eval_loss:.4f}")
                mlflow.log_metric("eval_loss", eval_loss, step=epoch_index)

            if "eval_accuracy" in eval_log:
                eval_acc = float(eval_log["eval_accuracy"])  # type: ignore[arg-type]
                print(f"Epoch {epoch_index}: Evaluation Accuracy = {eval_acc:.4f}")
                mlflow.log_metric("eval_accuracy", eval_acc, step=epoch_index)


def map_labels(example: Dict[str, Any], label2id: Dict[str, int]) -> Dict[str, Any]:
    """Map string labels to integer IDs."""
    label_raw = example.get("label") or example.get("labels")
    if label_raw is None:
        raise ValueError("No label found in example.")

    # Try to interpret as a number (float or int)
    try:
        label_num = float(label_raw)
        # If it's an integer (e.g., 0, 1, 2), use as is
        if label_num.is_integer():
            example["labels"] = int(label_num)
        else:
            raise ValueError(
                f"Label value '{label_raw}' is not an integer class index."
            )
    except (ValueError, TypeError):
        # Not a number, treat as string label
        label_val = str(label_raw).strip().lower()
        if label_val in label2id:
            example["labels"] = label2id[label_val]
        else:
            raise ValueError(
                f"Label '{label_raw}' not found in label2id mapping: {label2id}"
            )
    return example


def tokenize_fn(
    batch: Dict[str, Any],
    tokenizer,
    include_reference_answer: bool = False,
    include_chunk_text: bool = False,
) -> Dict[str, Any]:
    """Tokenize text data for model input."""
    # Determine batch size from any available field
    batch_size = len(next(iter(batch.values())))

    questions = batch.get("question", [""] * batch_size)
    student_answers = batch.get("student_answer", [""] * batch_size)
    reference_answers = batch.get("reference_answer", [""] * batch_size)
    chunk_texts = batch.get("chunk_text", [""] * batch_size)

    texts = []
    for q, a, ref_ans, chunk in zip(
        questions, student_answers, reference_answers, chunk_texts
    ):
        parts = [f"Question: {q}", f"Answer: {a}"]
        if include_reference_answer and isinstance(ref_ans, str) and ref_ans.strip():
            parts.append(f"Reference answer: {ref_ans}")
        if include_chunk_text and isinstance(chunk, str) and chunk.strip():
            parts.append(f"Reference: {chunk}")
        texts.append("\n".join(parts))

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


def load_and_preprocess_data(
    dataset_csv: str,
    cache_dir: str | None,
    seed: int = 42,
    test_size: float = 0.5,
    use_unseen_questions: bool = False,
):
    print(f"Loading dataset from {dataset_csv} ...")
    full_dataset = load_dataset(
        "csv",
        data_files={"data": dataset_csv},
        cache_dir=cache_dir,
        sep=";",
    )["data"]

    # Labels mapping (order matters)
    label_order = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Map labels on the full dataset (before splitting)
    full_dataset = full_dataset.map(lambda x: map_labels(x, label2id))

    # Ensure 'labels' is a ClassLabel feature to support stratified splitting
    full_dataset = full_dataset.cast_column("labels", ClassLabel(names=label_order))

    if use_unseen_questions:
        print("Splitting by questions (unseen questions in test set)...")
        # Get unique questions
        unique_questions = list(set(full_dataset["question"]))

        # Randomly shuffle and split questions
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_questions)

        n_test_questions = int(len(unique_questions) * test_size)
        test_questions = set(unique_questions[:n_test_questions])
        train_questions = set(unique_questions[n_test_questions:])

        print(
            f"Number of unique questions: {len(unique_questions)} "
            f"(train: {len(train_questions)}, test: {len(test_questions)})"
        )

        # Filter the full dataset based on question assignment
        train_indices = [
            i for i, q in enumerate(full_dataset["question"]) if q in train_questions
        ]
        test_indices = [
            i for i, q in enumerate(full_dataset["question"]) if q in test_questions
        ]

        raw = DatasetDict(
            {
                "train": full_dataset.select(train_indices),
                "test": full_dataset.select(test_indices),
            }
        )

    else:
        print("Splitting by samples (standard stratified split)...")
        # use stratified split
        raw = full_dataset.train_test_split(
            test_size=test_size, seed=seed, stratify_by_column="labels"
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

    if model_name == "swiss-ai/Apertus-8B-Instruct-2509":
        base_model = ApertusForCausalLM.from_pretrained(model_name)

    else:
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
    # IT IS EVAL_STRATEGY, NOT EVALUATION_STRATEGY
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg.training.num_epochs),
        per_device_train_batch_size=int(cfg.training.batch_size.train),
        per_device_eval_batch_size=int(cfg.training.batch_size.eval),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        eval_strategy=str(getattr(cfg.training, "eval_strategy", "epoch")),
        save_strategy=str(getattr(cfg.output, "save_strategy", "epoch")),
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
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback],
    ), loss_callback


def tokenize_dataset(
    raw_data,
    tokenizer,
    include_reference_answer: bool = False,
    include_chunk_text: bool = False,
):
    print("Tokenizing dataset...")

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenize_fn(
            batch, tokenizer, include_reference_answer, include_chunk_text
        )

    # Get column names from any available split (e.g., "train" or "test")
    first_split = next(iter(raw_data.keys()))
    columns_to_remove = [
        c for c in raw_data[first_split].column_names if c not in {"labels"}
    ]

    return raw_data.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
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
