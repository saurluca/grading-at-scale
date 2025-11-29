import numpy as np
from typing import Dict, Any
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from datasets import load_dataset, ClassLabel, DatasetDict, Value
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import concatenate_datasets


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


def sample_dataset(dataset, sample_fraction: float, seed: int = 42):
    if sample_fraction < 0.0 or sample_fraction > 1.0:
        raise ValueError(
            f"sample_fraction must be between 0.0 and 1.0, got {sample_fraction}"
        )

    if sample_fraction == 1.0:
        print(f"Using full dataset: {len(dataset)} samples")
        return dataset

    # Calculate number of samples to select
    total_samples = len(dataset)
    n_samples = int(total_samples * sample_fraction)

    if n_samples == 0:
        raise ValueError(f"sample_fraction {sample_fraction} results in 0 samples")

    # Use numpy random generator for reproducible sampling
    rng = np.random.default_rng(seed)
    indices = rng.choice(total_samples, size=n_samples, replace=False)

    # Sort indices to maintain some order
    indices = sorted(indices)

    sampled_dataset = dataset.select(indices)

    print(
        f"Sampled dataset: {len(sampled_dataset)}/{total_samples} samples ({sample_fraction:.1%}) using seed {seed}"
    )

    return sampled_dataset


def tokenize_fn(
    batch: Dict[str, Any],
    tokenizer,
    include_reference_answer: bool = False,
) -> Dict[str, Any]:
    # Determine batch size from any available field
    batch_size = len(next(iter(batch.values())))

    questions = batch.get("question", [""] * batch_size)
    student_answers = batch.get("student_answer", [""] * batch_size)
    reference_answers = batch.get("reference_answer", [""] * batch_size)

    texts = []
    for q, a, ref_ans in zip(questions, student_answers, reference_answers):
        parts = [f"Question: {q}", f"Answer: {a}"]
        if include_reference_answer and isinstance(ref_ans, str) and ref_ans.strip():
            parts.append(f"Reference answer: {ref_ans}")
        texts.append("\n".join(parts))

    return tokenizer(texts, truncation=True, max_length=1024)


def compute_metrics(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()

    # Calculate F1 scores
    _, _, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1).item()

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def load_and_preprocess_data(
    cache_dir: str | None,
    train_csv: str,
    val_csv: str,
    test_csv: str | None = None,
):
    if test_csv:
        print(
            f"Loading pre-split datasets from {train_csv}, {val_csv}, and {test_csv} ..."
        )
    else:
        print(
            f"Loading pre-split datasets from {train_csv} and {val_csv} (test set skipped)..."
        )

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

    print(f"Loaded train dataset: {len(train_dataset)} samples")
    print(f"Loaded validation dataset: {len(val_dataset)} samples")

    # Load test dataset only if provided
    if test_csv:
        test_dataset = load_dataset(
            "csv",
            data_files={"data": test_csv},
            cache_dir=cache_dir,
            sep=";",
        )["data"]
        print(f"Loaded test dataset: {len(test_dataset)} samples")
    else:
        test_dataset = None

    # Ensure consistent feature types across datasets before concatenation
    # Cast task_id to string to handle cases where different datasets have different types
    if "task_id" in train_dataset.column_names:
        train_dataset = train_dataset.cast_column("task_id", Value("string"))
    if "task_id" in val_dataset.column_names:
        val_dataset = val_dataset.cast_column("task_id", Value("string"))
    if test_dataset and "task_id" in test_dataset.column_names:
        test_dataset = test_dataset.cast_column("task_id", Value("string"))

    # Combine for topic counting (only if test dataset exists)
    if test_dataset:
        full_dataset = concatenate_datasets([train_dataset, val_dataset, test_dataset])
    else:
        full_dataset = concatenate_datasets([train_dataset, val_dataset])

    # Count samples per topic
    topic_counts = {}
    for topic in full_dataset["topic"]:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    print(f"\nDataset topic distribution: {topic_counts}")

    # Labels mapping (order matters)
    label_order = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Map labels on datasets
    train_dataset = train_dataset.map(lambda x: map_labels(x, label2id))
    val_dataset = val_dataset.map(lambda x: map_labels(x, label2id))
    if test_dataset:
        test_dataset = test_dataset.map(lambda x: map_labels(x, label2id))
    # Ensure 'labels' is a ClassLabel feature
    train_dataset = train_dataset.cast_column("labels", ClassLabel(names=label_order))
    val_dataset = val_dataset.cast_column("labels", ClassLabel(names=label_order))
    if test_dataset:
        test_dataset = test_dataset.cast_column("labels", ClassLabel(names=label_order))

    # Use pre-split data directly - include test only if provided
    raw_dict = {
        "train": train_dataset,
        "val": val_dataset,
    }
    if test_dataset:
        raw_dict["test"] = test_dataset
    raw = DatasetDict(raw_dict)

    print(f"Number of training samples: {len(raw['train'])}")
    if "val" in raw:
        print(f"Number of validation samples: {len(raw['val'])}")
    if "test" in raw:
        print(f"Number of test samples: {len(raw['test'])}")
    total_samples = len(raw["train"])
    if "val" in raw:
        total_samples += len(raw["val"])
    if "test" in raw:
        total_samples += len(raw["test"])
    print(f"Total samples: {total_samples}")

    # Show per-class counts in test set for verification (only if test set exists)
    if "test" in raw:
        test_labels = raw["test"]["labels"]
        counts = {name: 0 for name in label_order}
        for v in test_labels:
            counts[id2label[int(v)]] += 1
        print("Test set per-class counts:", counts)

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

    # Load the model (always execute, regardless of pad_token status)
    model_kwargs = {
        "num_labels": 3,
        "id2label": id2label,
        "label2id": label2id,
        "cache_dir": cache_dir,
    }

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_kwargs
    )

    # Resize embeddings if new tokens were added and align pad_token_id
    if hasattr(base_model, "resize_token_embeddings"):
        base_model.resize_token_embeddings(len(tokenizer))
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, base_model


def setup_training_args(cfg, output_dir: str):
    # Check if eval_steps is configured for step-based evaluation
    eval_steps = getattr(cfg.training, "eval_steps", None)
    if eval_steps is not None:
        eval_strategy = "steps"
        eval_steps = int(eval_steps)
        save_strategy = "steps"  # Match eval_strategy when using steps
    else:
        eval_strategy = str(getattr(cfg.training, "eval_strategy", "epoch"))
        save_strategy = str(getattr(cfg.output, "save_strategy", "epoch"))

    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": float(cfg.training.num_epochs),
        "per_device_train_batch_size": int(cfg.training.batch_size.train),
        "per_device_eval_batch_size": int(cfg.training.batch_size.eval),
        "learning_rate": float(cfg.training.learning_rate),
        "weight_decay": float(cfg.training.weight_decay),
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "logging_steps": int(getattr(cfg.training, "logging_steps", 10)),
        "logging_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "mlflow",
        "seed": int(getattr(cfg.project, "seed", 42)),
        "bf16": bool(getattr(cfg.training, "use_bf16", False)),
        "save_total_limit": 2,  # Keep best checkpoint and latest checkpoint
        "gradient_accumulation_steps": int(
            getattr(cfg.training, "gradient_accumulation_steps", 1)
        ),
    }

    # Add eval_steps if using step-based evaluation
    if eval_steps is not None:
        training_args_dict["eval_steps"] = eval_steps

    return TrainingArguments(**training_args_dict)


def setup_trainer(model, training_args, tokenized_data, tokenizer, cfg=None):
    print("Setting up trainer...")
    data_collator = DataCollatorWithPadding(tokenizer)

    # Use validation set if available, otherwise fallback to test set
    eval_dataset = tokenized_data.get("val", tokenized_data.get("test"))

    # Setup early stopping callback if patience is configured
    callbacks = []
    if cfg is not None:
        early_stopping_patience = getattr(cfg.training, "early_stopping_patience", None)
        if early_stopping_patience is not None:
            early_stopping_patience = int(early_stopping_patience)
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )
            print(f"Early stopping enabled with patience={early_stopping_patience}")

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )


def tokenize_dataset(
    raw_data,
    tokenizer,
    include_reference_answer: bool = False,
):
    print("Tokenizing dataset...")

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenize_fn(batch, tokenizer, include_reference_answer)

    # Get column names from any available split (e.g., "train" or "test")
    first_split = next(iter(raw_data.keys()))
    # Keep labels and topic columns, remove all others
    columns_to_remove = [
        c for c in raw_data[first_split].column_names if c not in {"labels", "topic"}
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

    # Handle case where predictions.predictions might be a tuple/list (like in compute_metrics)
    logits = predictions.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids

    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Calculate quadratic weighted kappa (for ordinal classification)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

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
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
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
        "quadratic_weighted_kappa": qwk,
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

    # Add per-topic metrics for all topics in test set
    print("\nPer-topic Metrics:")
    print("-" * 40)

    # Check if topic column exists in test dataset
    if "topic" in test_dataset.column_names:
        # Get topic information from the test dataset
        test_topics = test_dataset["topic"]

        # Get all unique topics in the test set
        unique_test_topics = list(set(test_topics))

        # Store per-topic weighted F1 scores for overall weighted average calculation
        topic_weighted_f1_scores = []
        topic_supports = []

        for topic in unique_test_topics:
            # Find indices where the topic matches
            topic_indices = [i for i, t in enumerate(test_topics) if t == topic]

            if len(topic_indices) == 0:
                print(f"{topic}: No samples found")
                evaluation_metrics[f"{topic}_quadratic_weighted_kappa"] = 0.0
                evaluation_metrics[f"{topic}_macro_f1"] = 0.0
                evaluation_metrics[f"{topic}_weighted_f1"] = 0.0
                evaluation_metrics[f"{topic}_support"] = 0
                continue

            # Get predictions and labels for this topic
            topic_y_true = np.array([y_true[i] for i in topic_indices])
            topic_y_pred = np.array([y_pred[i] for i in topic_indices])
            topic_support = len(topic_indices)

            # Calculate quadratic weighted kappa (consistent with overall)
            topic_kappa = cohen_kappa_score(
                topic_y_true, topic_y_pred, weights="quadratic"
            )

            # Calculate macro F1 for this topic
            _, _, topic_f1_macro, _ = precision_recall_fscore_support(
                topic_y_true, topic_y_pred, average="macro", zero_division=0
            )

            # Calculate weighted F1 for this topic
            _, _, topic_f1_weighted, _ = precision_recall_fscore_support(
                topic_y_true, topic_y_pred, average="weighted", zero_division=0
            )

            # Store metrics
            evaluation_metrics[f"{topic}_quadratic_weighted_kappa"] = topic_kappa
            evaluation_metrics[f"{topic}_macro_f1"] = topic_f1_macro
            evaluation_metrics[f"{topic}_weighted_f1"] = topic_f1_weighted
            evaluation_metrics[f"{topic}_support"] = topic_support

            # Store for overall weighted average calculation
            topic_weighted_f1_scores.append(topic_f1_weighted)
            topic_supports.append(topic_support)

            print(
                f"{topic}: quadratic_weighted_kappa={topic_kappa:.4f}, "
                f"macro_f1={topic_f1_macro:.4f}, "
                f"weighted_f1={topic_f1_weighted:.4f} "
                f"(support: {topic_support})"
            )

        # Calculate overall weighted average of per-topic weighted F1 scores
        if len(topic_weighted_f1_scores) > 0 and sum(topic_supports) > 0:
            overall_topic_weighted_f1 = np.average(
                topic_weighted_f1_scores, weights=topic_supports
            )
            evaluation_metrics["overall_topic_weighted_f1"] = overall_topic_weighted_f1
            print(
                f"\nOverall weighted average of per-topic weighted F1: {overall_topic_weighted_f1:.4f}"
            )
    else:
        print(
            "Warning: Topic column not found in test dataset. Skipping per-topic evaluation."
        )

    return evaluation_metrics
