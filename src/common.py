import numpy as np
import mlflow
from typing import Dict, Any
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training
import torch



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
    """
    Sample a fraction of the dataset using random sampling with a fixed seed.
    
    Args:
        dataset: The dataset to sample from
        sample_fraction: Fraction of data to use (0.0-1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Sampled dataset if fraction < 1.0, otherwise original dataset
    """
    if sample_fraction < 0.0 or sample_fraction > 1.0:
        raise ValueError(f"sample_fraction must be between 0.0 and 1.0, got {sample_fraction}")
    
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
    
    print(f"Sampled dataset: {len(sampled_dataset)}/{total_samples} samples ({sample_fraction:.1%}) using seed {seed}")
    
    return sampled_dataset


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

    return tokenizer(texts, truncation=True, max_length=1024)


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
    topics: list[str] | None = None,
):
    print(f"Loading dataset from {dataset_csv} ...")
    full_dataset = load_dataset(
        "csv",
        data_files={"data": dataset_csv},
        cache_dir=cache_dir,
        sep=";",
    )["data"]

    # Count samples per topic before any filtering
    topic_counts = {}
    for topic in full_dataset["topic"]:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    print(f"\nOriginal dataset topic distribution: {topic_counts}")

    # Labels mapping (order matters)
    label_order = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Map labels on the full dataset (before splitting)
    full_dataset = full_dataset.map(lambda x: map_labels(x, label2id))

    # Ensure 'labels' is a ClassLabel feature to support stratified splitting
    full_dataset = full_dataset.cast_column("labels", ClassLabel(names=label_order))

    # Handle topic filtering and train/test split
    if topics is not None and len(topics) > 0:
        print(f"Topic filtering enabled for topics: {topics}")
        
        # Separate data into in-topic and out-of-topic groups
        def _is_in_topic(example):
            return example["topic"] in topics
        
        in_topic_indices = [i for i, example in enumerate(full_dataset) if _is_in_topic(example)]
        out_of_topic_indices = [i for i, example in enumerate(full_dataset) if not _is_in_topic(example)]
        
        in_topic_data = full_dataset.select(in_topic_indices)
        out_of_topic_data = full_dataset.select(out_of_topic_indices)
        
        print(f"Topic separation: {len(in_topic_data)} in-topic samples, {len(out_of_topic_data)} out-of-topic samples")
        print(f"Out-of-topic samples will be added to test set")
        
        # Apply split logic to in-topic data only
        if use_unseen_questions:
            print("Splitting in-topic data by questions (unseen questions in test set)...")
            # Get unique questions from in-topic data
            unique_questions = list(set(in_topic_data["question"]))

            # Randomly shuffle and split questions
            rng = np.random.default_rng(seed)
            rng.shuffle(unique_questions)

            n_test_questions = int(len(unique_questions) * test_size)
            test_questions = set(unique_questions[:n_test_questions])
            train_questions = set(unique_questions[n_test_questions:])

            print(
                f"Number of unique questions in in-topic data: {len(unique_questions)} "
                f"(train: {len(train_questions)}, test: {len(test_questions)})"
            )

            # Filter the in-topic dataset based on question assignment
            train_indices = [
                i for i, q in enumerate(in_topic_data["question"]) if q in train_questions
            ]
            test_indices = [
                i for i, q in enumerate(in_topic_data["question"]) if q in test_questions
            ]

            in_topic_split = DatasetDict(
                {
                    "train": in_topic_data.select(train_indices),
                    "test": in_topic_data.select(test_indices),
                }
            )

        else:
            print("Splitting in-topic data by samples (standard stratified split)...")
            # use stratified split on in-topic data
            in_topic_split = in_topic_data.train_test_split(
                test_size=test_size, seed=seed, stratify_by_column="labels"
            )
        
        # Combine test sets: out-of-topic data + test portion from in-topic split
        from datasets import concatenate_datasets
        combined_test = concatenate_datasets([out_of_topic_data, in_topic_split["test"]])
        
        raw = DatasetDict(
            {
                "train": in_topic_split["train"],
                "test": combined_test,
            }
        )
        
    else:
        # No topic filtering - apply split logic to full dataset
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
    quantization_config: Dict[str, Any] | None = None,
):
    """
    Setup tokenizer and model, with optional quantization.
    
    Args:
        model_name: Name of the model to load
        label2id: Label to ID mapping
        id2label: ID to label mapping
        cache_dir: Cache directory for model files
        quantization_config: Optional dict with quantization settings:
            - load_in_4bit: bool
            - bnb_4bit_compute_dtype: str (e.g., "float16", "bfloat16")
            - bnb_4bit_quant_type: str (e.g., "nf4")
            - bnb_4bit_use_double_quant: bool
    
    Returns:
        tokenizer, base_model
    """
    print(f"Loading tokenizer and model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Ensure tokenizer has a pad token (required for batching/padding)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Setup quantization config if requested
    quant_config = None
    if quantization_config and quantization_config.get("load_in_4bit", False):
        print("Setting up 4-bit quantization...")
        compute_dtype_str = quantization_config.get("bnb_4bit_compute_dtype", "float16")
        if compute_dtype_str == "bfloat16":
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        else:
            compute_dtype = torch.float16
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    if model_name == "swiss-ai/Apertus-8B-Instruct-2509":
        base_model = ApertusForCausalLM.from_pretrained(model_name)

    else:
        model_kwargs = {
            "num_labels": 3,
            "id2label": id2label,
            "label2id": label2id,
            "cache_dir": cache_dir,
        }
        
        # Add quantization config if provided
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Prepare model for k-bit training if quantized
        if quant_config is not None:
            print("Preparing model for k-bit training...")
            base_model = prepare_model_for_kbit_training(base_model)

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
        logging_strategy="steps",  
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="mlflow",
        seed=int(getattr(cfg, "seed", 42)),
        bf16=True,
        # Enable evaluation and logging
        save_total_limit=2,
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
    )


def setup_trainer(model, training_args, tokenized_data, tokenizer):
    print("Setting up trainer...")
    data_collator = DataCollatorWithPadding(tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


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

    # Add per-topic micro F1 scores for all topics in test set
    print("\nPer-topic Micro F1 Scores:")
    print("-" * 40)
    
    # Check if topic column exists in test dataset
    if "topic" in test_dataset.column_names:
        # Get topic information from the test dataset
        test_topics = test_dataset["topic"]
        
        # Get all unique topics in the test set
        unique_test_topics = list(set(test_topics))
        
        for topic in unique_test_topics:
            # Find indices where the topic matches
            topic_indices = [i for i, t in enumerate(test_topics) if t == topic]
            
            if len(topic_indices) == 0:
                print(f"{topic}: No samples found")
                evaluation_metrics[f"{topic}_micro_f1"] = 0.0
                evaluation_metrics[f"{topic}_support"] = 0
                continue
            
            # Get predictions and labels for this topic
            topic_y_true = [y_true[i] for i in topic_indices]
            topic_y_pred = [y_pred[i] for i in topic_indices]
            
            # Calculate micro F1 for this topic
            topic_precision, topic_recall, topic_f1, topic_support = precision_recall_fscore_support(
                topic_y_true, topic_y_pred, average="micro", zero_division=0
            )
            
            print(f"{topic}: {topic_f1:.4f} (support: {len(topic_indices)})")
            evaluation_metrics[f"{topic}_micro_f1"] = topic_f1
            evaluation_metrics[f"{topic}_support"] = len(topic_indices)
    else:
        print("Warning: Topic column not found in test dataset. Skipping per-topic evaluation.")

    return evaluation_metrics
