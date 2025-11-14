import logging
import os
import sys
from pathlib import Path

import dspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from signatures import GraderPerQuestion, GraderSingle_without_prompt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model_builder import build_lm, model_configs  # noqa: E402
from src.mlflow_config import setup_mlflow  # noqa: E402

logging.getLogger("dspy").setLevel(logging.ERROR)

# Enable nice tqdm integration with pandas
tqdm.pandas(desc="Grading answers")


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


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    label_order: list = ["incorrect", "partial", "correct"],
):
    """
    Compute comprehensive evaluation metrics from predictions and true labels.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        test_df: DataFrame with test data (for topic information)
        label_order: List of label names in order

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("DETAILED EVALUATION")
    print("=" * 60)

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

    # Check if topic column exists in test dataframe
    if "topic" in test_df.columns:
        # Get topic information from the test dataframe
        test_topics = test_df["topic"].tolist()

        # Get all unique topics in the test set
        unique_test_topics = sorted(list(set(test_topics)))

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

    # Add y_true and y_pred to metrics for confusion matrix plotting
    evaluation_metrics["y_true"] = y_true.tolist()
    evaluation_metrics["y_pred"] = y_pred.tolist()

    return evaluation_metrics


def collect_predictions(
    test_df: pd.DataFrame,
    grader_single,
    grader_perq,
    mode: str,
    pass_reference_answer: bool,
):
    """
    Collect predictions from grader based on mode.

    Args:
        test_df: DataFrame with test data
        grader_single: Grader for single mode (GraderSingle_without_prompt)
        grader_perq: Grader for per_question mode (GraderPerQuestion)
        mode: "single" or "per_question"
        pass_reference_answer: Whether to pass reference_answer to grader

    Returns:
        Tuple of (y_true, y_pred) as numpy arrays
    """
    assert mode in {"single", "per_question"}, f"Invalid eval mode: {mode}"
    print(f"Collecting predictions (mode={mode})...")

    label_name_to_int = {
        "incorrect": 0,
        "partial": 1,
        "partially correct": 1,
        "correct": 2,
    }

    def compute_labels_list(df: pd.DataFrame):
        vals = []
        for _, row in df.iterrows():
            val = row.get("labels", None)
            if isinstance(val, str):
                label = label_name_to_int.get(val.strip().lower(), -1)
            elif pd.notna(val):
                label = int(val)
            else:
                label = -1
            vals.append(label)
        return vals

    if mode == "single":

        def grade_row(row):
            try:
                kwargs = {
                    "question": row["question"],
                    "answer": row["student_answer"],
                }
                if pass_reference_answer:
                    kwargs["reference_answer"] = row["reference_answer"]
                result = grader_single(**kwargs)
                predicted = int(result.label)
            except Exception as e:
                tqdm.write(f"Error grading answer: {e}")
                raise e

            val = row.get("labels", None)
            if isinstance(val, str):
                label = label_name_to_int.get(val.strip().lower(), 0)
            elif pd.notna(val):
                label = int(val)
            else:
                label = 0

            return {"predicted": predicted, "labels": label}

        results = test_df.progress_apply(grade_row, axis=1)
        predicted_labels = results.map(lambda d: d["predicted"]).tolist()
        labels = results.map(lambda d: d["labels"]).tolist()

    elif mode == "per_question":
        # Group by task_id if available, else by question text
        group_key = "task_id" if "task_id" in test_df.columns else "question"
        index_to_pos = {idx: pos for pos, idx in enumerate(test_df.index)}
        predicted_labels = [-1] * len(test_df)

        for _, group in tqdm(test_df.groupby(group_key), desc="Grading per question"):
            group = group.copy()
            answers = group["student_answer"].astype(str).tolist()
            question = str(group.iloc[0]["question"])
            reference_answer = str(group.iloc[0]["reference_answer"])
            try:
                kwargs = {
                    "question": question,
                    "answers": answers,
                }
                if pass_reference_answer:
                    kwargs["reference_answer"] = reference_answer
                batch_result = grader_perq(**kwargs)
                labels_batch = batch_result.predicted_labels

                # Align labels back to the dataframe rows
                for k, row_idx in enumerate(group.index):
                    if k < len(labels_batch):
                        predicted_labels[index_to_pos[row_idx]] = int(labels_batch[k])
            except Exception as e:
                tqdm.write(f"Error grading group: {e}")
                raise e

        labels = compute_labels_list(test_df)

    # Convert to numpy arrays
    y_true = np.array(labels)
    y_pred = np.array(predicted_labels)

    return y_true, y_pred


def log_config_params_to_mlflow(cfg):
    """Log configuration parameters to MLflow, excluding paths."""
    # Parameters to exclude
    exclude_keys = {
        "csv_test",
        "dir",  # file paths and directories
    }

    # Flatten the config and log parameters
    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            # Handle OmegaConf objects and regular dicts
            if hasattr(v, "items"):  # OmegaConf or dict-like object
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Convert OmegaConf to regular dict first, then flatten
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flattened_cfg = flatten_dict(cfg_dict)

    # Filter out excluded parameters
    params_to_log = {}
    for key, value in flattened_cfg.items():
        # Check if any part of the key path should be excluded
        should_exclude = any(exclude_key in key for exclude_key in exclude_keys)
        if not should_exclude:
            # Convert value to string for MLflow logging
            params_to_log[key] = str(value)

    # Log parameters to MLflow
    mlflow.log_params(params_to_log)
    print(f"Logged {len(params_to_log)} parameters to MLflow")


"""
Main evaluation pipeline
"""

# Load config and paths
base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")

# Support loading custom config via environment variable
dspy_eval_config_path = os.environ.get(
    "DSPY_EVAL_CONFIG", str(PROJECT_ROOT / "configs" / "dspy_eval.yaml")
)
dspy_eval_cfg = OmegaConf.load(dspy_eval_config_path)
cfg = OmegaConf.merge(base_cfg, dspy_eval_cfg)
output_dir = os.path.join(PROJECT_ROOT, cfg.output.dir)
os.makedirs(output_dir, exist_ok=True)

# Setup MLflow tracking URI from config
setup_mlflow(cfg, PROJECT_ROOT)

# Get single model from config
model_name = cfg.model.model if hasattr(cfg.model, "model") else cfg.model.base

# Validate model is available
available_models = set(model_configs.keys())
if model_name not in available_models:
    raise ValueError(
        f"Model {model_name} is not available in model_builder.py. "
        f"Available models: {sorted(available_models)}"
    )

print(f"Evaluating model: {model_name}")

# Get mode from config
mode = getattr(cfg.model, "mode", "single")
assert mode in {"single", "per_question"}, (
    f"Invalid mode: {mode}. Must be 'single' or 'per_question'"
)
print(f"Evaluation mode: {mode}")

# Load test dataset
test_csv_path = os.path.join(PROJECT_ROOT, cfg.dataset.csv_test)
print(f"Loading test set from {test_csv_path}")
test_df = pd.read_csv(test_csv_path, sep=";")
print(f"Test set size: {len(test_df)}")

# Get pass_reference_answer setting
pass_reference_answer = getattr(cfg.model, "pass_reference_answer", True)

# Configure MLflow experiment
mlflow.set_experiment("DSPy-Evaluation")

# Start MLflow run
model_short = model_name.split("/")[-1]
run_name = f"{model_short}_{mode}"

try:
    with mlflow.start_run(run_name=run_name) as run:
        # Log model name and evaluation method as parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("mode", mode)
        mlflow.log_param("evaluation_method", "dspy")
        log_config_params_to_mlflow(cfg)

        # Log test dataset as MLflow Dataset
        test_ml_dataset = mlflow.data.from_pandas(
            test_df, source=test_csv_path, name="test_dataset"
        )
        mlflow.log_input(test_ml_dataset, context="evaluation")

        # Build LM
        build_lm_kwargs = {
            "max_tokens": cfg.model.max_tokens,
            "cache": cfg.model.cache,
        }
        if hasattr(cfg.model, "temperature") and cfg.model.temperature is not None:
            build_lm_kwargs["temperature"] = cfg.model.temperature

        grader_lm = build_lm(model_name, **build_lm_kwargs)
        print(f"Using DSPy LM {model_name}")

        dspy.configure(lm=grader_lm)

        # Create graders based on mode
        grader_single = dspy.Predict(GraderSingle_without_prompt)
        grader_single.set_lm(grader_lm)

        grader_perq = dspy.Predict(GraderPerQuestion)
        grader_perq.set_lm(grader_lm)

        # Collect predictions
        print("Running evaluation...")
        y_true, y_pred = collect_predictions(
            test_df,
            grader_single,
            grader_perq,
            mode,
            pass_reference_answer,
        )

        # Compute comprehensive metrics
        evaluation_metrics = compute_evaluation_metrics(
            y_true, y_pred, test_df, label_order=["incorrect", "partial", "correct"]
        )

        # Log all metrics to MLflow
        for metric_name, metric_value in evaluation_metrics.items():
            # Skip y_true and y_pred from MLflow metrics (they're arrays, not scalars)
            if metric_name not in ["y_true", "y_pred"]:
                mlflow.log_metric(metric_name, metric_value)

        # Plot and save confusion matrix
        model_short_safe = model_short.replace("/", "_").replace("-", "_")
        mode_suffix = {"single": "single", "per_question": "perq"}.get(mode, mode)
        confusion_matrix_path = os.path.join(
            output_dir, f"confusion_matrix_{model_short_safe}_{mode_suffix}.png"
        )
        plot_confusion_matrix(
            evaluation_metrics["y_true"],
            evaluation_metrics["y_pred"],
            save_path=confusion_matrix_path,
        )

        # Log confusion matrix image to MLflow
        if os.path.exists(confusion_matrix_path):
            mlflow.log_artifact(confusion_matrix_path, "confusion_matrices")

        print(f"\nEvaluation complete. MLflow run ID: {run.info.run_id}")

except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback

    traceback.print_exc()
    raise

print("\n" + "=" * 80)
print("Evaluation complete!")
print("=" * 80)
