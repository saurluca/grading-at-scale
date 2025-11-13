# %%
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
from signatures import GraderPerQuestion, GraderSingle
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model_builder import build_lm  # noqa: E402

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

    plt.show()


def evaluate_grader_performance(
    answers_df,
    grader_single,
    mode: str = "single",
    grader_perq=None,
):
    """
    Evaluate the grader's performance on the generated answers.

    Args:
        answers_df: DataFrame with student answers and labels
        grader_single: The Grader instance for single evaluations
        mode: One of {"single", "per_question"}

    Returns:
        Dictionary with evaluation metrics
    """
    assert mode in {"single", "per_question"}, f"Invalid eval mode: {mode}"
    print(f"Evaluating grader performance (mode={mode})...")

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
                if getattr(cfg.api_eval, "pass_reference_answer", True):
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

        results = answers_df.progress_apply(grade_row, axis=1)
        predicted_labels = results.map(lambda d: d["predicted"]).tolist()
        labels = results.map(lambda d: d["labels"]).tolist()

    elif mode == "per_question":
        # Group by task_id if available, else by question text
        group_key = "task_id" if "task_id" in answers_df.columns else "question"
        index_to_pos = {idx: pos for pos, idx in enumerate(answers_df.index)}
        predicted_labels = [-1] * len(answers_df)

        for _, group in tqdm(
            answers_df.groupby(group_key), desc="Grading per question"
        ):
            group = group.copy()
            answers = group["student_answer"].astype(str).tolist()
            question = str(group.iloc[0]["question"])
            reference_answer = str(group.iloc[0]["reference_answer"])
            try:
                kwargs = {
                    "question": question,
                    "answers": answers,
                }
                if getattr(cfg.api_eval, "pass_reference_answer", True):
                    kwargs["reference_answer"] = reference_answer
                batch_result = grader_perq(**kwargs)
                labels = batch_result.predicted_labels

                # Align labels back to the dataframe rows
                for k, row_idx in enumerate(group.index):
                    if k < len(labels):
                        predicted_labels[index_to_pos[row_idx]] = int(labels[k])
            except Exception as e:
                tqdm.write(f"Error grading group: {e}")
                raise e

        labels = compute_labels_list(answers_df)

    # Calculate metrics
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(
        labels,
        predicted_labels,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        labels,
        predicted_labels,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        labels,
        predicted_labels,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    # Confusion matrix
    unique_labels = sorted(list(set(labels + predicted_labels)))
    cm = confusion_matrix(labels, predicted_labels, labels=unique_labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "predicted_labels": predicted_labels,
        "labels": labels,
    }


"""
Main evaluation pipeline
"""

# Load config and paths
base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
eval_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "evaluation.yaml")
cfg = OmegaConf.merge(base_cfg, eval_cfg)
output_dir = os.path.join(PROJECT_ROOT, cfg.output.dir)

print(f"Using model {cfg.api_eval.model} for evaluation")

# Build LM and Grader program(s)
grader_lm = build_lm(
    cfg.api_eval.model,
    max_tokens=16000,
    cache=cfg.api_eval.cache,
    temperature=cfg.api_eval.temperature,
)

if cfg.api_eval.chain_of_thought:
    grader_single = dspy.ChainOfThought(GraderSingle)
    grader_single.set_lm(grader_lm)
    grader_perq = dspy.ChainOfThought(GraderPerQuestion)
    grader_perq.set_lm(grader_lm)
else:
    grader_single = dspy.Predict(GraderSingle)
    grader_single.set_lm(grader_lm)
    grader_perq = dspy.Predict(GraderPerQuestion)
    grader_perq.set_lm(grader_lm)

if cfg.api_eval.data.infer_path:
    # Load generated answers CSV based on config-driven naming
    generated_filename = f"student_answers_c{cfg.api_eval.data.num_correct_answers}_p{cfg.api_eval.data.num_partial_answers}_i{cfg.api_eval.data.num_incorrect_answers}_{cfg.api_eval.data.generation_model}_{cfg.api_eval.data.generation_mode}.csv"
    generated_path = os.path.join(output_dir, generated_filename)
else:
    generated_path = os.path.join(PROJECT_ROOT, cfg.api_eval.data.manual_path)

if not os.path.exists(generated_path):
    raise FileNotFoundError(f"Dataset CSV not found at: {generated_path}")
student_answers_df = pd.read_csv(generated_path, sep=";")

# Evaluate
print("\n" + "=" * 50)
print("EVALUATING GRADER PERFORMANCE")
print("=" * 50)
eval_mode = getattr(cfg.api_eval, "mode", "single")
metrics = evaluate_grader_performance(
    student_answers_df,
    grader_single,
    mode=eval_mode,
    grader_perq=grader_perq,
)

# Display results
print("\n" + "=" * 50)
print("GRADER PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision (macro): {metrics['precision']:.3f}")
print(f"Recall (macro): {metrics['recall']:.3f}")
print(f"F1 Score (macro): {metrics['f1_score']:.3f}")

# Classification summary: expected vs predicted counts and failures
label_names = {0: "incorrect", 1: "partial", 2: "correct"}
labels = metrics["labels"]
predicted = metrics["predicted_labels"]

total = len(labels)
misclassified_total = sum(1 for i, p in zip(labels, predicted) if p != i)
invalid_predictions = sum(1 for p in predicted if p not in [0, 1, 2])

print("\nClassification details")
print(f"Total examples: {total}")
print(f"Misclassified (predicted != labels): {misclassified_total}")
if invalid_predictions > 0:
    print(f"Invalid predictions (not in [0, 1, 2]): {invalid_predictions}")

for c in [0, 1, 2]:
    expected_c = sum(1 for i in labels if i == c)
    predicted_c = sum(1 for p in predicted if p == c)
    misclassified_c = sum(1 for i, p in zip(labels, predicted) if i == c and p != c)
    print(
        f"Class '{label_names[c]}' ({c}) -> expected: {expected_c}, predicted: {predicted_c}, misclassified: {misclassified_c}"
    )

# Plot confusion matrix
mode_suffix = {"single": "single", "per_question": "perq"}.get(
    eval_mode, eval_mode
)
plot_filename = f"confusion_matrix_c{cfg.api_eval.data.num_correct_answers}_p{cfg.api_eval.data.num_partial_answers}_i{cfg.api_eval.data.num_incorrect_answers}_{cfg.api_eval.data.generation_model}_{mode_suffix}_{cfg.api_eval.data.generation_mode}.png"
plot_path = os.path.join(output_dir, plot_filename)
plot_confusion_matrix(
    metrics["labels"],
    metrics["predicted_labels"],
    save_path=plot_path,
)
