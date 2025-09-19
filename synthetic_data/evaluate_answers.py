# %%
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm
import dspy
from model_builder import build_lm

# Ensure project root is on sys.path for absolute imports (works in scripts and notebooks)
if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config

# Enable nice tqdm integration with pandas
tqdm.pandas(desc="Grading answers")


class Grader(dspy.Signature):
    """
    You are a university professor for a introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Answer based on the provided reference answer.
    Return the corrsponding integer label for the grading, 0 for incorrect, 1 for partially correct, 2 for correct.
    """

    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(
        description="The ground truth reference text", optional=True
    )
    reference_answer: str = dspy.InputField(
        description="The ground truth reference answer"
    )
    answer: str = dspy.InputField(description="The student answer")

    label: int = dspy.OutputField(
        description="2 if the student answer is correct, 1 if the student answer is partially correct, 0 if the student answer is incorrect"
    )


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot a confusion matrix using seaborn's heatmap.

    Parameters:
    - y_true: List or array of true labels
    - y_pred: List or array of predicted labels
    - save_path: Optional path to save the plot
    """
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Incorrect", "Partially Correct", "Correct"],
        yticklabels=["Incorrect", "Partially Correct", "Correct"],
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Grader Performance")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to: {save_path}")

    plt.show()


def evaluate_grader_performance(answers_df, grader):
    """
    Evaluate the grader's performance on the generated answers.

    Args:
        answers_df: DataFrame with student answers and intended labels
        grader: The Grader instance

    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating grader performance...")

    label_name_to_int = {
        "incorrect": 0,
        "partial": 1,
        "partially correct": 1,
        "correct": 2,
    }

    def grade_row(row):
        try:
            result = grader(
                question=row["question"],
                # reference=row["chunk_text"],
                reference_answer=row["reference_answer"],
                answer=row["student_answer"],
            )
            predicted = int(result.label)
        except Exception as e:
            tqdm.write(f"Error grading answer: {e}")
            predicted = -1

        val = row.get("intended_label", None)
        if isinstance(val, str):
            intended = label_name_to_int.get(val.strip().lower(), 0)
        elif pd.notna(val):
            intended = int(val)
        else:
            intended = 0

        return {"predicted": predicted, "intended": intended}

    results = answers_df.progress_apply(grade_row, axis=1)
    predicted_labels = results.map(lambda d: d["predicted"]).tolist()
    intended_labels = results.map(lambda d: d["intended"]).tolist()

    # Calculate metrics
    labels = [0, 1, 2]
    accuracy = accuracy_score(intended_labels, predicted_labels)
    precision = precision_score(
        intended_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        intended_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        intended_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(intended_labels, predicted_labels, labels=labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "predicted_labels": predicted_labels,
        "intended_labels": intended_labels,
    }


"""
Main evaluation pipeline
"""

# Load config and paths
cfg = load_config("synthetic_data")
output_dir = os.path.normpath(
    os.path.join(Path(__file__).resolve().parent, "../", cfg.output_dir)
)

print(f"Using model {cfg.teacher_model_name} for evaluation")

# Build LM and Grader program
grader_lm = build_lm(
    cfg.teacher_model_name,
)
grader_program = dspy.Predict(Grader)
grader_program.set_lm(grader_lm)

# Load generated answers CSV based on config-driven naming
generated_filename = f"student_answers_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}.csv"
generated_path = os.path.join(output_dir, generated_filename)
if not os.path.exists(generated_path):
    raise FileNotFoundError(f"Generated answers CSV not found at: {generated_path}")

student_answers_df = pd.read_csv(generated_path)

# Evaluate
print("\n" + "=" * 50)
print("EVALUATING GRADER PERFORMANCE")
print("=" * 50)
metrics = evaluate_grader_performance(student_answers_df, grader_program)

# Display results
print("\n" + "=" * 50)
print("GRADER PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision (macro): {metrics['precision']:.3f}")
print(f"Recall (macro): {metrics['recall']:.3f}")
print(f"F1 Score (macro): {metrics['f1_score']:.3f}")

# Plot confusion matrix
plot_filename = f"confusion_matrix_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}.png"
plot_path = os.path.join(output_dir, plot_filename)
plot_confusion_matrix(
    metrics["intended_labels"],
    metrics["predicted_labels"],
    save_path=plot_path,
)

# Add predictions to DF and save
label_int_to_name = {0: "incorrect", 1: "partial", 2: "correct"}
student_answers_df["predicted_label"] = metrics["predicted_labels"]
student_answers_df["predicted_label_name"] = student_answers_df["predicted_label"].map(
    label_int_to_name
)

complete_output_filename = f"student_answers_with_predictions_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}.csv"
complete_output_path = os.path.join(output_dir, complete_output_filename)
student_answers_df.to_csv(complete_output_path, index=False)
print(f"\nSaved complete results to: {complete_output_path}")

# Sample a few rows for inspection
print("\n" + "=" * 50)
print("SAMPLE RESULTS")
print("=" * 50)
print(student_answers_df.head(10))
