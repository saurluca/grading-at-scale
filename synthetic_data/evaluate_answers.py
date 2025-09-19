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
from typing import List, Optional

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


class GraderSingle(dspy.Signature):
    """
    You are a university professor for a introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Answer based on the provided reference answer.
    Return the corrsponding integer label for the grading, 0 for incorrect, 1 for partially correct, 2 for correct.
    """

    question: str = dspy.InputField(description="The question text")
    reference: Optional[str] = dspy.InputField(
        description="The ground truth reference text", optional=True
    )
    reference_answer: str = dspy.InputField(
        description="The ground truth reference answer"
    )
    answer: str = dspy.InputField(description="The student answer")

    label: int = dspy.OutputField(
        description="2 if the student answer is correct, 1 if the student answer is partially correct, 0 if the student answer is incorrect"
    )


class GraderPerQuestion(dspy.Signature):
    """
    You are a university professor for a introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Answer based on the provided reference answer.
    Return the corrsponding integer label for the grading, 0 for incorrect, 1 for partially correct, 2 for correct.
    """

    question: str = dspy.InputField(description="The question text")
    reference_answer: str = dspy.InputField(
        description="The ground truth reference answer"
    )
    answers: List[str] = dspy.InputField(description="The list of student answers")

    predicted_labels: List[int] = dspy.OutputField(
        description="Your labels for the provided answers, 0 for incorrect, 1 for partially correct, 2 for correct"
    )


class GraderAll(dspy.Signature):
    """
    You are a university professor for a introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Answer based on the provided reference answer.
    Return the corrsponding integer label for the grading, 0 for incorrect, 1 for partially correct, 2 for correct.
    """

    questions: List[str] = dspy.InputField(description="Unique questions list")
    reference_answers: List[str] = dspy.InputField(
        description="Correct reference answers aligned with questions, same order as questions"
    )
    counts_per_question: List[int] = dspy.InputField(
        description="Number of student answers for each question (same order as questions)"
    )
    answers_flat: List[str] = dspy.InputField(
        description="All student answers flattened in question-major order"
    )

    labels_flat: List[int] = dspy.OutputField(
        description="Labels flattened in question-major order, 0 for incorrect, 1 for partially correct, 2 for correct"
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


def evaluate_grader_performance(
    answers_df,
    grader_single,
    mode: str = "single",
    grader_perq=None,
    grader_all=None,
):
    """
    Evaluate the grader's performance on the generated answers.

    Args:
        answers_df: DataFrame with student answers and intended labels
        grader_single: The Grader instance for single evaluations
        mode: One of {"single", "per_question", "all"}

    Returns:
        Dictionary with evaluation metrics
    """
    assert mode in {"single", "per_question", "all"}, f"Invalid eval mode: {mode}"
    print(f"Evaluating grader performance (mode={mode})...")

    label_name_to_int = {
        "incorrect": 0,
        "partial": 1,
        "partially correct": 1,
        "correct": 2,
    }

    def compute_intended_list(df: pd.DataFrame):
        vals = []
        for _, row in df.iterrows():
            val = row.get("intended_label", None)
            if isinstance(val, str):
                intended = label_name_to_int.get(val.strip().lower(), -1)
            elif pd.notna(val):
                intended = int(val)
            else:
                intended = -1
            vals.append(intended)
        return vals

    if mode == "single":

        def grade_row(row):
            try:
                result = grader_single(
                    question=row["question"],
                    # reference=row["chunk_text"],
                    reference_answer=row["reference_answer"],
                    answer=row["student_answer"],
                )
                predicted = int(result.label)
            except Exception as e:
                tqdm.write(f"Error grading answer: {e}")
                raise e

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
                batch_result = grader_perq(
                    question=question,
                    reference_answer=reference_answer,
                    answers=answers,
                )
                labels = batch_result.predicted_labels

                # Align labels back to the dataframe rows
                for k, row_idx in enumerate(group.index):
                    if k < len(labels):
                        predicted_labels[index_to_pos[row_idx]] = int(labels[k])
            except Exception as e:
                tqdm.write(f"Error grading group: {e}")
                raise e

        intended_labels = compute_intended_list(answers_df)

    else:  # mode == "all"
        # Build compact inputs without repetition
        group_key = "task_id" if "task_id" in answers_df.columns else "question"
        questions_unique = []
        references_unique = []
        counts_per_question = []
        answers_flat = []

        for _, group in answers_df.groupby(group_key):
            questions_unique.append(str(group.iloc[0]["question"]))
            references_unique.append(str(group.iloc[0]["reference_answer"]))
            group_answers = group["student_answer"].astype(str).tolist()
            counts_per_question.append(len(group_answers))
            answers_flat.extend(group_answers)

        try:
            res = grader_all(
                questions=questions_unique,
                reference_answers=references_unique,
                counts_per_question=counts_per_question,
                answers_flat=answers_flat,
            )
            labels_flat = res.labels_flat

            if len(labels_flat) < len(answers_flat):
                labels_flat = labels_flat + [-1] * (
                    len(answers_flat) - len(labels_flat)
                )

            # Map back into the original row order
            predicted_labels = []
            cursor = 0
            for _, group in answers_df.groupby(group_key):
                size = len(group)
                chunk = labels_flat[cursor : cursor + size]
                predicted_labels.extend([int(x) for x in chunk])
                cursor += size
        except Exception as e:
            tqdm.write(f"Error grading all answers in batch: {e}")
            raise e

        intended_labels = compute_intended_list(answers_df)

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


max_tokens = 8192 if cfg.eval_mode == "all" else 512

# Build LM and Grader program(s)
grader_lm = build_lm(
    cfg.teacher_model_name,
    max_tokens=max_tokens,
    # cache=False,
)

grader_single = dspy.Predict(GraderSingle)
grader_single.set_lm(grader_lm)
grader_perq = dspy.Predict(GraderPerQuestion)
grader_perq.set_lm(grader_lm)
grader_all = dspy.Predict(GraderAll)
grader_all.set_lm(grader_lm)

# Load generated answers CSV based on config-driven naming
generated_filename = f"student_answers_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}_{cfg.create_mode}.csv"
generated_path = os.path.join(output_dir, generated_filename)
if not os.path.exists(generated_path):
    raise FileNotFoundError(f"Generated answers CSV not found at: {generated_path}")

student_answers_df = pd.read_csv(generated_path)

# Evaluate
print("\n" + "=" * 50)
print("EVALUATING GRADER PERFORMANCE")
print("=" * 50)
eval_mode = getattr(cfg, "eval_mode", "single")
metrics = evaluate_grader_performance(
    student_answers_df,
    grader_single,
    mode=eval_mode,
    grader_perq=grader_perq,
    grader_all=grader_all,
)

# Display results
print("\n" + "=" * 50)
print("GRADER PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision (macro): {metrics['precision']:.3f}")
print(f"Recall (macro): {metrics['recall']:.3f}")
print(f"F1 Score (macro): {metrics['f1_score']:.3f}")

# Plot confusion matrix
mode_suffix = {"single": "single", "per_question": "perq", "all": "all"}.get(
    eval_mode, eval_mode
)
plot_filename = f"confusion_matrix_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}_{mode_suffix}_{cfg.create_mode}.png"
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
complete_output_filename = f"student_answers_with_predictions_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}_{mode_suffix}_{cfg.create_mode}.csv"
complete_output_path = os.path.join(output_dir, complete_output_filename)
student_answers_df.to_csv(complete_output_path, index=False)
print(f"\nSaved complete results to: {complete_output_path}")

# Sample a few rows for inspection
print("\n" + "=" * 50)
print("SAMPLE RESULTS")
print("=" * 50)
print(student_answers_df.head(10))
