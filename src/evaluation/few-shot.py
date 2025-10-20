# %%
import logging
import os
import sys
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from signatures import GraderSingle, GraderSingle_without_prompt
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
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    label_names = {0: "Incorrect", 1: "Partially Correct", 2: "Correct"}
    label_display_names = [
        label_names.get(label, f"Label {label}") for label in unique_labels
    ]

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
    grader=None,
):
    """
    Evaluate the grader's performance on the generated answers.

    Args:
        answers_df: DataFrame with student answers and labels
        grader: The Grader instance for single evaluations

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating grader performance (mode=single)...")

    label_name_to_int = {
        "incorrect": 0,
        "partial": 1,
        "partially correct": 1,
        "correct": 2,
    }

    def grade_row(row):
        try:
            kwargs = {
                "question": row["question"],
                "answer": row["student_answer"],
            }
            if (
                getattr(cfg, "pass_reference", False)
                and "chunk_text" in row.index
            ):
                kwargs["reference"] = row["chunk_text"]
            if getattr(cfg, "pass_reference_answer", True):
                kwargs["reference_answer"] = row["reference_answer"]
            result = grader(**kwargs)
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


def convert_df_to_dspy_format(dataframe, include_reference: bool = False, include_reference_answer: bool = False):
    """Convert DataFrame rows to DSPy Example objects."""
    examples = []
    for _, row in dataframe.iterrows():
        # Create DSPy Example with inputs and targets
        target = {
            "label": row["labels"],
        }
        inputs = {
            "question": row["question"],
            "answer": row["student_answer"],
        }
        
        # conditionally include reference and reference answer
        if include_reference:
            inputs["reference"] = row["chunk_text"]
        if include_reference_answer:
            inputs["reference_answer"] = row["reference_answer"]
            
        # create example with inputs and targets
        example = dspy.Example(**inputs, **target).with_inputs(**inputs.keys())
        examples.append(example)
    return examples


def metric(gold, pred, trace=None):
    # gold is a DSPy Example with targets, pred is the model output
    
    generated_answer = int(pred.label)  # Changed from pred.labels to pred.label
    correct_answer = int(gold.label)
    
    # raise Exception(f"generated_answer: {generated_answer}, correct_answer: {correct_answer}")

    # Check for NaN values first
    if pd.isna(correct_answer) or pd.isna(generated_answer):
        return 0.0

    # For single sample comparison, use simple accuracy instead of Cohen's kappa
    if generated_answer == correct_answer:
        return 1.0
    else:
        return 0.0

"""
Main evaluation pipeline
"""

# Load config and paths
base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
few_shot_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "few_shot.yaml")
cfg = OmegaConf.merge(base_cfg, few_shot_cfg)
output_dir = os.path.join(PROJECT_ROOT, cfg.output.dir)

print(f"Using model {cfg.model.base} for evaluation")

# Build LM and Grader program(s)
grader_lm = build_lm(
    cfg.model.base,
    max_tokens=cfg.model.max_tokens,
    cache=cfg.model.cache,
    temperature=cfg.model.temperature,
)

print(f"Using LM {cfg.model.base}")

if cfg.evaluation.with_prompt:
    grader = dspy.Predict(GraderSingle)
else:
    grader = dspy.Predict(GraderSingle_without_prompt)

dspy.configure(lm=grader_lm)
grader.set_lm(grader_lm)


train_csv_path = os.path.join(PROJECT_ROOT, cfg.dataset.csv_train)
test_csv_path = os.path.join(PROJECT_ROOT, cfg.dataset.csv_test)

print(f"Loading train set from {train_csv_path}")
train_df = pd.read_csv(train_csv_path, sep=";")
print(f"Loading test set from {test_csv_path}")
test_df = pd.read_csv(test_csv_path, sep=";")

if cfg.dataset.n_train_examples is not None:
    train_df = train_df.sample(n=cfg.dataset.n_train_examples, random_state=42)
if cfg.dataset.n_test_examples is not None:
    test_df = test_df.sample(n=cfg.dataset.n_test_examples, random_state=42)

print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Convert DataFrame to DSPy format
trainset = convert_df_to_dspy_format(train_df, include_reference=cfg.pass_reference, include_reference_answer=cfg.pass_reference_answer)
testset = convert_df_to_dspy_format(test_df, include_reference=cfg.pass_reference, include_reference_answer=cfg.pass_reference_answer)


# %%
evaluator = Evaluate(
    devset=testset, num_threads=16, display_progress=True, display_table=False
)

# Launch evaluation with DSPy
evaluator(grader, metric=metric)


# %%

# improve the grader with few-shot optimization

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) examples of your program's steps.
# The optimizer will repeat this multiple times before selecting its best attempt on the devset.
# config = dict(
#     max_bootstrapped_demos=4,
#     max_labeled_demos=4,
#     num_candidate_programs=4,
#     num_threads=8,
# )

# teleprompter = dspy.BootstrapFewShotWithRandomSearch(metric=metric, **config)

# simba = dspy.teleprompt.SIMBA(metric=metric, max_steps=12, max_demos=10)
# labeled_few_shot = dspy.teleprompt.LabeledFewShot(k=8)

# grader = labeled_few_shot.compile(grader, trainset=trainset)


# optimizer_b = dspy.BootstrapFewShot(
#     metric=metric,
#     max_bootstrapped_demos=4,
#     max_labeled_demos=3,
# )
# grader = optimizer_b.compile(grader, trainset=trainset)
# grader.save("optimized_grader.json")

# Evaluate optimized grader with DSPy
# evaluator(grader, metric=metric)

# %%


if cfg.evaluation.manual:
    # Evaluate
    print("\n" + "=" * 50)
    print("EVALUATING GRADER PERFORMANCE")
    print("=" * 50)
    metrics = evaluate_grader_performance(
        test_df,
        grader=grader,
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
    mode_suffix = {"per_question": "perq"}.get(
        "per_question", "per_question"
    )
    plot_filename = f"confusion_matrix_per_question_{cfg.model.base}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plot_confusion_matrix(
        metrics["labels"],
        metrics["predicted_labels"],
        save_path=plot_path,
    )
