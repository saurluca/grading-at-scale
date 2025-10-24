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
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model_builder import build_lm  # noqa: E402

logging.getLogger("dspy").setLevel(logging.ERROR)

# Enable nice tqdm integration with pandas
tqdm.pandas(desc="Grading answers")

# Configure MLflow tracking (autolog will be called inside the run context)
mlflow.set_experiment("DSPy-Optimization")


def log_config_params_to_mlflow(cfg):
    """Log configuration parameters to MLflow, excluding paths and evaluation.manual."""
    # Parameters to exclude
    exclude_keys = {
        'csv_train', 'csv_test', 'dir',  # file paths and directories
        'manual'  # evaluation.manual
    }
    
    # Flatten the config and log parameters
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            # Handle OmegaConf objects and regular dicts
            if hasattr(v, 'items'):  # OmegaConf or dict-like object
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
    print(f"Parameters logged: {list(params_to_log.keys())}")


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
                getattr(cfg.model, "pass_reference", False)
                and "chunk_text" in row.index
            ):
                kwargs["reference"] = row["chunk_text"]
            if getattr(cfg.model, "pass_reference_answer", True):
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
        # Create combined dictionary with inputs and targets
        example_data = {
            "question": row["question"],
            "answer": row["student_answer"],
            "label": row["labels"],
        }
        
        # conditionally include reference and reference answer
        if include_reference:
            example_data["reference"] = row["chunk_text"]
        if include_reference_answer:
            example_data["reference_answer"] = row["reference_answer"]
        
        # Define input keys (everything except the target)
        input_keys = ["question", "answer"]
        if include_reference:
            input_keys.append("reference")
        if include_reference_answer:
            input_keys.append("reference_answer")
            
        # create example with combined data and specify input keys
        example = dspy.Example(**example_data).with_inputs(*input_keys)
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

# Support loading custom config via environment variable (for sweeps)
few_shot_config_path = os.environ.get(
    "FEW_SHOT_CONFIG",
    str(PROJECT_ROOT / "configs" / "few_shot.yaml")
)
few_shot_cfg = OmegaConf.load(few_shot_config_path)
cfg = OmegaConf.merge(base_cfg, few_shot_cfg)
output_dir = os.path.join(PROJECT_ROOT, cfg.output.dir)

print(f"Using model {cfg.model.base} for evaluation")

# Create run name based on model and configuration
model_name = cfg.model.base
model_short = model_name.split('/')[-1]
prompt_str = "prompt" if cfg.model.with_prompt else "noprompt"
ref_str = "ref" if cfg.model.pass_reference else "noref"
refans_str = "refans" if cfg.model.pass_reference_answer else "norefans"
run_name = f"few_shot_{model_short}_{prompt_str}_{ref_str}_{refans_str}"

# Start parent MLflow run
with mlflow.start_run(run_name=run_name) as run:
    # Enable DSPy autologging INSIDE the parent run
    # mlflow.dspy.autolog(
    #     log_compiles=True,
    #     log_evals=True,
    #     log_traces_from_compile=True
    # )
    
    # Log configuration parameters early
    log_config_params_to_mlflow(cfg)
    
    # Build LM and Grader program(s)
    grader_lm = build_lm(
        cfg.model.base,
        max_tokens=cfg.model.max_tokens,
        cache=cfg.model.cache,
        temperature=cfg.model.temperature,
    )

    print(f"Using LM {cfg.model.base}")

    if cfg.model.with_prompt:
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
        print(f"Sampling {cfg.dataset.n_train_examples} examples from train set")
        train_df = train_df.sample(n=cfg.dataset.n_train_examples, random_state=42)
    if cfg.dataset.n_test_examples is not None:
        print(f"Sampling {cfg.dataset.n_test_examples} examples from test set")
        test_df = test_df.sample(n=cfg.dataset.n_test_examples, random_state=42)

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Log datasets as MLflow Datasets
    train_ml_dataset = mlflow.data.from_pandas(
        train_df,
        source=train_csv_path,
        name="train_dataset"
    )
    mlflow.log_input(train_ml_dataset, context="training")
    
    test_ml_dataset = mlflow.data.from_pandas(
        test_df,
        source=test_csv_path,
        name="test_dataset"
    )
    mlflow.log_input(test_ml_dataset, context="evaluation")
    print("Successfully logged datasets as MLflow Datasets")

    # Convert DataFrame to DSPy format
    trainset = convert_df_to_dspy_format(train_df, include_reference=cfg.model.pass_reference, include_reference_answer=cfg.model.pass_reference_answer)
    testset = convert_df_to_dspy_format(test_df, include_reference=cfg.model.pass_reference, include_reference_answer=cfg.model.pass_reference_answer)


    # %%
    evaluator = Evaluate(
        devset=testset, num_threads=16, display_progress=True, display_table=False
    )

    # Launch evaluation with DSPy - autologging will track this within parent run
    result = evaluator(grader, metric=metric)
    
    # print(f"Evaluation result: {result}")
    
    mlflow.log_metric("evaluation_accuracy", result.score)
    
    print(f"Evaluation complete. MLflow run ID: {run.info.run_id}")

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
