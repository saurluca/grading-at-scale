# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import dspy




class Grader(dspy.Signature):
    """You are a university professor for a introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Answer based on the provided reference answer and reference text.
    """

    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The ground truth reference text")
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
    cm = confusion_matrix(y_true, y_pred)

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
        answers_df: DataFrame with student answers and intended correctness
        grader: The Grader instance

    Returns:
        Dictionary with evaluation metrics
    """
    predicted_correct = []
    intended_correct = []

    print("Evaluating grader performance...")

    for idx, row in tqdm(answers_df.iterrows()):
        try:
            # Get grader's prediction
            graded_result = grader(
                question=row["question"],
                reference=row["reference"],
                answer=row["student_answer"],
            )

            predicted_correct.append(graded_result.label)
            intended_correct.append(row["intended_correct"])

            if idx % 10 == 0:  # Progress indicator
                print(f"Processed {idx + 1}/{len(answers_df)} answers")

        except Exception as e:
            print(f"Error grading answer {idx}: {e}")
            # Default to incorrect if grading fails
            predicted_correct.append(False)
            intended_correct.append(row["intended_correct"])

    # Calculate metrics
    accuracy = accuracy_score(intended_correct, predicted_correct)
    precision = precision_score(intended_correct, predicted_correct, zero_division=0)
    recall = recall_score(intended_correct, predicted_correct, zero_division=0)
    f1 = f1_score(intended_correct, predicted_correct, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(intended_correct, predicted_correct)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "predicted_correct": predicted_correct,
        "intended_correct": intended_correct,
    }



# %%

# Evaluate grader performance
print("\n" + "=" * 50)
print("EVALUATING GRADER PERFORMANCE")
print("=" * 50)

metrics = evaluate_grader_performance(student_answers_df, grader)

# Display results
print("\n" + "=" * 50)
print("GRADER PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")

print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Correct  Incorrect")
print(
    f"Actual Correct   {metrics['confusion_matrix'][1][1]:>8}  {metrics['confusion_matrix'][1][0]:>9}"
)
print(
    f"Actual Incorrect {metrics['confusion_matrix'][0][1]:>8}  {metrics['confusion_matrix'][0][0]:>9}"
)

# Plot confusion matrix
plot_filename = f"confusion_matrix_{cfg.n_students_answers_per_question}_{cfg.percentile_correct}.png"
plot_confusion_matrix(
    metrics["intended_correct"],
    metrics["predicted_correct"],
    save_path=os.path.join(cfg.output_dir, plot_filename),
)

# Add predicted correctness to the dataframe
student_answers_df["predicted_correct"] = metrics["predicted_correct"]

# Save the complete dataframe with predictions
complete_output_filename = f"student_answers_with_predictions_{cfg.n_students_answers_per_question}_{cfg.percentile_correct}.csv"
complete_output_path = os.path.join(
    cfg.output_dir, complete_output_filename
)
student_answers_df.to_csv(complete_output_path, index=False)
print(f"\nSaved complete results to: {complete_output_path}")

# Display some example results
print("\n" + "=" * 50)
print("SAMPLE RESULTS")
print("=" * 50)
sample_results = student_answers_df.head(10)
