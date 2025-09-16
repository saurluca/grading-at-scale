# %%
import os
import pandas as pd
import dspy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from dotenv import load_dotenv
from types import SimpleNamespace
load_dotenv()

cfg = SimpleNamespace(**{})


# Configuration
cfg.output_dir = os.path.join(os.path.dirname(__file__), "../", "data", "privacy")

# Non-cfg configuration values
N_STUDENTS_ANSWERS_PER_QUESTION = 6
PERCENTILE_CORRECT = 0.5
TASKS_FILENAME = "privacy_data.csv"
# student_lm = dspy.LM(
#     "ollama_chat/llama3.2:3b",
#     api_base="http://localhost:11434",
#     api_key="",
#     temperature=0.5,
#     max_tokens=50,
# )
student_lm = dspy.LM(
    "azure/gpt-4o-mini",
    api_base=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2024-12-01-preview",
    temperature=1.5,
    cache=False,
    # max_tokens=50,
)
teacher_lm = dspy.LM(
    "azure/gpt-4o",
    api_base=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2024-12-01-preview",
)

# read in data
tasks_file_path = os.path.join(cfg.output_dir, TASKS_FILENAME)
tasks = pd.read_csv(tasks_file_path)

# %%


class Grader(dspy.Signature):
    """You are a university professor for an introduction to Privacy and Anonmyisation class at university.
    Your job is to grade privacy exercises and decide if the students answered correctly (true or false).
    Answer based on the provided reference answer and reference text.
    """

    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The ground truth reference text")
    answer: str = dspy.InputField(description="The student answer")

    label: bool = dspy.OutputField(
        description="True if the student answer is correct, False if the student answer is incorrect"
    )


class CorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A correct student answer that demonstrates understanding of the question. The answer should be accurate and well-reasoned."
    )


class IncorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="An incorrect student answer that shows misunderstanding or error in reasoning. The answer should be plausible but definitly wrong."
    )


# Create DSPy programs
grader = dspy.Predict(Grader)
correct_answer_generator = dspy.Predict(CorrectAnswerGenerator)
incorrect_answer_generator = dspy.Predict(IncorrectAnswerGenerator)

grader.set_lm(teacher_lm)

# Set the student LM for answer generation
correct_answer_generator.set_lm(student_lm)
incorrect_answer_generator.set_lm(student_lm)

# %%


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
        xticklabels=["Incorrect", "Correct"],
        yticklabels=["Incorrect", "Correct"],
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Grader Performance")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to: {save_path}")

    plt.show()


def generate_student_answers_df(tasks_df, n_answers_per_question, correct_percentile):
    """
    Generate a dataframe with student answers for each question.

    Args:
        tasks_df: DataFrame containing questions and reference answers
        n_answers_per_question: Number of student answers to generate per question
        correct_percentile: Percentile of answers that should be correct (0.0 to 1.0)

    Returns:
        DataFrame with columns: task_id, question, reference, student_answer, intended_correct
    """
    all_answers = []

    for idx, task in tqdm(tasks_df.iterrows()):
        question = task["question"]
        reference = task["chunk_text"]

        # Calculate number of correct and incorrect answers
        n_correct = int(n_answers_per_question * correct_percentile)
        n_incorrect = n_answers_per_question - n_correct

        # print(
        #     f"Generating answers for task {idx}: {n_correct} correct, {n_incorrect} incorrect"
        # )

        # Generate correct answers using the correct answer generator
        for i in range(n_correct):
            try:
                generated_result = correct_answer_generator(
                    question=question, reference=reference
                )
                student_answer = generated_result.answer
            except Exception as e:
                print(f"Error generating correct answer for task {idx}: {e}")
                # Fallback to reference answer
                student_answer = reference

            all_answers.append(
                {
                    "task_id": idx,
                    "question": question,
                    "reference": reference,
                    "student_answer": student_answer,
                    "intended_correct": True,
                }
            )

        # Generate incorrect answers using the incorrect answer generator
        for i in range(n_incorrect):
            try:
                generated_result = incorrect_answer_generator(
                    question=question, reference=reference
                )
                student_answer = generated_result.answer
            except Exception as e:
                print(f"Error generating incorrect answer for task {idx}: {e}")
                # Fallback if generation fails
                student_answer = f"Incorrect answer {i + 1} for question {idx}"

            all_answers.append(
                {
                    "task_id": idx,
                    "question": question,
                    "reference": reference,
                    "student_answer": student_answer,
                    "intended_correct": False,
                }
            )

    return pd.DataFrame(all_answers)


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

# Generate student answers dataframe
print(f"Generating {N_STUDENTS_ANSWERS_PER_QUESTION} student answers per question...")
print(f"Target correct percentage: {PERCENTILE_CORRECT * 100}%")

student_answers_df = generate_student_answers_df(
    tasks, N_STUDENTS_ANSWERS_PER_QUESTION, PERCENTILE_CORRECT
)

print(f"Generated {len(student_answers_df)} total student answers")
print(f"Intended correct answers: {student_answers_df['intended_correct'].sum()}")
print(f"Intended incorrect answers: {(~student_answers_df['intended_correct']).sum()}")

# Save the dataframe
student_answers_filename = "student_answers.csv"
output_path = os.path.join(cfg.output_dir, student_answers_filename)
student_answers_df.to_csv(output_path, index=False)
print(f"Saved student answers to: {output_path}")

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
plot_filename = "confusion_matrix.png"
plot_confusion_matrix(
    metrics["intended_correct"],
    metrics["predicted_correct"],
    save_path=os.path.join(cfg.output_dir, plot_filename),
)

# Add predicted correctness to the dataframe
student_answers_df["predicted_correct"] = metrics["predicted_correct"]

# Save the complete dataframe with predictions
complete_output_filename = "student_answers_with_predictions.csv"
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
# for idx, row in sample_results.iterrows():
#     print(
#         f"Task {row['task_id']}: Intended={row['intended_correct']}, Predicted={row['predicted_correct']}"
#     )
#     print(f"  Question: {row['question'][:100]}...")
#     print(f"  Answer: {row['student_answer'][:100]}...")
#     print()
