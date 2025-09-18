# %%
import os
import sys
from pathlib import Path
import pandas as pd
import dspy
from tqdm import tqdm
from model_builder import build_lm

# Ensure project root is on sys.path for absolute imports (works in scripts and notebooks)
if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config

# Configuration via YAML (OmegaConf)
cfg = load_config("synthetic_data")
output_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../", cfg.output_dir)
)

student_lm = build_lm(
    cfg.model_name,
    temperature=getattr(cfg, "lm_temperature", 0.8),
    cache=getattr(cfg, "lm_cache", False),
)

# read in data
tasks_file_path = os.path.join(output_dir, cfg.tasks_filename)
tasks = pd.read_csv(tasks_file_path)

# %%


class CorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    # reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A short correct student answer that demonstrates understanding of the question. The answer should be accurate and well-reasoned."
    )


class PartialAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    # reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A short partially correct student answer that demonstrates understanding of the question but is wrong in some kind of way. Your goal is to get half the points."
    )


class IncorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    # reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="An shortincorrect student answer that shows misunderstanding or error in reasoning. The answer should be plausible but wrong."
    )


# Create DSPy programs
correct_answer_generator = dspy.Predict(CorrectAnswerGenerator)
partial_answer_generator = dspy.Predict(PartialAnswerGenerator)
incorrect_answer_generator = dspy.Predict(IncorrectAnswerGenerator)

# Set the student LM for answer generation
correct_answer_generator.set_lm(student_lm)
partial_answer_generator.set_lm(student_lm)
incorrect_answer_generator.set_lm(student_lm)


# %%
def generate_student_answers_df(tasks_df, num_correct, num_partial, num_incorrect):
    """
    Generate a dataframe with student answers for each question.

    Args:
        tasks_df: DataFrame containing questions and reference answers
        num_correct: Number of correct answers to generate per question
        num_partial: Number of partial answers to generate per question
        num_incorrect: Number of incorrect answers to generate per question

    Returns:
        DataFrame with columns: task_id, question, reference, student_answer, intended_label
        - intended_label: one of {"incorrect", "partial", "correct"}
    """
    all_answers = []

    for idx, task in tqdm(tasks_df.iterrows()):
        question = task["question"]
        reference = task["chunk_text"]

        # Generate correct answers using the correct answer generator
        for i in range(num_correct):
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
                    "intended_label": "correct",
                }
            )

        # Generate partial answers using the partial answer generator
        for i in range(num_partial):
            try:
                generated_result = partial_answer_generator(
                    question=question, reference=reference
                )
                student_answer = generated_result.answer
            except Exception as e:
                print(f"Error generating partial answer for task {idx}: {e}")
                # Fallback if generation fails
                student_answer = f"Partial answer {i + 1} for question {idx}"

            all_answers.append(
                {
                    "task_id": idx,
                    "question": question,
                    "reference": reference,
                    "student_answer": student_answer,
                    "intended_label": "partial",
                }
            )

        # Generate incorrect answers using the incorrect answer generator
        for i in range(num_incorrect):
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
                    "intended_label": "incorrect",
                }
            )

    return pd.DataFrame(all_answers)


# %%

total_per_question = (
    cfg.num_correct_answers + cfg.num_partial_answers + cfg.num_incorrect_answers
)
print(f"Generating {total_per_question} student answers per question...")
print(
    f"Per-question targets -> correct: {cfg.num_correct_answers}, partial: {cfg.num_partial_answers}, incorrect: {cfg.num_incorrect_answers}"
)

student_answers_df = generate_student_answers_df(
    tasks,
    cfg.num_correct_answers,
    cfg.num_partial_answers,
    cfg.num_incorrect_answers,
)

print(f"Generated {len(student_answers_df)} total student answers")
num_correct = (student_answers_df["intended_label"] == "correct").sum()
num_partial = (student_answers_df["intended_label"] == "partial").sum()
num_incorrect = (student_answers_df["intended_label"] == "incorrect").sum()
print(f"Intended correct answers: {num_correct}")
print(f"Intended partial answers: {num_partial}")
print(f"Intended incorrect answers: {num_incorrect}")

# Save the dataframe
student_answers_filename = f"student_answers_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}.csv"
output_path = os.path.join(output_dir, student_answers_filename)
student_answers_df.to_csv(output_path, index=False)
print(f"Saved student answers to: {output_path}")

# %%

"""
Sample a few random incorrect student answers to display
"""
incorrect_df = student_answers_df[student_answers_df["intended_label"] == "incorrect"]
sample_n = min(5, len(incorrect_df))
if sample_n > 0:
    incorrect_examples = incorrect_df.sample(n=sample_n, random_state=42)
    for idx, example in incorrect_examples.iterrows():
        print("\nSampled Incorrect Example:")
        print("question: ", example["question"])
        print("answer: ", example["student_answer"])
