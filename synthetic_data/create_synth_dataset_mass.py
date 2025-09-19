# %%
import os
import sys
from pathlib import Path
import pandas as pd
import dspy
from tqdm import tqdm
from model_builder import build_lm
from typing import List


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
    temperature=getattr(cfg, "lm_temperature_creation", None),
    cache=False,
)

# read in data
tasks_file_path = os.path.join(output_dir, cfg.tasks_filename)
tasks = pd.read_csv(tasks_file_path)

# %%


class CorrectAnswerGenerator(dspy.Signature):
    """
    You are an excellent university student and you goal is to provide the correct answer to the questions provided.
    Generate a number of correct answers for a list of questions. The answer should be accurate, short and slightly different from each other.
    Aim to get exactly the full points.
    """

    questions: List[str] = dspy.InputField(description="The list of questions")
    # references: List[str] = dspy.InputField(
    #     description="A reference text containing the correct answer and useful information"
    # )
    reference_answers: List[str] = dspy.InputField(
        description="A list of correct reference answers"
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="The number of correct answers to generate per question"
    )
    answers: List[str] = dspy.OutputField(
        description="A list of short correct answers that demonstrate understanding of the question. The answers should be accurate and well-reasoned."
    )


class PartialAnswerGenerator(dspy.Signature):
    """
    You are a mediocre university student and you goal is to provide the partially correct answers to the questions provided.
    Generate a number of partial answers for a list of questions. The answer should be partially correct, short and different from each other.
    Aim to get exactly half the points.
    """

    questions: List[str] = dspy.InputField(description="The list of questions")
    # references: List[str] = dspy.InputField(
    #     description="A reference text containing the correct answer and useful information"
    # )
    reference_answers: List[str] = dspy.InputField(
        description="A list of correct reference answers"
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="The number of partially correct answers to generate per question"
    )
    answers: List[str] = dspy.OutputField(
        description="A list of short partially correct answers that demonstrate understanding of the question but are wrong."
    )


class IncorrectAnswerGenerator(dspy.Signature):
    """
    You are a very bad university student and you goal is to provide the wrong and incorrect answers to the questions provided.
    Generate a number of incorrect answers for a list of questions. The answer should be incorrect but plausible, short and different from each other.
    Some of you answers maybe be irrelevant, some contradictory and some just incorrect.
    """

    questions: List[str] = dspy.InputField(description="The list of questions")
    # reference: List[str] = dspy.InputField(
    #     description="A reference text containing the correct answer and useful information"
    # )
    # reference_answer: List[str] = dspy.InputField(description="A list of correct reference answers")
    number_of_answers_per_question: int = dspy.InputField(
        description="The number of incorrect answers to generate per question"
    )
    answers: List[str] = dspy.OutputField(
        description="A list of short incorrect answers that show misunderstanding or error in reasoning. Be creative. The answers should be plausible but wrong."
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
    Generate a dataframe with student answers for each question using batched LLM calls.

    Args:
        tasks_df: DataFrame containing questions and reference answers
        num_correct: Number of correct answers to generate per question
        num_partial: Number of partial answers to generate per question
        num_incorrect: Number of incorrect answers to generate per question

    Returns:
        DataFrame with columns: task_id, question, reference_answer, chunk_text, topic, student_answer, intended_label
        - intended_label: one of {"incorrect", "partial", "correct"}
    """

    def group_flat_answers(flat_answers, answers_per_question, num_questions):
        if answers_per_question == 0 or num_questions == 0:
            return [[] for _ in range(num_questions)]
        safe_flat = flat_answers if isinstance(flat_answers, list) else []
        expected_len = answers_per_question * num_questions
        if len(safe_flat) < expected_len:
            # pad with empty strings to maintain shape
            safe_flat = safe_flat + [""] * (expected_len - len(safe_flat))
        else:
            safe_flat = safe_flat[:expected_len]
        grouped = []
        for i in range(num_questions):
            start = i * answers_per_question
            end = start + answers_per_question
            grouped.append(safe_flat[start:end])
        return grouped

    num_questions = len(tasks_df)

    assert num_questions > 0, "No questions to generate answers for"
    assert num_correct >= 0, "Number of correct answers must be non-negative"
    assert num_partial >= 0, "Number of partial answers must be non-negative"
    assert num_incorrect >= 0, "Number of incorrect answers must be non-negative"

    questions_list = tasks_df["question"].astype(str).tolist()
    reference_texts_list = tasks_df["chunk_text"].astype(str).tolist()
    reference_answers_list = tasks_df["answer"].astype(str).tolist()

    # Batched generation calls (at most 3)
    correct_answers_flat = []
    partial_answers_flat = []
    incorrect_answers_flat = []

    try:
        correct_result = correct_answer_generator(
            questions=questions_list,
            references=reference_texts_list,
            reference_answers=reference_answers_list,
            number_of_answers_per_question=num_correct,
        )
        correct_answers_flat = getattr(correct_result, "answers", [])

        partial_result = partial_answer_generator(
            questions=questions_list,
            references=reference_texts_list,
            reference_answers=reference_answers_list,
            number_of_answers_per_question=num_partial,
        )
        partial_answers_flat = getattr(partial_result, "answers", [])

        incorrect_result = incorrect_answer_generator(
            questions=questions_list,
            number_of_answers_per_question=num_incorrect,
        )
        incorrect_answers_flat = getattr(incorrect_result, "answers", [])
    except Exception as e:
        raise e

    # Group answers back per question
    grouped_correct = group_flat_answers(
        correct_answers_flat, num_correct, num_questions
    )
    grouped_partial = group_flat_answers(
        partial_answers_flat, num_partial, num_questions
    )
    grouped_incorrect = group_flat_answers(
        incorrect_answers_flat, num_incorrect, num_questions
    )

    # Build the rows
    all_answers = []
    for position, (idx, task) in enumerate(tqdm(tasks_df.iterrows())):
        question = task["question"]
        reference_answer = task["answer"]
        chunk_text = task["chunk_text"]
        topic = task["topic"]

        # Correct answers for this question
        for ans in grouped_correct[position]:
            all_answers.append(
                {
                    "task_id": idx,
                    "question": question,
                    "reference_answer": reference_answer,
                    "chunk_text": chunk_text,
                    "topic": topic,
                    "student_answer": ans,
                    "intended_label": "correct",
                }
            )

        # Partial answers for this question
        for ans in grouped_partial[position]:
            all_answers.append(
                {
                    "task_id": idx,
                    "question": question,
                    "reference_answer": reference_answer,
                    "chunk_text": chunk_text,
                    "topic": topic,
                    "student_answer": ans,
                    "intended_label": "partial",
                }
            )

        # Incorrect answers for this question
        for ans in grouped_incorrect[position]:
            all_answers.append(
                {
                    "task_id": idx,
                    "question": question,
                    "reference_answer": reference_answer,
                    "chunk_text": chunk_text,
                    "topic": topic,
                    "student_answer": ans,
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
student_answers_filename = f"student_answers_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}_mass.csv"
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
        print("reference_answer: ", example["reference_answer"])
        print("chunk_text: ", example["chunk_text"])
        print("topic: ", example["topic"])
        print("student_answer: ", example["student_answer"])
