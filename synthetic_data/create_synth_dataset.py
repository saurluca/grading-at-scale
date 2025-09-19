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

max_tokens = 4096 if cfg.create_mode == "all" else 512

student_lm = build_lm(
    cfg.model_name,
    temperature=getattr(cfg, "lm_temperature_creation", None),
    cache=False,
    max_tokens=max_tokens,
)

# read in data
tasks_file_path = os.path.join(output_dir, cfg.tasks_filename)
tasks = pd.read_csv(tasks_file_path)


class CorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A short correct student answer that demonstrates understanding of the question. The answer should be accurate and well-reasoned."
    )


class PartialAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A short partially correct student answer that demonstrates understanding of the question but is wrong."
    )


class IncorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    # reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A short incorrect student answer that shows misunderstanding or error in reasoning. Be creative. The answer should be plausible but wrong."
    )


# Per-question batched generators (many answers for one question per call)
class CorrectAnswerGeneratorPerQuestion(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference_answer: str = dspy.InputField(description="The correct reference answer")
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers to generate for this question"
    )
    answers: List[str] = dspy.OutputField(description="List of correct student answers")


class PartialAnswerGeneratorPerQuestion(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference_answer: str = dspy.InputField(description="The correct reference answer")
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers to generate for this question"
    )
    answers: List[str] = dspy.OutputField(
        description="List of partially correct student answers"
    )


class IncorrectAnswerGeneratorPerQuestion(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers to generate for this question"
    )
    answers: List[str] = dspy.OutputField(
        description="List of incorrect student answers"
    )


# All-questions batched generators (many answers for all questions in one call)
class CorrectAnswerGeneratorAll(dspy.Signature):
    questions: List[str] = dspy.InputField(description="The list of questions")
    # references: List[str] = dspy.InputField(description="Optional context texts")
    reference_answers: List[str] = dspy.InputField(
        description="The list of correct reference answers"
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers per question"
    )
    answers: List[str] = dspy.OutputField(
        description="Flat list of answers in question-major order"
    )


class PartialAnswerGeneratorAll(dspy.Signature):
    questions: List[str] = dspy.InputField(description="The list of questions")
    # references: List[str] = dspy.InputField(description="Optional context texts")
    reference_answers: List[str] = dspy.InputField(
        description="The list of correct reference answers"
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers per question"
    )
    answers: List[str] = dspy.OutputField(
        description="Flat list of answers in question-major order"
    )


class IncorrectAnswerGeneratorAll(dspy.Signature):
    questions: List[str] = dspy.InputField(description="The list of questions")
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers per question"
    )
    answers: List[str] = dspy.OutputField(
        description="Flat list of answers in question-major order"
    )


# Create DSPy programs
correct_answer_generator = dspy.Predict(CorrectAnswerGenerator)
partial_answer_generator = dspy.Predict(PartialAnswerGenerator)
incorrect_answer_generator = dspy.Predict(IncorrectAnswerGenerator)

correct_answer_generator_perq = dspy.Predict(CorrectAnswerGeneratorPerQuestion)
partial_answer_generator_perq = dspy.Predict(PartialAnswerGeneratorPerQuestion)
incorrect_answer_generator_perq = dspy.Predict(IncorrectAnswerGeneratorPerQuestion)

correct_answer_generator_all = dspy.Predict(CorrectAnswerGeneratorAll)
partial_answer_generator_all = dspy.Predict(PartialAnswerGeneratorAll)
incorrect_answer_generator_all = dspy.Predict(IncorrectAnswerGeneratorAll)

# Set the student LM for answer generation
correct_answer_generator.set_lm(student_lm)
partial_answer_generator.set_lm(student_lm)
incorrect_answer_generator.set_lm(student_lm)

# Increase max_tokens
correct_answer_generator_perq.set_lm(student_lm)
partial_answer_generator_perq.set_lm(student_lm)
incorrect_answer_generator_perq.set_lm(student_lm)

# Increase max_tokens
correct_answer_generator_all.set_lm(student_lm)
partial_answer_generator_all.set_lm(student_lm)
incorrect_answer_generator_all.set_lm(student_lm)


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
        DataFrame with columns: task_id, question, reference_answer, chunk_text, topic, student_answer, intended_label
        - intended_label: one of {"incorrect", "partial", "correct"}
    """
    all_answers = []

    for idx, task in tqdm(tasks_df.iterrows()):
        question = task["question"]
        reference_answer = task["answer"]
        chunk_text = task["chunk_text"]
        topic = task["topic"]

        # Generate correct answers using the correct answer generator
        for i in range(num_correct):
            try:
                generated_result = correct_answer_generator(
                    question=question, reference=reference_answer
                )
                student_answer = generated_result.answer
            except Exception as e:
                print(f"Error generating correct answer for task {idx}: {e}")
                # Fallback to reference answer
                student_answer = reference_answer

            all_answers.append(
                {
                    "task_id": idx,
                    "question": question,
                    "reference_answer": reference_answer,
                    "chunk_text": chunk_text,
                    "topic": topic,
                    "student_answer": student_answer,
                    "intended_label": "correct",
                }
            )

        # Generate partial answers using the partial answer generator
        for i in range(num_partial):
            try:
                generated_result = partial_answer_generator(
                    question=question, reference=reference_answer
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
                    "reference_answer": reference_answer,
                    "chunk_text": chunk_text,
                    "topic": topic,
                    "student_answer": student_answer,
                    "intended_label": "partial",
                }
            )

        # Generate incorrect answers using the incorrect answer generator
        for i in range(num_incorrect):
            try:
                generated_result = incorrect_answer_generator(
                    question=question, reference=reference_answer
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
                    "reference_answer": reference_answer,
                    "chunk_text": chunk_text,
                    "topic": topic,
                    "student_answer": student_answer,
                    "intended_label": "incorrect",
                }
            )

    return pd.DataFrame(all_answers)


# Per-question batched generation
def generate_student_answers_df_per_question(
    tasks_df, num_correct, num_partial, num_incorrect
):
    all_answers = []

    for idx, task in tqdm(tasks_df.iterrows()):
        question = task["question"]
        reference_answer = task["answer"]
        chunk_text = task["chunk_text"]
        topic = task["topic"]

        # Correct answers in one call
        if num_correct > 0:
            try:
                res = correct_answer_generator_perq(
                    question=question,
                    reference_answer=reference_answer,
                    number_of_answers_per_question=num_correct,
                )
                answers = list(getattr(res, "answers", []))
            except Exception as e:
                print(f"Error generating correct answers for task {idx}: {e}")
                answers = [reference_answer] * num_correct
            for ans in answers:
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

        # Partial answers in one call
        if num_partial > 0:
            try:
                res = partial_answer_generator_perq(
                    question=question,
                    reference_answer=reference_answer,
                    number_of_answers_per_question=num_partial,
                )
                answers = list(getattr(res, "answers", []))
            except Exception as e:
                print(f"Error generating partial answers for task {idx}: {e}")
                answers = [
                    f"Partial answer {i + 1} for question {idx}"
                    for i in range(num_partial)
                ]
            for ans in answers:
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

        # Incorrect answers in one call
        if num_incorrect > 0:
            try:
                res = incorrect_answer_generator_perq(
                    question=question,
                    number_of_answers_per_question=num_incorrect,
                )
                answers = list(getattr(res, "answers", []))
            except Exception as e:
                print(f"Error generating incorrect answers for task {idx}: {e}")
                answers = [
                    f"Incorrect answer {i + 1} for question {idx}"
                    for i in range(num_incorrect)
                ]
            for ans in answers:
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


# All-questions batched generation (3 total LLM calls)
def generate_student_answers_df_all(tasks_df, num_correct, num_partial, num_incorrect):
    num_questions = len(tasks_df)
    assert num_questions > 0, "No questions to generate answers for"

    def group_flat(flat_answers, per_q):
        if per_q <= 0:
            return [[] for _ in range(num_questions)]
        flat = flat_answers if isinstance(flat_answers, list) else []
        expected = per_q * num_questions
        if len(flat) < expected:
            flat = flat + [""] * (expected - len(flat))
        else:
            flat = flat[:expected]
        grouped = []
        for i in range(num_questions):
            s = i * per_q
            e = s + per_q
            grouped.append(flat[s:e])
        return grouped

    questions_list = tasks_df["question"].astype(str).tolist()
    reference_answers_list = tasks_df["answer"].astype(str).tolist()
    # reference_texts_list = tasks_df["chunk_text"].astype(str).tolist()

    correct_flat = []
    partial_flat = []
    incorrect_flat = []

    try:
        if num_correct > 0:
            r = correct_answer_generator_all(
                questions=questions_list,
                # references=reference_texts_list,
                reference_answers=reference_answers_list,
                number_of_answers_per_question=num_correct,
            )
            correct_flat = getattr(r, "answers", [])

        if num_partial > 0:
            r = partial_answer_generator_all(
                questions=questions_list,
                # references=reference_texts_list,
                reference_answers=reference_answers_list,
                number_of_answers_per_question=num_partial,
            )
            partial_flat = getattr(r, "answers", [])

        if num_incorrect > 0:
            r = incorrect_answer_generator_all(
                questions=questions_list,
                number_of_answers_per_question=num_incorrect,
            )
            incorrect_flat = getattr(r, "answers", [])
    except Exception as e:
        raise e

    grouped_correct = group_flat(correct_flat, num_correct)
    grouped_partial = group_flat(partial_flat, num_partial)
    grouped_incorrect = group_flat(incorrect_flat, num_incorrect)

    rows = []
    for pos, (idx, task) in enumerate(tqdm(tasks_df.iterrows())):
        question = task["question"]
        reference_answer = task["answer"]
        chunk_text = task["chunk_text"]
        topic = task["topic"]

        for ans in grouped_correct[pos]:
            rows.append(
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
        for ans in grouped_partial[pos]:
            rows.append(
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
        for ans in grouped_incorrect[pos]:
            rows.append(
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

    return pd.DataFrame(rows)


# %%

total_per_question = (
    cfg.num_correct_answers + cfg.num_partial_answers + cfg.num_incorrect_answers
)
print(f"Generating {total_per_question} student answers per question...")
print(f"Using mode: {cfg.create_mode} with model: {cfg.model_name}")
print(
    f"Per-question targets -> correct: {cfg.num_correct_answers}, partial: {cfg.num_partial_answers}, incorrect: {cfg.num_incorrect_answers}"
)

create_mode = getattr(cfg, "create_mode", "single")
if create_mode == "single":
    student_answers_df = generate_student_answers_df(
        tasks,
        cfg.num_correct_answers,
        cfg.num_partial_answers,
        cfg.num_incorrect_answers,
    )
elif create_mode == "per_question":
    student_answers_df = generate_student_answers_df_per_question(
        tasks,
        cfg.num_correct_answers,
        cfg.num_partial_answers,
        cfg.num_incorrect_answers,
    )
elif create_mode == "all":
    student_answers_df = generate_student_answers_df_all(
        tasks,
        cfg.num_correct_answers,
        cfg.num_partial_answers,
        cfg.num_incorrect_answers,
    )
else:
    raise ValueError(f"Invalid create_mode: {create_mode}")

print(f"Generated {len(student_answers_df)} total student answers")
num_correct = (student_answers_df["intended_label"] == "correct").sum()
num_partial = (student_answers_df["intended_label"] == "partial").sum()
num_incorrect = (student_answers_df["intended_label"] == "incorrect").sum()
print(f"Intended correct answers: {num_correct}")
print(f"Intended partial answers: {num_partial}")
print(f"Intended incorrect answers: {num_incorrect}")

# Save the dataframe
mode_suffix = {
    "single": "single",
    "per_question": "per_question",
    "all": "all",
}.get(create_mode, create_mode)
student_answers_filename = f"student_answers_c{cfg.num_correct_answers}_p{cfg.num_partial_answers}_i{cfg.num_incorrect_answers}_{cfg.model_name}_{mode_suffix}.csv"
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
