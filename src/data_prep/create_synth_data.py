# %%
import logging
import os
import sys
from pathlib import Path

import dspy
import pandas as pd
from omegaconf import OmegaConf
from signatures import (
    CorrectAnswerGenerator,
    CorrectAnswerGeneratorAll,
    CorrectAnswerGeneratorPerQuestion,
    IncorrectAnswerGenerator,
    IncorrectAnswerGeneratorAll,
    IncorrectAnswerGeneratorPerQuestion,
    PartialAnswerGenerator,
    PartialAnswerGeneratorAll,
    PartialAnswerGeneratorPerQuestion,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


from src.model_builder import build_lm  # noqa: E402


logging.getLogger("dspy").setLevel(logging.ERROR)

tqdm.pandas(desc="Creating synthetic dataset")


# Configuration via YAML (OmegaConf)
base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
data_gen_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "data_generation.yaml")
cfg = OmegaConf.merge(base_cfg, data_gen_cfg)
output_dir = os.path.join(PROJECT_ROOT, cfg.input.synth_dir)

max_tokens = 4096 if cfg.generation.mode == "all" else 512

lm = build_lm(
    cfg.generation_model,
    temperature=getattr(cfg.lm, "temp_generation", None),
    cache=cfg.generation.cache,
    max_tokens=max_tokens,
)
dspy.settings.configure(lm=lm)

# read in data
tasks_file_path = os.path.join(output_dir, cfg.input.tasks_filename)
tasks = pd.read_csv(tasks_file_path, sep=";")


# ------------------------------
# Helpers to build kwargs/rows
# ------------------------------

DEFAULT_PASS_REFERENCE = {
    "correct": False,
    "partial": False,
    "incorrect": False,
}
DEFAULT_PASS_REFERENCE_ANSWER = {
    "correct": True,
    "partial": True,
    "incorrect": False,
}


def _should_pass_reference(label: str) -> bool:
    return bool(
        getattr(cfg.generation.pass_reference_for, label, DEFAULT_PASS_REFERENCE[label])
    )


def _should_pass_reference_answer(label: str) -> bool:
    return bool(
        getattr(
            cfg.generation.pass_reference_answer_for,
            label,
            DEFAULT_PASS_REFERENCE_ANSWER[label],
        )
    )


def _build_kwargs_single(
    label: str, question: str, chunk_text: str, reference_answer: str
):
    kwargs = {"question": question}
    if _should_pass_reference(label):
        kwargs["reference"] = chunk_text
    if _should_pass_reference_answer(label):
        kwargs["reference_answer"] = reference_answer
    return kwargs


def _build_kwargs_perq(
    label: str, question: str, chunk_text: str, reference_answer: str, count: int
):
    kwargs = _build_kwargs_single(label, question, chunk_text, reference_answer)
    kwargs["number_of_answers_per_question"] = count
    return kwargs


def _build_kwargs_all(
    label: str,
    questions_list,
    reference_texts_list,
    reference_answers_list,
    count: int,
):
    kwargs = {
        "questions": questions_list,
        "number_of_answers_per_question": count,
    }
    if _should_pass_reference(label):
        kwargs["references"] = reference_texts_list
    if _should_pass_reference_answer(label):
        kwargs["reference_answers"] = reference_answers_list
    return kwargs


def _append_rows(rows, idx, task, answers, label: str):
    question = task["question"]
    reference_answer = task["reference_answer"]
    chunk_text = task["chunk_text"]
    topic = task["topic"]
    for ans in answers:
        rows.append(
            {
                "task_id": idx,
                "question": question,
                "reference_answer": reference_answer,
                "chunk_text": chunk_text,
                "topic": topic,
                "student_answer": ans,
                "label": label,
            }
        )


def generate_student_answers_df(
    tasks_df, num_correct, num_partial, num_incorrect, models
):
    """
    Generate a dataframe with student answers for each question.

    Args:
        tasks_df: DataFrame containing questions and reference answers
        num_correct: Number of correct answers to generate per question
        num_partial: Number of partial answers to generate per question
        num_incorrect: Number of incorrect answers to generate per question

    Returns:
        DataFrame with columns: task_id, question, reference_answer, chunk_text, topic, student_answer, label
        - label: one of {"incorrect", "partial", "correct"}
    """
    all_answers = []

    label_counts = [
        ("correct", num_correct),
        ("partial", num_partial),
        ("incorrect", num_incorrect),
    ]

    for idx, task in tqdm(tasks_df.iterrows()):
        question = task["question"]
        reference_answer = task["reference_answer"]
        chunk_text = task["chunk_text"]

        for label, count in label_counts:
            if count <= 0:
                continue
            predictor = models.get(label)
            for i in range(count):
                try:
                    kwargs = _build_kwargs_single(
                        label, question, chunk_text, reference_answer
                    )
                    result = predictor(**kwargs)
                    ans = getattr(result, "answer", None)
                except Exception as e:
                    print(f"Error generating {label} answer for task {idx}: {e}")
                    if label == "correct":
                        ans = reference_answer
                    elif label == "partial":
                        ans = f"Partial answer {i + 1} for question {idx}"
                    else:
                        ans = f"Incorrect answer {i + 1} for question {idx}"
                _append_rows(all_answers, idx, task, [ans], label)

    return pd.DataFrame(all_answers)


# Per-question batched generation
def generate_student_answers_df_per_question(
    tasks_df, num_correct, num_partial, num_incorrect, models
):
    all_answers = []

    for idx, task in tqdm(tasks_df.iterrows()):
        question = task["question"]
        reference_answer = task["reference_answer"]
        chunk_text = task["chunk_text"]

        label_counts = [
            ("correct", num_correct),
            ("partial", num_partial),
            ("incorrect", num_incorrect),
        ]
        for label, count in label_counts:
            if count <= 0:
                continue
            predictor = models.get(label)
            try:
                kwargs = _build_kwargs_perq(
                    label, question, chunk_text, reference_answer, count
                )
                res = predictor(**kwargs)
                answers = res.answers
                if len(answers) >= count:
                    answers = answers[:count]
                elif len(answers) < count:
                    raise ValueError(
                        f"Expected {count} answers for label '{label}' on task {idx}, got {len(answers)}. "
                        f"Answers: {answers}"
                    )
                _append_rows(all_answers, idx, task, answers, label)
            except Exception as e:
                print(f"Error generating {label} answers for task {idx}: {e}")
                print(f"Answers: {answers}")
                print(f"\nFull output: {res}\n")
                print(f"Full kwargs: {kwargs}\n")
                continue

    return pd.DataFrame(all_answers)


# All-questions batched generation (3 total LLM calls)
def generate_student_answers_df_all(
    tasks_df, num_correct, num_partial, num_incorrect, models
):
    num_questions = len(tasks_df)
    assert num_questions > 0, "No questions to generate answers for"

    def group_flat(flat_answers, per_q):
        if per_q <= 0:
            return [[] for _ in range(num_questions)]
        flat = flat_answers if isinstance(flat_answers, list) else []
        expected = per_q * num_questions
        # Strict validation: do not pad or truncate
        if len(flat) != expected:
            print(
                f"Expected {expected} answers (per_q={per_q} * num_questions={num_questions}), got {len(flat)}."
            )
        grouped = []
        for i in range(num_questions):
            s = i * per_q
            e = s + per_q
            grouped.append(flat[s:e])
        return grouped

    questions_list = tasks_df["question"].astype(str).tolist()
    reference_answers_list = tasks_df["reference_answer"].astype(str).tolist()
    reference_texts_list = tasks_df["chunk_text"].astype(str).tolist()

    correct_flat = []
    partial_flat = []
    incorrect_flat = []

    try:
        if num_correct > 0:
            kwargs = _build_kwargs_all(
                "correct",
                questions_list,
                reference_texts_list,
                reference_answers_list,
                num_correct,
            )
            result = models["correct"](**kwargs)
            correct_flat = result.answers

        if num_partial > 0:
            kwargs = _build_kwargs_all(
                "partial",
                questions_list,
                reference_texts_list,
                reference_answers_list,
                num_partial,
            )
            result = models["partial"](**kwargs)
            partial_flat = result.answers

        if num_incorrect > 0:
            kwargs = _build_kwargs_all(
                "incorrect",
                questions_list,
                reference_texts_list,
                reference_answers_list,
                num_incorrect,
            )
            result = models["incorrect"](**kwargs)
            incorrect_flat = result.answers
    except Exception as e:
        raise e

    grouped_correct = group_flat(correct_flat, num_correct)
    grouped_partial = group_flat(partial_flat, num_partial)
    grouped_incorrect = group_flat(incorrect_flat, num_incorrect)

    rows = []
    for pos, (idx, task) in enumerate(tqdm(tasks_df.iterrows())):
        _append_rows(rows, idx, task, grouped_correct[pos], "correct")
        _append_rows(rows, idx, task, grouped_partial[pos], "partial")
        _append_rows(rows, idx, task, grouped_incorrect[pos], "incorrect")

    return pd.DataFrame(rows)


def _validate_generated_counts(
    student_answers_df: pd.DataFrame, tasks_df: pd.DataFrame, cfg
) -> None:
    """
    Validate that the number of generated answers matches the configuration exactly,
    both overall and per question for each label.
    """
    num_tasks = len(tasks_df)
    # Overall expected totals
    expected_per_label = {
        "correct": int(getattr(cfg.generation, "num_correct_answers", 0)),
        "partial": int(getattr(cfg.generation, "num_partial_answers", 0)),
        "incorrect": int(getattr(cfg.generation, "num_incorrect_answers", 0)),
    }
    expected_total = num_tasks * sum(expected_per_label.values())

    actual_total = len(student_answers_df)
    if actual_total != expected_total:
        print(
            f"Total generated answers {actual_total} != expected {expected_total} (num_tasks={num_tasks}, per_question={expected_per_label})."
        )

    # Overall counts per label
    for label, per_q in expected_per_label.items():
        expected_label_total = num_tasks * per_q
        actual_label_total = (student_answers_df["label"] == label).sum()
        if actual_label_total != expected_label_total:
            print(
                f"Generated {actual_label_total} '{label}' answers != expected {expected_label_total} (num_tasks={num_tasks} * per_q={per_q})."
            )

    # Per-question counts per label
    counts = (
        student_answers_df.groupby(["task_id", "label"]).size().unstack(fill_value=0)
    )
    for task_id in tasks_df.index:
        row = (
            counts.loc[task_id]
            if task_id in counts.index
            else {"correct": 0, "partial": 0, "incorrect": 0}
        )
        for label, per_q in expected_per_label.items():
            actual = (
                int(row.get(label, 0))
                if hasattr(row, "get")
                else int(row[label])
                if label in row
                else 0
            )
            if actual != per_q:
                raise ValueError(
                    f"Task {task_id}: generated {actual} '{label}' answers != expected {per_q}."
                )


total_per_question = (
    cfg.generation.num_correct_answers
    + cfg.generation.num_partial_answers
    + cfg.generation.num_incorrect_answers
)
print(f"Generating {total_per_question} student answers per question...")
print(f"Using mode: {cfg.generation.mode} with model: {cfg.generation_model}")
print(
    f"Per-question targets -> correct: {cfg.generation.num_correct_answers}, partial: {cfg.generation.num_partial_answers}, incorrect: {cfg.generation.num_incorrect_answers}"
)

if cfg.generation.mode == "single":
    if cfg.generation.chain_of_thought:
        models_single = {
            "correct": dspy.ChainOfThought(CorrectAnswerGenerator),
            "partial": dspy.ChainOfThought(PartialAnswerGenerator),
            "incorrect": dspy.ChainOfThought(IncorrectAnswerGenerator),
        }
    else:
        models_single = {
            "correct": dspy.Predict(CorrectAnswerGenerator),
            "partial": dspy.Predict(PartialAnswerGenerator),
            "incorrect": dspy.Predict(IncorrectAnswerGenerator),
        }

    student_answers_df = generate_student_answers_df(
        tasks,
        cfg.generation.num_correct_answers,
        cfg.generation.num_partial_answers,
        cfg.generation.num_incorrect_answers,
        models_single,
    )
elif cfg.generation.mode == "per_question":
    if cfg.generation.chain_of_thought:
        models_perq = {
            "correct": dspy.ChainOfThought(CorrectAnswerGeneratorPerQuestion),
            "partial": dspy.ChainOfThought(PartialAnswerGeneratorPerQuestion),
            "incorrect": dspy.ChainOfThought(IncorrectAnswerGeneratorPerQuestion),
        }
    else:
        models_perq = {
            "correct": dspy.Predict(CorrectAnswerGeneratorPerQuestion),
            "partial": dspy.Predict(PartialAnswerGeneratorPerQuestion),
            "incorrect": dspy.Predict(IncorrectAnswerGeneratorPerQuestion),
        }

    student_answers_df = generate_student_answers_df_per_question(
        tasks,
        cfg.generation.num_correct_answers,
        cfg.generation.num_partial_answers,
        cfg.generation.num_incorrect_answers,
        models_perq,
    )
elif cfg.generation.mode == "all":
    if cfg.generation.chain_of_thought:
        models_all = {
            "correct": dspy.ChainOfThought(CorrectAnswerGeneratorAll),
            "partial": dspy.ChainOfThought(PartialAnswerGeneratorAll),
            "incorrect": dspy.ChainOfThought(IncorrectAnswerGeneratorAll),
        }
    else:
        models_all = {
            "correct": dspy.Predict(CorrectAnswerGeneratorAll),
            "partial": dspy.Predict(PartialAnswerGeneratorAll),
            "incorrect": dspy.Predict(IncorrectAnswerGeneratorAll),
        }

    student_answers_df = generate_student_answers_df_all(
        tasks,
        cfg.generation.num_correct_answers,
        cfg.generation.num_partial_answers,
        cfg.generation.num_incorrect_answers,
        models_all,
    )
else:
    raise ValueError(f"Invalid create_mode: {cfg.generation.mode}")

# Validate exact counts before proceeding
_validate_generated_counts(student_answers_df, tasks, cfg)

print(f"Generated {len(student_answers_df)} total student answers")
num_correct = (student_answers_df["label"] == "correct").sum()
num_partial = (student_answers_df["label"] == "partial").sum()
num_incorrect = (student_answers_df["label"] == "incorrect").sum()
print(f"Intended correct answers: {num_correct}")
print(f"Intended partial answers: {num_partial}")
print(f"Intended incorrect answers: {num_incorrect}")

# Save the dataframe
student_answers_filename = f"student_answers_c{cfg.generation.num_correct_answers}_p{cfg.generation.num_partial_answers}_i{cfg.generation.num_incorrect_answers}_{cfg.generation_model}_{cfg.generation.mode}.csv"
output_path = os.path.join(output_dir, student_answers_filename)
student_answers_df.to_csv(output_path, index=False, sep=";")
print(f"Saved student answers to: {output_path}")

# """
# Sample a few random incorrect student answers to display
# """
# incorrect_df = student_answers_df[student_answers_df["label"] == "incorrect"]
# sample_n = min(5, len(incorrect_df))
# if sample_n > 0:
#     incorrect_examples = incorrect_df.sample(n=sample_n, random_state=42)
#     for idx, example in incorrect_examples.iterrows():
#         print("\nSampled Incorrect Example:")
#         print("question: ", example["question"])
#         print("reference_answer: ", example["reference_answer"])
#         print("chunk_text: ", example["chunk_text"])
#         print("topic: ", example["topic"])
#         print("student_answer: ", example["student_answer"])
