# %%
import os
import pandas as pd
import dspy
from tqdm import tqdm
from types import SimpleNamespace
from model_builder import build_lm

# Configuration
cfg = SimpleNamespace(**{})
cfg.output_dir = os.path.join(os.path.dirname(__file__), "../", "data", "privacy")
cfg.n_students_answers_per_question = 3
cfg.percentile_correct = 0.0
cfg.tasks_filename = "privacy_data.csv"

student_lm = build_lm("apertus-8b")
                    #   temperature=1.0,
                    #   cache=False)

# read in data
tasks_file_path = os.path.join(cfg.output_dir, cfg.tasks_filename)
tasks = pd.read_csv(tasks_file_path)

# %%

class Test(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    answer: str = dspy.OutputField(description="The answer to the question")
    
    
test_program = dspy.Predict(Test)
test_program.set_lm(student_lm)

output = test_program(question="What is the capital of France?")

print(output.answer)

# %%

class CorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A correct student answer that demonstrates understanding of the question. The answer should be accurate and well-reasoned."
    )
    
    
class PartialAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="A partially correct student answer that demonstrates understanding of the question but not the full answer."
    )


class IncorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The correct reference answer")
    answer: str = dspy.OutputField(
        description="An incorrect student answer that shows misunderstanding or error in reasoning. The answer should be plausible but wrong."
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


# %%

# Generate student answers dataframe
print(f"Generating {cfg.n_students_answers_per_question} student answers per question...")
print(f"Target correct percentage: {cfg.percentile_correct * 100}%")

student_answers_df = generate_student_answers_df(
    tasks, cfg.n_students_answers_per_question, cfg.percentile_correct
)

print(f"Generated {len(student_answers_df)} total student answers")
print(f"Intended correct answers: {student_answers_df['intended_correct'].sum()}")
print(f"Intended incorrect answers: {(~student_answers_df['intended_correct']).sum()}")

# Save the dataframe
student_answers_filename = f"student_answers_{cfg.n_students_answers_per_question}_{cfg.percentile_correct}.csv"
output_path = os.path.join(cfg.output_dir, student_answers_filename)
student_answers_df.to_csv(output_path, index=False)
print(f"Saved student answers to: {output_path}")

# %%

# Sample 5 random incorrect student answers to display
incorrect_examples = student_answers_df[~student_answers_df['intended_correct']].sample(n=5, random_state=42)

for idx, example in incorrect_examples.iterrows():
    print("\nSampled Incorrect Example:")
    print("question: ", example["question"])
    print("answer: ", example["student_answer"])