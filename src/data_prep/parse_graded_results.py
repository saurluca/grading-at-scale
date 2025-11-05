# %%
"""
Transform graded quiz results from wide format to long format.
Converts one row per participant to one row per participant-question combination.
"""

from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# %%
# Load data files
df_graded = pd.read_csv(
    PROJECT_ROOT / "data" / "logic" / "quiz_1_subtasks_graded.csv", sep=";"
)
df_tasks = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_tasks.csv", sep=";")

print(f"Loaded {len(df_graded)} participants")
print(f"Loaded {len(df_tasks)} questions")

# %%
# Create mapping from task column names to question and reference_answer
# Filter tasks by type
sentences_tasks = df_tasks[df_tasks["task"] == "sentences"].reset_index(drop=True)
validity_tasks = df_tasks[
    df_tasks["task"].str.strip() == "validity_and_soundness"
].reset_index(drop=True)

print(f"Sentences tasks: {len(sentences_tasks)}")
print(f"Validity tasks: {len(validity_tasks)}")

# Create mapping: task_column_name -> {question, reference_answer}
task_mapping = {}

# Map sentences subtasks (a-f)
for idx, letter in enumerate(["a", "b", "c", "d", "e", "f"]):
    task_col = f"sentences_{letter}"
    task_mapping[task_col] = {
        "question": sentences_tasks.iloc[idx]["question"],
        "reference_answer": sentences_tasks.iloc[idx]["answer"],
    }

# Map validity_and_soundness subtasks (a-h)
for idx, letter in enumerate(["a", "b", "c", "d", "e", "f", "g", "h"]):
    task_col = f"validity_and_soundness_{letter}"
    task_mapping[task_col] = {
        "question": validity_tasks.iloc[idx]["question"],
        "reference_answer": validity_tasks.iloc[idx]["answer"],
    }

# %%
# Process columns directly
# Rename Teilnehmer to student
df_graded = df_graded.rename(columns={"Teilnehmer": "student"})


# Filter out Quiz Group 28 and higher
def should_keep_student(student_name):
    """Check if student should be kept (exclude Quiz Group 28+)"""
    if pd.isna(student_name):
        return True
    student_str = str(student_name)
    if student_str.startswith("Quiz Group "):
        # Extract number from "Quiz Group X"
        match = re.search(r"Quiz Group (\d+)", student_str)
        if match:
            group_num = int(match.group(1))
            return group_num < 28
    return True


df_graded = df_graded[df_graded["student"].apply(should_keep_student)].reset_index(
    drop=True
)

print(f"After filtering: {len(df_graded)} participants")

# Drop sentences and validity_and_soundness columns
df_graded = df_graded.drop(
    columns=["sentences", "validity_and_soundness"], errors="ignore"
)

# Identify all task columns (sentences_a, sentences_b, ..., validity_and_soundness_h)
task_columns = [col for col in df_graded.columns if col in task_mapping]


def clean_student_answer(answer):
    """Remove 'True/False, because ...<br />' and 'Yes/No, because ...<br />' prefixes"""
    if pd.isna(answer) or answer == "":
        return ""

    answer_str = str(answer)

    # Remove "True/False, because ...<br />" pattern (case insensitive, flexible spacing)
    answer_str = re.sub(
        r"True/False,\s*because\s*\.\.\.\s*<br\s*/?>",
        "",
        answer_str,
        flags=re.IGNORECASE,
    )

    # Remove "Yes/No, because ...<br />" pattern (case insensitive, flexible spacing)
    answer_str = re.sub(
        r"Yes/No,\s*because\s*\.\.\.\s*<br\s*/?>",
        "",
        answer_str,
        flags=re.IGNORECASE,
    )

    return answer_str.strip()


# %%
# Pivot to long format
results = []

for idx, row in df_graded.iterrows():
    student = row["student"]

    for task_col in task_columns:
        # Get student answer from task column and clean it
        student_answer_raw = row[task_col] if pd.notna(row[task_col]) else ""
        student_answer = clean_student_answer(student_answer_raw)

        # Get points from corresponding points column
        points_col = f"{task_col}_points"
        points = float(row[points_col]) if pd.notna(row[points_col]) else -1.0

        # Get question and reference answer from mapping
        question = task_mapping[task_col]["question"]
        reference_answer = task_mapping[task_col]["reference_answer"]

        results.append(
            {
                "student": student,
                "task": task_col,
                "student_answer": student_answer,
                "points": points,
                "question": question,
                "reference_answer": reference_answer,
            }
        )

# %%
# Create DataFrame and save
df_final = pd.DataFrame(results)

print(f"\nFinal DataFrame shape: {df_final.shape}")
print(f"Columns: {list(df_final.columns)}")
print("\nSample rows:")
print(df_final.head(10))

# Save to CSV
output_path = PROJECT_ROOT / "data" / "logic" / "quiz_1_final.csv"
df_final.to_csv(output_path, index=False, sep=";")

print(f"\nSaved to: {output_path}")
print(f"Total rows: {len(df_final)}")
