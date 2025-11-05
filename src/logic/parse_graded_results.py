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


# Filter out Quiz Groups 3-9
def should_keep_student(student_name):
    """Check if student should be kept (exclude Quiz Groups 3-9)"""
    if pd.isna(student_name):
        return True
    student_str = str(student_name)
    if student_str.startswith("Quiz Group "):
        # Extract number from "Quiz Group X"
        match = re.search(r"Quiz Group (\d+)", student_str)
        if match:
            group_num = int(match.group(1))
            # Exclude groups 3-9
            return (group_num < 3 or group_num > 9) and group_num < 28
    return True


df_graded = df_graded[df_graded["student"].apply(should_keep_student)].reset_index(
    drop=True
)

print(f"After filtering: {len(df_graded)} participants")

# %%
# HTML cleaning functions (from logic_new.py)


def remove_background_colors(text):
    """Remove all background color spans from HTML text"""
    if pd.isna(text):
        return text
    text = str(text)
    # Remove all <span style="background-color:rgb(...); ..."> tags (catch any style after or not)
    text = re.sub(
        r'<span style="[^"]*background-color:rgb\(\d{1,3},\d{1,3},\d{1,3}\);?[^"]*">',
        "",
        text,
    )
    # Also remove <span style="background-color:transparent;color:#000000;">
    text = re.sub(
        r'<span style="?background-color:transparent;color:#000000;"?>',
        "",
        text,
    )
    # Also remove <span style="color:rgb(0,0,0);">
    text = re.sub(
        r'<span style="?color:rgb\(0,0,0\);"?>',
        "",
        text,
    )
    return text


def clean_html_text(text):
    """Apply all HTML cleaning steps to text"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = remove_background_colors(text)
    # remove <!--HTML-->
    text = text.replace("<!--HTML-->", "")
    # replace </p><p> with <br /> for uniform parsing
    text = text.replace("</p><p>", "<br />")
    # replace <br /><br /> with just <br />
    text = text.replace("<br /><br />", "<br />")
    # remove </span> tags
    text = text.replace("</span>", "")
    return text


# Clean HTML columns before extraction
df_graded["sentences"] = df_graded["sentences"].apply(clean_html_text)
df_graded["validity_and_soundness"] = df_graded["validity_and_soundness"].apply(
    clean_html_text
)

# %%
# Improved extraction function


def extract_subtask(text, letter, task_type="sentences"):
    """
    Extract content for a specific subtask letter from text.

    Only stops at the NEXT letter in sequence (e.g., when extracting "a.",
    only stops at "b.", not at "a." within the text).
    Avoids false matches on "e.g." by checking that it's not followed by "g."

    Args:
        text: HTML text containing subtasks
        letter: Letter to extract (a-h)
        task_type: "sentences" (a-f) or "validity_and_soundness" (a-h)

    Returns:
        Extracted content for the subtask
    """
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    # Define letter sequences
    if task_type == "sentences":
        letters = ["a", "b", "c", "d", "e", "f"]
    else:  # validity_and_soundness
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]

    # Find position of current letter
    try:
        letter_idx = letters.index(letter.lower())
        next_letter = letters[letter_idx + 1] if letter_idx + 1 < len(letters) else None
    except ValueError:
        return ""

    # Pattern to match: letter. followed by content until next letter. or end
    # Use word boundary to match at word boundaries (e.g., "a." not "xa.")
    pattern = rf"\b{letter}\.\s*"

    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return ""

    start_pos = match.end()

    # Find end position: either next letter or end of text
    if next_letter:
        # Look for next letter pattern
        # Pattern: word boundary, letter, period, NOT followed by word character
        # This avoids matching "e.g." because "e.g." has "g" (word char) after "e."
        # It will match "e. " (space), "e.<br" (HTML), "e.</p" (HTML), or end of text
        next_pattern = rf"\b{next_letter}\.(?!\w)"

        next_match = re.search(next_pattern, text[start_pos:], re.IGNORECASE)
        if next_match:
            end_pos = start_pos + next_match.start()
        else:
            end_pos = len(text)
    else:
        # Last letter, go to end
        end_pos = len(text)

    content = text[start_pos:end_pos].strip()

    # Clean up any trailing HTML closing tags or whitespace
    content = re.sub(r"\s*</?p>\s*$", "", content)
    content = re.sub(r"\s*<br\s*/?>\s*$", "", content)
    # Remove any leftover </span> tags
    content = content.replace("</span>", "")
    # Remove any leftover <br /> tags throughout the content
    content = re.sub(r"<br\s*/?>", " ", content)
    # Clean up multiple spaces
    content = re.sub(r"\s+", " ", content)

    return content.strip()


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

    # Remove any leftover </span> tags
    answer_str = answer_str.replace("</span>", "")
    # Remove any leftover <br /> tags
    answer_str = re.sub(r"<br\s*/?>", " ", answer_str)
    # Clean up multiple spaces
    answer_str = re.sub(r"\s+", " ", answer_str)

    return answer_str.strip()


# %%
# Pivot to long format - extract from HTML columns directly
results = []

for idx, row in df_graded.iterrows():
    student = row["student"]

    # Extract sentences subtasks (a-f)
    sentences_html = row["sentences"] if pd.notna(row["sentences"]) else ""
    for letter in ["a", "b", "c", "d", "e", "f"]:
        task_col = f"sentences_{letter}"
        # Extract answer from HTML
        student_answer_raw = extract_subtask(
            sentences_html, letter, task_type="sentences"
        )
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
                "points": points,
                "student_answer": student_answer,
                "question": question,
                "reference_answer": reference_answer,
            }
        )

    # Extract validity_and_soundness subtasks (a-h)
    validity_html = (
        row["validity_and_soundness"] if pd.notna(row["validity_and_soundness"]) else ""
    )
    for letter in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        task_col = f"validity_and_soundness_{letter}"
        # Extract answer from HTML
        student_answer_raw = extract_subtask(
            validity_html, letter, task_type="validity_and_soundness"
        )
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
                "points": points,
                "student_answer": student_answer,
                "question": question,
                "reference_answer": reference_answer,
            }
        )

# %%
# Create DataFrame and save
df_final = pd.DataFrame(results)

# Ensure columns are in the correct order: student, task, points, student_answer, question, reference_answer
column_order = [
    "student",
    "task",
    "points",
    "student_answer",
    "question",
    "reference_answer",
]
df_final = df_final[column_order]

print(f"\nFinal DataFrame shape: {df_final.shape}")
print(f"Columns: {list(df_final.columns)}")
print("\nSample rows:")
print(df_final.head(10))

# Save to CSV
output_path = PROJECT_ROOT / "data" / "logic" / "quiz_1_final.csv"
df_final.to_csv(output_path, index=False, sep=";")

print(f"\nSaved to: {output_path}")
print(f"Total rows: {len(df_final)}")
