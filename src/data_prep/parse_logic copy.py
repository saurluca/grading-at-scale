# %%
from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

df = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_parsed.csv", sep=";")

df_tasks = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_tasks.csv", sep=";")

df_points = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_points.csv", sep=";")

df_remarks = pd.read_csv(
    PROJECT_ROOT / "data" / "logic" / "quiz_1_remarks.csv", sep=";"
)

# remove column Teilnehmer
# df = df[["1. Sentences", "3. Validity and Soundness (II)"]]
# df = df.rename(
# columns={
# "1. Sentences": "sentences",
# "3. Validity and Soundness (II)": "validity_and_soundness",
# }
# )

# Maximalpunktzahl:;6;4;8;2;20
df_points = df_points[["1. Sentences", "3. Validity and Soundness (II)"]]
df_points = df_points.rename(
    columns={
        "1. Sentences": "sentences_points",
        "3. Validity and Soundness (II)": "validity_and_soundness_points",
    }
)

max_points_sentences = df_points["sentences_points"].max()
max_points_validity_and_soundness = df_points["validity_and_soundness_points"].max()

df = pd.concat([df, df_points], axis=1)

df_remarks = df_remarks.drop(columns=["username"])


df

# %%

# # replace </p><p> with <br /> for uniform parsing

# df["sentences"] = df["sentences"].str.replace("</p><p>", "<br />")
# df["validity_and_soundness"] = df["validity_and_soundness"].str.replace(
#     "</p><p>", "<br />"
# )

# # remove all background color spans (even if they only have background-color style) for all colors with regex


# def remove_background_colors(text):
#     # Remove all <span style="background-color:rgb(...); ..."> tags (catch any style after or not)
#     text = re.sub(
#         r'<span style="[^"]*background-color:rgb\(\d{1,3},\d{1,3},\d{1,3}\);?[^"]*">',
#         "",
#         text,
#     )
#     # Also remove <span style="background-color:transparent;color:#000000;">
#     text = re.sub(
#         r'<span style="?background-color:transparent;color:#000000;"?>',
#         "",
#         text,
#     )
#     # Also remove <span style="color:rgb(0,0,0);">
#     text = re.sub(
#         r'<span style="?color:rgb\(0,0,0\);"?>',
#         "",
#         text,
#     )
#     return text


# df["sentences"] = df["sentences"].apply(remove_background_colors)
# df["validity_and_soundness"] = df["validity_and_soundness"].apply(
#     remove_background_colors
# )

# # remove <!--HTML-->
# df["sentences"] = df["sentences"].str.replace("<!--HTML-->", "")
# df["validity_and_soundness"] = df["validity_and_soundness"].str.replace(
#     "<!--HTML-->", ""
# )


# # replace <br /><br /> with just <br />
# df["sentences"] = df["sentences"].str.replace("<br /><br />", "<br />")
# df["validity_and_soundness"] = df["validity_and_soundness"].str.replace(
#     "<br /><br />", "<br />"
# )

# df.to_csv(PROJECT_ROOT / "data" / "logic" /rue/False, because ...rue/False, because ...<br />true becaus...	5	<br />true becaus...	5	 "quiz_1_parsed.csv", index=False, sep=";")


# %%

df_sentences = df[["sentences"]].copy()
df_validity_and_soundness = df[["validity_and_soundness"]].copy()

# # select remarks for sentences and validity and soundness tasks
# df_sentence_remarks = df_remarks[df_remarks["task_id"] == 0]
# df_validity_and_soundness_remarks = df_remarks[df_remarks["task_id"] == 2]

# # drop task_id column from df_sentence_remarks and df_validity_and_soundness_remarks
# df_sentence_remarks = df_sentence_remarks.drop(columns=["task_id"])
# df_validity_and_soundness_remarks = df_validity_and_soundness_remarks.drop(
#     columns=["task_id"]
# )

# # concatenate remarks with sentences and validity and soundness dataframes
# df_sentences = pd.concat([df_sentences, df_sentence_remarks], axis=1)
# df_validity_and_soundness = pd.concat(
#     [df_validity_and_soundness, df_validity_and_soundness_remarks], axis=1
# )

# # save dataframes to csv
# df_sentences.to_csv(
#     PROJECT_ROOT / "data" / "logic" / "quiz_1_sentences.csv", index=False, sep=";"
# )
# df_validity_and_soundness.to_csv(
#     PROJECT_ROOT / "data" / "logic" / "quiz_1_validity_and_soundness.csv",
#     index=False,
#     sep=";",
# )


# %%

# Split subtasks into separate rows ....


# replace <br /> with ;
df_sentences = df_sentences.str.replace("<br />", ";")
df_validity_and_soundness = df_validity_and_soundness.str.replace("<br />", ";")

# %%


# Split student answers by <br /> tags and create unified dataframe
def split_answers_by_br(text):
    """Split text by <br /> tags and return list of answers"""
    if pd.isna(text):
        return []
    return [answer.strip() for answer in text.split("<br />") if answer.strip()]


# Split answers for both columns
df["sentences_split"] = df["sentences"].apply(split_answers_by_br)
df["validity_and_soundness_split"] = df["validity_and_soundness"].apply(
    split_answers_by_br
)

# Create separate dataframes for each task type
sentences_data = []
validity_data = []

# Process sentences data (6 questions per student)
for student_idx, row in df.iterrows():
    answers = row["sentences_split"]
    for q_idx, answer in enumerate(answers):
        if q_idx < 6:  # Only 6 sentences questions
            sentences_data.append(
                {
                    "student_idx": student_idx,
                    "question_idx": q_idx,
                    "student_answer": answer,
                    "task": "sentences",
                }
            )

# Process validity_and_soundness data (7 questions per student)
for student_idx, row in df.iterrows():
    answers = row["validity_and_soundness_split"]
    for q_idx, answer in enumerate(answers):
        if q_idx < 8:  # Only 7 validity questions
            validity_data.append(
                {
                    "student_idx": student_idx,
                    "question_idx": q_idx,
                    "student_answer": answer,
                    "task": "validity_and_soundness",
                }
            )

# Convert to dataframes
df_sentences = pd.DataFrame(sentences_data)
df_validity = pd.DataFrame(validity_data)

df_sentences.head()

# %%

# Get questions for each task type from df_tasks
sentences_questions = df_tasks[df_tasks["task"] == "sentences"].reset_index(drop=True)
# Handle both 'validity_and_soundness' and ' validity_and_soundness' (with leading space)
validity_questions = df_tasks[
    df_tasks["task"].str.strip() == "validity_and_soundness"
].reset_index(drop=True)

# Add question and reference_answer to sentences data
df_sentences["question"] = df_sentences["question_idx"].map(
    sentences_questions["question"]
)
df_sentences["reference_answer"] = df_sentences["question_idx"].map(
    sentences_questions["answer"]
)

# Add question and reference_answer to validity data
df_validity["question"] = df_validity["question_idx"].map(
    validity_questions["question"]
)
df_validity["reference_answer"] = df_validity["question_idx"].map(
    validity_questions["answer"]
)

# Combine both dataframes
df_unified = pd.concat([df_sentences, df_validity], ignore_index=True)

# remove all <p> and </p> tags from student_answer
df_unified["student_answer"] = (
    df_unified["student_answer"].str.replace("<p>", "").str.replace("</p>", "")
)

# add colum points, filled with 1 for every row
df_unified["points"] = 1

# if colum student_answer is empty set to 0
df_unified["points"] = df_unified["points"].apply(lambda x: 0 if x == "" else x)

# if column student answer contains Yes/No, because ... and has the lenght of Yes/No, because ... plus 3, exaclty set to 0
df_unified["points"] = df_unified["points"].apply(
    lambda x: 0
    if "Yes/No, because ..." in x and len(x) == len("Yes/No, because ...") + 3
    else x
)

# prin out number of 0
print(f"Number of 0 points: {df_unified['points'].value_counts()[0]}")

# Select final columns and reorder
df_final = df_unified[
    ["question", "reference_answer", "student_answer", "task", "points"]
].copy()

# # Save the unified dataframe
output_path = PROJECT_ROOT / "data" / "logic" / "quiz_1_unified.csv"
# df_final.to_csv(output_path, index=False, sep=";")

print(f"Unified dataframe created with {len(df_final)} rows")
print(f"Saved to: {output_path}")
print(f"Columns: {list(df_final.columns)}")
print("Task distribution:")
print(df_final["task"].value_counts())

# %%

df_tasks.columns
