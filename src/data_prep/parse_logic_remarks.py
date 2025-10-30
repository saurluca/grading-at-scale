# %%
from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

df_unified = pd.read_csv(
    PROJECT_ROOT / "data" / "logic" / "quiz_1_unified.csv", sep=";"
)

df_remarks = pd.read_csv(
    PROJECT_ROOT / "data" / "logic" / "quiz_1_remarks.csv", sep=";"
)

df_points = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_points.csv", sep=";")

# remove all columns where task_id is not 0 or 2, because irrelvant tasks
df_remarks = df_remarks[df_remarks["task_id"].isin([0, 2])]

# %%

df_unified

# %%

# remove all <p> and </p> tags from student_answer
df_unified["student_answer"] = (
    df_unified["student_answer"].str.replace("<p>", "").str.replace("</p>", "")
)

# also remove all </span> and <span> tags from student_answer
df_unified["student_answer"] = (
    df_unified["student_answer"].str.replace("</span>", "").str.replace("<span>", "")
)

# add colum points, filled with 1 for every row
df_unified["points"] = 1

# if colum student_answer is empty set to 0
df_unified.loc[df_unified["student_answer"] == "", "points"] = 0

# if column student answer contains Yes/No, because ... and has the lenght of Yes/No, because ... plus 3, exaclty set to 0
df_unified.loc[
    (df_unified["student_answer"].str.contains("Yes/No, because ...", na=False))
    & (df_unified["student_answer"].str.len() == len("Yes/No, because ...") + 3),
    "points",
] = 0

# if column student answer contains True/False, because ... and has the lenght of True/False, because ... plus 3, exaclty set to 0
df_unified.loc[
    (df_unified["student_answer"].str.contains("True/False, because ...", na=False))
    & (df_unified["student_answer"].str.len() == len("True/False, because ...") + 3),
    "points",
] = 0

# prin out number of 0
print(f"Number of 0 points: {df_unified['points'].value_counts()}")

# Select final columns and reorder
df_final = df_unified[
    ["question", "reference_answer", "student_answer", "task", "points"]
].copy()

df_final

# save to csv
df_final.to_csv(
    PROJECT_ROOT / "data" / "logic" / "quiz_1_unified_remarks.csv", index=False, sep=";"
)

# %%


# %%

# convert 1. Sentences and 3. Validity and Soundness (II) to float
# replace comma with dot for German number notation
df_points["1. Sentences"] = (
    df_points["1. Sentences"].str.replace(",", ".").astype(float)
)
df_points["3. Validity and Soundness (II)"] = (
    df_points["3. Validity and Soundness (II)"].str.replace(",", ".").astype(float)
)

# points checksum
df_points_checksum = df_points["1. Sentences"].sum()
df_points_checksum += df_points["3. Validity and Soundness (II)"].sum()

df_final_checksum = df_final["points"].sum()

print(f"Points checksum: {df_points_checksum}")
print(f"Final checksum: {df_final_checksum}")
# %%
