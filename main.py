# %%
import os
import pandas as pd

# Roars datase: https://github.com/owenhenkel/ROARS-dataset/blob/main/ROARS_datadictionary.csv
file_path = os.path.join(os.path.dirname(__file__), "data", "ROARS.csv")

# read in data
data = pd.read_csv(
    file_path,
    usecols=[
        "story",
        "story_text",
        "passage_text",
        "question_num",
        "question_text",
        "student_answer",
        "3class_human_groundtruth",
        "3class_best_model",
    ],
)


# %%
# print number of unique values for each column
data.nunique()

# %%
# print correlation matrix for only numeric columns
data.select_dtypes(include=["number"]).corr()
