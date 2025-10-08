# %%
from pathlib import Path
import pandas as pd

# df = pd.read_csv(
#     "../data/synth/student_answers_c3_p3_i3_gpt-4o_per_question.csv", sep=";"
# )

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

df_1 = pd.read_csv(PROJECT_ROOT / "data" / "synth" / "all_tasks.csv", sep=";")

df_2 = pd.read_csv(PROJECT_ROOT / "results" / "tasks_cnp.csv", sep=";")


# %%

print(f"len df 1: {len(df_1)}")
print(f"len df 2 : {len(df_2)}")

print(f"Columns df 1: {df_1.columns}")
print(f"Columns df 2: {df_2.columns}")

# merge both datasets
df = pd.concat([df_1, df_2])
print(f"len df: {len(df)}")

df.head()


# %%
df.to_csv(PROJECT_ROOT / "data" / "synth" / "all_tasks_n.csv", index=False, sep=";")


def preprocess_ai_tasks(df):
    df["short_answer"] = df["answer"]
    df["reference_answer"] = df["short_answer"] + " - Explanation: " + df["explanation"]
    df["chunk_text"] = ""
    df["topic"] = "ai"
    return df


def preprocess_neuro_tasks(df):
    df["reference_answer"] = df["answer"]
    df["short_answer"] = ""
    df["explanation"] = ""
    df.drop(columns=["answer"], inplace=True)
    return df
