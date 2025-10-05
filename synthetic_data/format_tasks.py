# %%
import pandas as pd

# df = pd.read_csv(
#     "../data/synth/student_answers_c3_p3_i3_gpt-4o_per_question.csv", sep=";"
# )

df_neuro = pd.read_csv("../data/synth/neuro_tasks_with_explanation.csv", sep=";")

df_ai = pd.read_csv("../data/synth/ai_tasks_with_explanation.csv", sep=";")


# %%

print(f"len neuro: {len(df_neuro)}")
print(f"len ai: {len(df_ai)}")

# merge both datasets
df = pd.concat([df_neuro, df_ai])
df.drop(columns=["answer"], inplace=True)
print(f"len df: {len(df)}")

df.head()

df.to_csv("../data/synth/all_tasks.csv", index=False, sep=";")

# %%


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
