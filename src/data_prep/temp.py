# %%
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

df_1 = pd.read_csv(PROJECT_ROOT / "data" / "synth" / "all_tasks_n.csv", sep=";")


print(f"len df 1: {len(df_1)}")

print(f"Columns df 1: {df_1.columns}")

# drop column topic
df_1.drop(columns=["short_answer", "explanation"], inplace=True)

df_1 = df_1[["topic", "question", "reference_answer", "chunk_text"]]

df_1.head()


# %%
df_1.to_csv(PROJECT_ROOT / "data" / "synth" / "all_tasks_n.csv", index=False, sep=";")
