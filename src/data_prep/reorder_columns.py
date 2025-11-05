from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Read CSV
df = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_final.csv", sep=";")

# Reorder columns: student, task, points, student_answer, question, reference_answer
df = df[["student", "task", "points", "student_answer", "question", "reference_answer"]]

# Save back
df.to_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_final.csv", index=False, sep=";")

print("Columns reordered successfully!")
