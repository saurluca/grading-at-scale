import pandas as pd
import csv
from pathlib import Path


def get_max_task_id(full_csv_path: Path) -> int:
    """Read full.csv and return the maximum task_id."""
    df = pd.read_csv(full_csv_path, sep=";")
    return int(df["task_id"].max())


def convert_points_to_labels(points) -> int:
    """Convert points to labels: 0.0 → 0, 0.5 → 1, 1.0 → 2."""
    points_float = float(points)
    if points_float == 0.0:
        return 0
    elif points_float == 0.5:
        return 1
    elif points_float == 1.0:
        return 2
    else:
        raise ValueError(f"Unexpected points value: {points_float}")


def determine_topic(task: str) -> str:
    """Determine topic based on task prefix."""
    if task.startswith("sentences_"):
        return "sentences"
    elif task.startswith("validity_and_soundness_"):
        return "validity"
    else:
        raise ValueError(f"Unknown task prefix: {task}")


def merge_quiz_data():
    """Merge quiz_1_final.csv into full.csv."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    quiz_path = project_root / "data" / "gras" / "quiz_1_final.csv"
    full_path = project_root / "data" / "gras" / "full.csv"

    # Get max task_id from full.csv
    max_task_id = get_max_task_id(full_path)
    print(f"Max task_id in full.csv: {max_task_id}")

    # Read quiz data - handle semicolons in fields by using csv module
    quiz_rows = []
    with open(quiz_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            quiz_rows.append(row)

    quiz_df = pd.DataFrame(quiz_rows)
    print(f"Loaded {len(quiz_df)} rows from quiz_1_final.csv")

    # Filter out header row if it exists as data
    quiz_df = quiz_df[quiz_df["task"] != "task"]

    # Get unique tasks and create mapping
    unique_tasks = sorted(quiz_df["task"].unique())
    task_to_id = {task: max_task_id + 1 + i for i, task in enumerate(unique_tasks)}
    print(f"Created task_id mapping: {task_to_id}")

    # Transform quiz data
    transformed_rows = []
    for _, row in quiz_df.iterrows():
        task = row["task"]
        task_id = task_to_id[task]
        topic = determine_topic(task)

        transformed_row = {
            "task_id": task_id,
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "topic": topic,
            "student_answer": row["student_answer"],
            "labels": convert_points_to_labels(
                row["points"]
            ),  # Map points to labels (0, 1, 2)
        }
        transformed_rows.append(transformed_row)

    # Create DataFrame with transformed data
    transformed_df = pd.DataFrame(transformed_rows)

    # Ensure correct column order
    column_order = [
        "task_id",
        "question",
        "reference_answer",
        "topic",
        "student_answer",
        "labels",
    ]
    transformed_df = transformed_df[column_order]

    print(f"Transformed {len(transformed_df)} rows")

    # Read existing full.csv
    full_df = pd.read_csv(full_path, sep=";")
    print(f"Loaded {len(full_df)} rows from full.csv")

    # Append transformed data
    merged_df = pd.concat([full_df, transformed_df], ignore_index=True)
    print(f"Merged dataset has {len(merged_df)} rows")

    # Write back to full.csv
    merged_df.to_csv(full_path, sep=";", index=False)
    print(f"Successfully wrote merged data to {full_path}")


if __name__ == "__main__":
    merge_quiz_data()
