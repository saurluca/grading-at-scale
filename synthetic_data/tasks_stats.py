import os
import sys
from pathlib import Path
import pandas as pd


# Ensure project root is on sys.path for absolute imports (works in scripts and notebooks)
if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config


def main():
    cfg = load_config("synthetic_data")

    base_dir = os.path.dirname(__file__)
    output_dir = os.path.normpath(os.path.join(base_dir, "../", cfg.output_dir))
    csv_path = os.path.join(output_dir, cfg.tasks_filename)

    if not os.path.exists(csv_path):
        print(f"Unified tasks CSV not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    total_tasks = len(df)
    print(f"Total tasks: {total_tasks}")

    if "topic" in df.columns:
        counts = df["topic"].value_counts().sort_values(ascending=False)
        print(f"Number of topics: {counts.shape[0]}")
        print("Tasks per topic:")
        for topic, count in counts.items():
            print(f" - {topic}: {count}")
    else:
        print("Column 'topic' not found in CSV; per-topic counts unavailable.")


if __name__ == "__main__":
    main()
