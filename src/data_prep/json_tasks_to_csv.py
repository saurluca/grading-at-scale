# %%
import json
import pandas as pd
import os
from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load configuration
base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
data_gen_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "data_generation.yaml")
cfg = OmegaConf.merge(base_cfg, data_gen_cfg)

# Resolve directories relative to project root
output_dir = PROJECT_ROOT / cfg.output.dir
raw_tasks_dir = PROJECT_ROOT / cfg.input.raw_tasks_dir

"""
Process all task JSON files under raw tasks directory into a single CSV with columns:
question, answer, topic (topic is the file stem, e.g., 'language').
"""

# Discover all task files in the raw tasks directory
tasks_files = {f for f in os.listdir(raw_tasks_dir) if f.endswith(".json")}
common_files = sorted(tasks_files)

if not common_files:
    print("No task files found in tasks directory.")

data = []
for filename in common_files:
    topic = os.path.splitext(filename)[0]
    tasks_json_path = os.path.join(raw_tasks_dir, filename)

    with open(tasks_json_path, "r") as f:
        tasks = json.load(f)

    # Extract rows
    for task in tasks:
        question = task.get("question", "")
        answer_options = task.get("answer_options", []) or []
        # Prefer the option marked as correct; fallback to first if none marked
        correct = next((opt for opt in answer_options if opt.get("is_correct")), None)
        if correct is None and answer_options:
            correct = answer_options[0]
        answer = correct.get("answer", "") if correct else ""

        data.append(
            {
                "question": question,
                "answer": answer,
                "topic": topic,
            }
        )

# Create DataFrame
df = pd.DataFrame(data)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save to CSV using config filename
output_path = os.path.join(output_dir, cfg.input.tasks_filename)
df.to_csv(output_path, index=False, sep=";")

print(f"Processed topics: {', '.join(os.path.splitext(f)[0] for f in common_files)}")
print(f"Created CSV with {len(df)} rows")
print(f"Saved to: {output_path}")
print("\nFirst few rows:")
print(df.head())
