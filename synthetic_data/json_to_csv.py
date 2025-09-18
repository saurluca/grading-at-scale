# %%
import json
import pandas as pd
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for absolute imports (works in scripts and notebooks)
if "__file__" in globals():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    _PROJECT_ROOT = Path.cwd().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils import load_config

# Load configuration
cfg = load_config("synthetic_data")

# Resolve directories relative to project root
base_dir = os.path.dirname(__file__)
output_dir = os.path.normpath(os.path.join(base_dir, "../", cfg.output_dir))
raw_tasks_dir = os.path.normpath(os.path.join(base_dir, "../", cfg.raw_tasks_dir))
raw_chunks_dir = os.path.normpath(os.path.join(base_dir, "../", cfg.raw_chunks_dir))

"""
Unify all task/chunk JSON pairs under raw directories into a single CSV with columns:
question, answer, chunk_text, topic (topic is the file stem, e.g., 'privacy').
"""

# Discover matching topic files present in both raw directories
tasks_files = {f for f in os.listdir(raw_tasks_dir) if f.endswith(".json")}
chunks_files = {f for f in os.listdir(raw_chunks_dir) if f.endswith(".json")}
common_files = sorted(tasks_files.intersection(chunks_files))

if not common_files:
    print("No matching topic files found between tasks and chunks directories.")

data = []
for filename in common_files:
    topic = os.path.splitext(filename)[0]
    tasks_json_path = os.path.join(raw_tasks_dir, filename)
    chunks_json_path = os.path.join(raw_chunks_dir, filename)

    with open(tasks_json_path, "r") as f:
        tasks = json.load(f)
    with open(chunks_json_path, "r") as f:
        chunks = json.load(f)

    # Map chunk_id to chunk_text
    chunk_dict = {chunk["id"]: chunk.get("chunk_text", "") for chunk in chunks}

    # Extract rows
    for task in tasks:
        question = task.get("question", "")
        answer_options = task.get("answer_options", []) or []
        # Prefer the option marked as correct; fallback to first if none marked
        correct = next((opt for opt in answer_options if opt.get("is_correct")), None)
        if correct is None and answer_options:
            correct = answer_options[0]
        answer = correct.get("answer", "") if correct else ""

        chunk_id = task.get("chunk_id")
        chunk_text = chunk_dict.get(chunk_id, "")

        data.append(
            {
                "question": question,
                "answer": answer,
                "chunk_text": chunk_text,
                "topic": topic,
            }
        )

# Create DataFrame
df = pd.DataFrame(data)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save to CSV using config filename
output_path = os.path.join(output_dir, cfg.tasks_filename)
df.to_csv(output_path, index=False)

print(f"Processed topics: {', '.join(os.path.splitext(f)[0] for f in common_files)}")
print(f"Created CSV with {len(df)} rows")
print(f"Saved to: {output_path}")
print("\nFirst few rows:")
print(df.head())
