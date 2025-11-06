# %%
from pathlib import Path
import pandas as pd
import yaml
import sys
import numpy as np
from datasets import DatasetDict, Dataset

# Add src to path to import common module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from common import load_and_preprocess_data

# Load configuration
config_path = PROJECT_ROOT / "configs" / "data_generation.yaml"
base_config_path = PROJECT_ROOT / "configs" / "base.yaml"

with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

with open(config_path, "r") as f:
    data_config = yaml.safe_load(f)

# Merge configurations (data_config takes precedence)
config = {**base_config, **data_config}

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
synth_dir = PROJECT_ROOT / "data" / "gras"
seed = config["project"]["seed"]

# Load the full dataset
print("Loading full dataset...")
df_full = pd.read_csv(data_dir / "gras" / "full.csv", sep=";")

# Save to temporary file for load_and_preprocess_data function
temp_csv_path = synth_dir / "temp_full.csv"
df_full.to_csv(temp_csv_path, index=False, sep=";")

# %%
# Split dataset by task_id (questions) to ensure no overlap between splits
print("Splitting dataset into train/val/test (6/2/2) by task_id...")

# Load and preprocess data (for label mapping, etc.)
# Note: load_and_preprocess_data always splits by questions (task_id) now
raw_data, label_order, label2id, id2label = load_and_preprocess_data(
    dataset_csv=str(temp_csv_path),
    cache_dir=None,
    seed=seed,
    test_size=0.2,  # 20% for test
)

# Get unique task_ids from the full dataset
df_full_with_labels = raw_data["train"].to_pandas()
df_test_with_labels = raw_data["test"].to_pandas()
df_combined = pd.concat([df_full_with_labels, df_test_with_labels], ignore_index=True)

# Group task_ids by topic for stratified splitting
task_id_to_topic = df_combined.groupby("task_id")["topic"].first().to_dict()
topics_to_task_ids = {}
for task_id, topic in task_id_to_topic.items():
    if topic not in topics_to_task_ids:
        topics_to_task_ids[topic] = []
    topics_to_task_ids[topic].append(task_id)

print(f"Total unique task_ids: {len(task_id_to_topic)}")
print(f"Topics found: {sorted(topics_to_task_ids.keys())}")

# Split task_ids by topic to maintain topic proportions across splits
rng = np.random.default_rng(seed)
train_task_ids = []
val_task_ids = []
test_task_ids = []

for topic, task_ids in sorted(topics_to_task_ids.items()):
    # Shuffle task_ids for this topic
    shuffled_task_ids = task_ids.copy()
    rng.shuffle(shuffled_task_ids)
    
    # Calculate split sizes for this topic (60% train, 20% val, 20% test)
    n_total = len(shuffled_task_ids)
    
    # For very small topics, ensure at least one question per split if possible
    if n_total <= 2:
        # If only 1-2 questions, put all in train (can't split properly)
        n_train = n_total
        n_val = 0
        n_test = 0
    elif n_total == 3:
        # 3 questions: 1 train, 1 val, 1 test
        n_train = 1
        n_val = 1
        n_test = 1
    else:
        # For 4+ questions, use proportional split (60/20/20)
        # Calculate target sizes and round to nearest integers
        n_test = max(1, round(n_total * 0.2))
        n_val = max(1, round(n_total * 0.2))
        n_train = n_total - n_test - n_val  # Remaining goes to train
    
    topic_train = shuffled_task_ids[:n_train]
    topic_val = shuffled_task_ids[n_train:n_train + n_val] if n_val > 0 else []
    topic_test = shuffled_task_ids[n_train + n_val:] if n_test > 0 else []
    
    train_task_ids.extend(topic_train)
    if topic_val:
        val_task_ids.extend(topic_val)
    if topic_test:
        test_task_ids.extend(topic_test)
    
    print(
        f"Topic '{topic}': {n_total} questions -> train={len(topic_train)}, val={len(topic_val)}, test={len(topic_test)}"
    )

train_task_ids = set(train_task_ids)
val_task_ids = set(val_task_ids)
test_task_ids = set(test_task_ids)

print(
    f"\nOverall task_id split: train={len(train_task_ids)}, val={len(val_task_ids)}, test={len(test_task_ids)}"
)

# Filter datasets based on task_id assignment
train_indices = [
    i for i, task_id in enumerate(df_combined["task_id"]) if task_id in train_task_ids
]
val_indices = [
    i for i, task_id in enumerate(df_combined["task_id"]) if task_id in val_task_ids
]
test_indices = [
    i for i, task_id in enumerate(df_combined["task_id"]) if task_id in test_task_ids
]

# Convert back to datasets
train_dataset = Dataset.from_pandas(df_combined.iloc[train_indices].reset_index(drop=True))
val_dataset = Dataset.from_pandas(df_combined.iloc[val_indices].reset_index(drop=True))
test_dataset = Dataset.from_pandas(df_combined.iloc[test_indices].reset_index(drop=True))

# Create final dataset with 6/2/2 split (60% train, 20% val, 20% test)
final_dataset = DatasetDict(
    {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
)

print(f"Final split sizes:")
print(f"Train: {len(final_dataset['train'])} samples")
print(f"Validation: {len(final_dataset['val'])} samples")
print(f"Test: {len(final_dataset['test'])} samples")
print(
    f"Total: {len(final_dataset['train']) + len(final_dataset['val']) + len(final_dataset['test'])} samples"
)

# %%
# Save the split datasets
print("Saving split datasets...")

# Convert back to pandas DataFrames and save
for split_name, dataset in final_dataset.items():
    # Convert to pandas DataFrame
    df_split = dataset.to_pandas()

    # Save to CSV
    output_path = synth_dir / f"{split_name}.csv"
    df_split.to_csv(output_path, index=False, sep=";")
    print(f"Saved {split_name} split to {output_path}")

# Clean up temporary file
temp_csv_path.unlink()

print("Dataset splitting completed!")
