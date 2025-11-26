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

# TODO can we isolate train_test_split to not rely on common.py?
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

# Calculate label distribution per task_id
# This will help us stratify by labels while splitting by task_id
task_id_label_counts = {}
for task_id in df_combined["task_id"].unique():
    task_rows = df_combined[df_combined["task_id"] == task_id]
    label_counts = task_rows["labels"].value_counts().to_dict()
    # Ensure all labels are present (0=incorrect, 1=partial, 2=correct)
    task_id_label_counts[task_id] = {
        0: label_counts.get(0, 0),
        1: label_counts.get(1, 0),
        2: label_counts.get(2, 0),
    }

print(f"Total unique task_ids: {len(task_id_to_topic)}")
print(f"Topics found: {sorted(topics_to_task_ids.keys())}")

# Split task_ids by topic with label stratification
rng = np.random.default_rng(seed)
train_task_ids = []
val_task_ids = []
test_task_ids = []

for topic, task_ids in sorted(topics_to_task_ids.items()):
    shuffled_task_ids = task_ids.copy()
    rng.shuffle(shuffled_task_ids)
    n_total = len(shuffled_task_ids)

    # Calculate total label counts for this topic
    topic_label_counts = np.array([0, 0, 0])
    for task_id in shuffled_task_ids:
        topic_label_counts += np.array(
            [task_id_label_counts[task_id][i] for i in range(3)]
        )

    # Calculate target label counts per split (60/20/20)
    target_counts = {
        "train": np.round(topic_label_counts * 0.6).astype(int),
        "val": np.round(topic_label_counts * 0.2).astype(int),
        "test": np.round(topic_label_counts * 0.2).astype(int),
    }
    # Adjust for rounding errors - add remainder to train
    for i in range(3):
        diff = topic_label_counts[i] - (
            target_counts["train"][i]
            + target_counts["val"][i]
            + target_counts["test"][i]
        )
        target_counts["train"][i] += diff

    # Calculate number of questions per split
    if n_total <= 2:
        n_train, n_val, n_test = n_total, 0, 0
    elif n_total == 3:
        n_train, n_val, n_test = 1, 1, 1
    else:
        n_test = max(1, round(n_total * 0.2))
        n_val = max(1, round(n_total * 0.2))
        n_train = n_total - n_test - n_val

    # Initialize splits: (task_ids list, current label counts, max questions)
    splits = {
        "train": ([], np.array([0, 0, 0]), n_train),
        "val": ([], np.array([0, 0, 0]), n_val),
        "test": ([], np.array([0, 0, 0]), n_test),
    }

    # Greedy assignment: assign each task_id to minimize label imbalance
    for task_id in shuffled_task_ids:
        task_labels = np.array([task_id_label_counts[task_id][i] for i in range(3)])

        if n_total <= 2:
            best_split = "train"
        else:
            # Find split with space that minimizes distance to target
            best_split = "train"
            best_distance = float("inf")
            for split_name, (task_list, label_counts, max_q) in splits.items():
                if len(task_list) >= max_q:
                    continue
                new_counts = label_counts + task_labels
                distance = np.sum(np.abs(new_counts - target_counts[split_name]))
                if distance < best_distance:
                    best_distance = distance
                    best_split = split_name

        # Assign to chosen split
        task_list, label_counts, _ = splits[best_split]
        task_list.append(task_id)
        splits[best_split] = (
            task_list,
            label_counts + task_labels,
            splits[best_split][2],
        )

    # Extract results
    topic_train = splits["train"][0]
    topic_val = splits["val"][0]
    topic_test = splits["test"][0]

    train_task_ids.extend(topic_train)
    val_task_ids.extend(topic_val)
    test_task_ids.extend(topic_test)

    # Print assignment with label distribution
    train_labels = splits["train"][1]
    val_labels = splits["val"][1]
    test_labels = splits["test"][1]
    print(
        f"Topic '{topic}': {n_total} questions -> "
        f"train={len(topic_train)}, val={len(topic_val)}, test={len(topic_test)} | "
        f"Labels train=({train_labels[0]}/{train_labels[1]}/{train_labels[2]}), "
        f"val=({val_labels[0]}/{val_labels[1]}/{val_labels[2]}), "
        f"test=({test_labels[0]}/{test_labels[1]}/{test_labels[2]})"
    )

train_task_ids = set(train_task_ids)
val_task_ids = set(val_task_ids)
test_task_ids = set(test_task_ids)

print(
    f"\nOverall task_id split: train={len(train_task_ids)}, val={len(val_task_ids)}, test={len(test_task_ids)}"
)

# Calculate and print overall label distribution
print("\n" + "=" * 80)
print("LABEL DISTRIBUTION BY SPLIT")
print("=" * 80)

# Calculate label counts for each split using numpy arrays
split_label_counts = {
    "train": np.array([0, 0, 0]),
    "val": np.array([0, 0, 0]),
    "test": np.array([0, 0, 0]),
}

for split_name, task_ids in [
    ("train", train_task_ids),
    ("val", val_task_ids),
    ("test", test_task_ids),
]:
    for task_id in task_ids:
        split_label_counts[split_name] += np.array(
            [task_id_label_counts[task_id][i] for i in range(3)]
        )

# Print label distribution
label_names = ["incorrect", "partial", "correct"]
for split_name in ["train", "val", "test"]:
    counts = split_label_counts[split_name]
    total = counts.sum()
    print(f"\n{split_name.upper()}:")
    for label_id, label_name in enumerate(label_names):
        count = counts[label_id]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {label_name:12}: {count:6} ({percentage:5.1f}%)")
    print(f"  {'Total':12}: {total:6}")

# Print per-topic label distribution
print("\n" + "=" * 80)
print("LABEL DISTRIBUTION BY TOPIC AND SPLIT")
print("=" * 80)

for topic in sorted(topics_to_task_ids.keys()):
    print(f"\nTopic: {topic}")
    topic_splits = {
        "train": [tid for tid in train_task_ids if task_id_to_topic.get(tid) == topic],
        "val": [tid for tid in val_task_ids if task_id_to_topic.get(tid) == topic],
        "test": [tid for tid in test_task_ids if task_id_to_topic.get(tid) == topic],
    }

    for split_name, topic_task_ids in topic_splits.items():
        if not topic_task_ids:
            continue
        counts = np.array([0, 0, 0])
        for task_id in topic_task_ids:
            counts += np.array([task_id_label_counts[task_id][i] for i in range(3)])
        print(
            f"  {split_name:6}: incorrect={counts[0]:4}, partial={counts[1]:4}, correct={counts[2]:4} (total={counts.sum()})"
        )

print("=" * 80 + "\n")

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
train_dataset = Dataset.from_pandas(
    df_combined.iloc[train_indices].reset_index(drop=True)
)
val_dataset = Dataset.from_pandas(df_combined.iloc[val_indices].reset_index(drop=True))
test_dataset = Dataset.from_pandas(
    df_combined.iloc[test_indices].reset_index(drop=True)
)

# Create final dataset with 6/2/2 split (60% train, 20% val, 20% test)
final_dataset = DatasetDict(
    {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
)

print("Final split sizes:")
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
