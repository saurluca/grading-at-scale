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

def calculate_label_distance(current_counts, target_counts):
    """Calculate L1 distance between current and target label counts."""
    return sum(abs(current_counts[i] - target_counts[i]) for i in range(3))

def assign_task_id_greedy(task_id, splits_info, target_counts):
    """Assign a task_id to the split that minimizes label distribution distance."""
    task_labels = task_id_label_counts[task_id]
    
    best_split = None
    best_distance = float('inf')
    
    for split_name, split_info in splits_info.items():
        if split_info['assigned'] >= split_info['max_count']:
            continue  # Skip if split is full
        
        # Calculate what counts would be after adding this task_id
        new_counts = {
            i: split_info['label_counts'][i] + task_labels[i]
            for i in range(3)
        }
        
        # Calculate distance to target
        distance = calculate_label_distance(new_counts, target_counts[split_name])
        
        if distance < best_distance:
            best_distance = distance
            best_split = split_name
    
    # Fallback: if no split available (shouldn't happen with proper max_count), use train
    if best_split is None:
        best_split = 'train'
    
    return best_split

for topic, task_ids in sorted(topics_to_task_ids.items()):
    # Shuffle task_ids for this topic
    shuffled_task_ids = task_ids.copy()
    rng.shuffle(shuffled_task_ids)
    
    n_total = len(shuffled_task_ids)
    
    # Calculate total label counts for this topic
    topic_label_counts = {0: 0, 1: 0, 2: 0}
    for task_id in shuffled_task_ids:
        labels = task_id_label_counts[task_id]
        for label_id in range(3):
            topic_label_counts[label_id] += labels[label_id]
    
    # Calculate target label counts for each split (60/20/20)
    target_counts = {
        'train': {i: round(topic_label_counts[i] * 0.6) for i in range(3)},
        'val': {i: round(topic_label_counts[i] * 0.2) for i in range(3)},
        'test': {i: round(topic_label_counts[i] * 0.2) for i in range(3)},
    }
    
    # Adjust targets to sum to total (handle rounding errors)
    for label_id in range(3):
        total_target = target_counts['train'][label_id] + target_counts['val'][label_id] + target_counts['test'][label_id]
        diff = topic_label_counts[label_id] - total_target
        if diff != 0:
            # Add difference to train (largest split)
            target_counts['train'][label_id] += diff
    
    # Calculate target number of questions per split
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
        n_test = max(1, round(n_total * 0.2))
        n_val = max(1, round(n_total * 0.2))
        n_train = n_total - n_test - n_val
    
    # Initialize split tracking
    splits_info = {
        'train': {
            'task_ids': [],
            'label_counts': {0: 0, 1: 0, 2: 0},
            'assigned': 0,
            'max_count': n_train,
        },
        'val': {
            'task_ids': [],
            'label_counts': {0: 0, 1: 0, 2: 0},
            'assigned': 0,
            'max_count': n_val,
        },
        'test': {
            'task_ids': [],
            'label_counts': {0: 0, 1: 0, 2: 0},
            'assigned': 0,
            'max_count': n_test,
        },
    }
    
    # Greedy assignment: assign each task_id to the split that minimizes label imbalance
    for task_id in shuffled_task_ids:
        if n_total <= 2:
            # Small topics: assign all to train
            best_split = 'train'
        elif splits_info['train']['assigned'] < splits_info['train']['max_count'] or \
             splits_info['val']['assigned'] < splits_info['val']['max_count'] or \
             splits_info['test']['assigned'] < splits_info['test']['max_count']:
            # Use greedy assignment
            best_split = assign_task_id_greedy(task_id, splits_info, target_counts)
        else:
            # All splits are full, assign to train as fallback
            best_split = 'train'
        
        # Assign task_id to chosen split
        splits_info[best_split]['task_ids'].append(task_id)
        splits_info[best_split]['assigned'] += 1
        task_labels = task_id_label_counts[task_id]
        for label_id in range(3):
            splits_info[best_split]['label_counts'][label_id] += task_labels[label_id]
    
    topic_train = splits_info['train']['task_ids']
    topic_val = splits_info['val']['task_ids']
    topic_test = splits_info['test']['task_ids']
    
    train_task_ids.extend(topic_train)
    if topic_val:
        val_task_ids.extend(topic_val)
    if topic_test:
        test_task_ids.extend(topic_test)
    
    # Print assignment with label distribution info
    train_labels = splits_info['train']['label_counts']
    val_labels = splits_info['val']['label_counts']
    test_labels = splits_info['test']['label_counts']
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

# Calculate label counts for each split
split_label_counts = {
    'train': {0: 0, 1: 0, 2: 0},
    'val': {0: 0, 1: 0, 2: 0},
    'test': {0: 0, 1: 0, 2: 0},
}

for task_id in train_task_ids:
    labels = task_id_label_counts[task_id]
    for label_id in range(3):
        split_label_counts['train'][label_id] += labels[label_id]

for task_id in val_task_ids:
    labels = task_id_label_counts[task_id]
    for label_id in range(3):
        split_label_counts['val'][label_id] += labels[label_id]

for task_id in test_task_ids:
    labels = task_id_label_counts[task_id]
    for label_id in range(3):
        split_label_counts['test'][label_id] += labels[label_id]

# Print label distribution
label_names = ['incorrect', 'partial', 'correct']
for split_name in ['train', 'val', 'test']:
    counts = split_label_counts[split_name]
    total = sum(counts.values())
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
    topic_train_ids = [tid for tid in train_task_ids if task_id_to_topic.get(tid) == topic]
    topic_val_ids = [tid for tid in val_task_ids if task_id_to_topic.get(tid) == topic]
    topic_test_ids = [tid for tid in test_task_ids if task_id_to_topic.get(tid) == topic]
    
    for split_name, topic_task_ids in [('train', topic_train_ids), ('val', topic_val_ids), ('test', topic_test_ids)]:
        if not topic_task_ids:
            continue
        topic_split_counts = {0: 0, 1: 0, 2: 0}
        for task_id in topic_task_ids:
            labels = task_id_label_counts[task_id]
            for label_id in range(3):
                topic_split_counts[label_id] += labels[label_id]
        total = sum(topic_split_counts.values())
        print(f"  {split_name:6}: incorrect={topic_split_counts[0]:4}, partial={topic_split_counts[1]:4}, correct={topic_split_counts[2]:4} (total={total})")

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
