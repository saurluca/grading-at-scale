# %%
from pathlib import Path
import pandas as pd
import yaml
import sys
from datasets import DatasetDict

# Add src to path to import common module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from common import load_and_preprocess_data

# Load configuration
config_path = PROJECT_ROOT / "configs" / "data_generation.yaml"
base_config_path = PROJECT_ROOT / "configs" / "base.yaml"

with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

with open(config_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Merge configurations (data_config takes precedence)
config = {**base_config, **data_config}

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
synth_dir = PROJECT_ROOT / "data" / "gras"
# Parameters
use_unseen_questions = True  # Set to True to split by questions, False for standard split
seed = config["project"]["seed"]

# Load the full dataset
print("Loading full dataset...")
df_full = pd.read_csv(data_dir / "gras" / "full.csv", sep=";")

# Save to temporary file for load_and_preprocess_data function
temp_csv_path = synth_dir / "temp_full.csv"
df_full.to_csv(temp_csv_path, index=False, sep=";")

# %%
# Split dataset using common.py functions
print("Splitting dataset into train/val/test (6/2/2)...")

# First split: 80% train+val, 20% test
raw_data, label_order, label2id, id2label = load_and_preprocess_data(
    dataset_csv=str(temp_csv_path),
    cache_dir=None,
    seed=seed,
    test_size=0.2,  # 20% for test
    use_unseen_questions=use_unseen_questions
)

# Second split: split the 80% into 75% train, 25% val (which gives us 60% train, 20% val of original)
train_val_data = raw_data["train"]
train_val_split = train_val_data.train_test_split(
    test_size=0.25,  # 25% of 80% = 20% of original
    seed=seed,
    stratify_by_column="labels"
)

# Create final dataset with 6/2/2 split
final_dataset = DatasetDict({
    "train": train_val_split["train"],      # 60% of original
    "val": train_val_split["test"],         # 20% of original  
    "test": raw_data["test"]                # 20% of original
})

print(f"Final split sizes:")
print(f"Train: {len(final_dataset['train'])} samples")
print(f"Validation: {len(final_dataset['val'])} samples") 
print(f"Test: {len(final_dataset['test'])} samples")
print(f"Total: {len(final_dataset['train']) + len(final_dataset['val']) + len(final_dataset['test'])} samples")

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