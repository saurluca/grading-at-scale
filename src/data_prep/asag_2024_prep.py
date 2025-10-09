# %%
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load the ASAG2024 dataset
print("Loading ASAG2024 dataset from Hugging Face...")
dataset = load_dataset("Meyerger/ASAG2024")

# Convert to pandas DataFrame
print("Converting to pandas DataFrame...")
df = dataset["train"].to_pandas()

print(f"Loaded {len(df)} examples")
print(f"Columns: {df.columns.tolist()}")

# Rename columns
print("Renaming columns...")
df = df.rename(columns={"provided_answer": "student_answer", "data_source": "topic"})

# Remove grade column
print("Removing 'grade' column...")
df = df.drop(columns=["grade"])

# Convert normalized_grade to 3-class labels
print("Converting normalized_grade to labels (0, 1, 2)...")


def normalized_grade_to_label(norm_grade):
    """
    Convert normalized grade (0-1) to class label:
    - 0-0.25 → 0 (incorrect)
    - 0.25-0.75 → 1 (partial)
    - 0.75-1.0 → 2 (correct)
    """
    if norm_grade < 0.25:
        return 0
    elif norm_grade < 0.75:
        return 1
    else:
        return 2


df["labels"] = df["normalized_grade"].apply(normalized_grade_to_label)

# Reorder columns to match expected format
print("Reordering columns...")
expected_columns = [
    "index",
    "question",
    "reference_answer",
    "topic",
    "student_answer",
    "normalized_grade",
    "weight",
    "labels",
]
df = df[expected_columns]

# Split into train/val/test (60/20/20)
print("\nSplitting dataset into train/val/test (60/20/20)...")
# First split: 60% train, 40% temp (val+test)
train_df, temp_df = train_test_split(
    df, test_size=0.4, random_state=42, stratify=df["labels"]
)

# Second split: split temp into 50/50 for val and test (each 20% of total)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["labels"]
)

print(f"Train size: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
print(f"Val size: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
print(f"Test size: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
print(f"Total: {len(train_df) + len(val_df) + len(test_df)}")

# Save to separate CSV files
output_dir = PROJECT_ROOT / "data" / "ASAG2024"
output_dir.mkdir(parents=True, exist_ok=True)

print("\nSaving splits to CSV files...")
train_file = output_dir / "train.csv"
val_file = output_dir / "val.csv"
test_file = output_dir / "test.csv"

train_df.to_csv(train_file, index=False, sep=";")
val_df.to_csv(val_file, index=False, sep=";")
test_df.to_csv(test_file, index=False, sep=";")

print(f"✓ Saved train set to {train_file}")
print(f"✓ Saved val set to {val_file}")
print(f"✓ Saved test set to {test_file}")

# Print label distributions for each split
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION BY SPLIT")
print("=" * 60)

for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{name} set:")
    label_counts = split_df["labels"].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count / len(split_df) * 100
        label_name = ["incorrect", "partial", "correct"][label]
        print(f"  {label} ({label_name}): {count:>6} ({percentage:>5.1f}%)")

print("\n" + "=" * 60)
print("Dataset preparation complete!")
