from datasets import load_dataset
from datasets import ClassLabel
from pathlib import Path

# Load dataset from HuggingFace
dataset = load_dataset("nkazi/SciEntsBank")

# Verify we have the expected splits
print(f"Dataset splits: {list(dataset.keys())}")
for split_name in dataset.keys():
    print(f"  {split_name}: {len(dataset[split_name])} samples")

# Convert 5-way labels to 3-way classification
# Mapping: correct → 2, partially_correct_incomplete → 1, others → 0
dataset = dataset.align_labels_with_mapping(
    {
        "correct": 2,
        "partially_correct_incomplete": 1,
        "contradictory": 0,
        "irrelevant": 0,
        "non_domain": 0,
    },
    "label",
)

# Cast to ClassLabel with names matching numeric indices (0, 1, 2)
dataset = dataset.cast_column(
    "label", ClassLabel(names=["incorrect", "partially correct", "correct"])
)

# Create output directory
output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "SciEntsBank_3way"
output_dir.mkdir(parents=True, exist_ok=True)

# Transform columns and save each split to CSV
for split_name in dataset.keys():
    # Convert to pandas DataFrame first
    df = dataset[split_name].to_pandas()

    # Rename columns: id → task_id, label → labels
    df = df.rename(columns={"id": "task_id", "label": "labels"})

    # Add topic column with value "scientsbank"
    df["topic"] = "scientsbank"

    # Reorder columns to match expected format: task_id, question, reference_answer, topic, student_answer, labels
    df = df[
        ["task_id", "question", "reference_answer", "topic", "student_answer", "labels"]
    ]

    # Save to CSV with semicolon separator
    output_path = output_dir / f"{split_name}.csv"
    df.to_csv(output_path, index=False, sep=";")
    print(f"Saved {split_name} split ({len(df)} samples) to {output_path}")

print("\nDataset preparation completed!")
