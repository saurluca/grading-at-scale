# %%
from datasets import load_dataset, ClassLabel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load the dataset
dataset = load_dataset("nkazi/SciEntsBank")

# Align labels with mapping
dataset = dataset.align_labels_with_mapping(
    {
        "correct": 2,
        "contradictory": 0,
        "partially_correct_incomplete": 1,
        "irrelevant": 0,
        "non_domain": 0,
    },
    "label",
)
dataset = dataset.cast_column(
    "label", ClassLabel(names=["incorrect", "partial", "correct"])
)

# Rename columns to match expected format
dataset = dataset.rename_column("id", "task_id")
dataset = dataset.rename_column("label", "labels")


# Add missing columns
def add_chunk_text(example):
    example["chunk_text"] = ""
    return example


def add_topic(example):
    example["topic"] = "scientsbank"
    return example


dataset = dataset.map(add_chunk_text)
dataset = dataset.map(add_topic)

# Save each split as CSV with semicolon separator
output_dir = PROJECT_ROOT / "data" / "SciEntsBank_3way"
output_dir.mkdir(parents=True, exist_ok=True)

for split_name, split_data in dataset.items():
    # Convert to pandas DataFrame
    df = split_data.to_pandas()

    # Ensure columns are in the expected order
    expected_columns = [
        "task_id",
        "question",
        "reference_answer",
        "chunk_text",
        "topic",
        "student_answer",
        "labels",
    ]
    df = df[expected_columns]

    # Save as CSV with semicolon separator
    output_file = output_dir / f"{split_name}.csv"
    df.to_csv(output_file, index=False, sep=";")
    print(f"Saved {split_name} split to {output_file} with {len(df)} examples")

# %%

# from datasets import DatasetDict


# def print_label_dist(dataset):
#     for split_name in dataset:
#         print(split_name, ":")
#         num_examples = 0
#         for label in dataset[split_name].features["label"].names:
#             count = dataset[split_name]["label"].count(
#                 dataset[split_name].features["label"].str2int(label)
#             )
#             print(" ", label, ":", count)
#             num_examples += count
#         print("  total :", num_examples)


# dataset = DatasetDict.load_from_disk("../data/SciEntsBank_3way")

# print_label_dist(dataset)
