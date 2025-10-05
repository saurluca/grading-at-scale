# %%
from datasets import load_dataset, ClassLabel

dataset = load_dataset("nkazi/SciEntsBank")


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

# columns to have: task_id; question; reference_answer; chunk_text; topic; student_answer; label

# rename id to task_id
dataset = dataset.rename_column("id", "task_id")
dataset = dataset.rename_column("label", "labels")


# add column chunk_text with ""
def add_chunk_text(example):
    example["chunk_text"] = ""
    return example


dataset = dataset.map(add_chunk_text)


# add column topic with "scientsbank"
def add_topic(example):
    example["topic"] = "scientsbank"
    return example


dataset = dataset.map(add_topic)

dataset.save_to_disk("../data/SciEntsBank_3way")

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
