# %%
import os
import pandas as pd
import dspy
from sklearn.metrics import f1_score
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ROARS")

mlflow.dspy.autolog()

lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)


# Roars datase: https://github.com/owenhenkel/ROARS-dataset/blob/main/ROARS_datadictionary.csv
file_path = os.path.join(os.path.dirname(__file__), "data", "ROARS.csv")

# read in data
data = pd.read_csv(
    file_path,
    usecols=[
        "story",
        "story_text",
        "passage_text",
        "question_num",
        "question_text",
        "student_answer",
        "2class_human_groundtruth",
        "2class_best_model",
    ],
)

# %%
# print number of unique values for each column
print(data.nunique())
# print correlation matrix for only numeric columns
print(data.select_dtypes(include=["number"]).corr())

# print number of rows
print(f"Number of rows: {len(data)}")
# print nubmer of true and false labels
print(f"Number of true labels: {data['2class_human_groundtruth'].sum()}")
print(f"Number of false labels: {len(data) - data['2class_human_groundtruth'].sum()}")

data.head()


# %%


class RoarsGrader(dspy.Signature):
    """You are a teacher of childern aged 9-21.
    You're job is to grade text comprehension questions.
    Decide if the student correctly answered the question based on the passage.
    """

    passage_text: str = dspy.InputField(description="The passage text")
    question_text: str = dspy.InputField(description="The question text")
    student_answer: str = dspy.InputField(description="The student answer")

    label: bool = dspy.OutputField(
        description="True if the student answer is correct, False if the student answer is incorrect"
    )


def convert_to_dspy_format(dataset_split: pd.DataFrame):
    examples = []
    for idx, example in dataset_split.iterrows():
        example = dspy.Example(
            question_text=example["question_text"],
            passage_text=example["passage_text"],
            student_answer=example["student_answer"],
            label=example["2class_human_groundtruth"],
        ).with_inputs("question_text", "passage_text", "student_answer")
        examples.append(example)
    return examples


grader = dspy.Predict(RoarsGrader)


def metric(
    gold,
    prediction,
    trace=None,
):
    return int(gold.label) == int(prediction.label)


# %%

n_train = 200
n_val = 100

# Sample train set
train_set = data.sample(n=n_train, random_state=42)
# Remove train indices from data, then sample val set
remaining_data = data.drop(train_set.index)
val_set = remaining_data.sample(n=n_val, random_state=43)

# Reset indices
train_set = train_set.reset_index(drop=True)
val_set = val_set.reset_index(drop=True)

train_set = convert_to_dspy_format(train_set)
val_set = convert_to_dspy_format(val_set)


# %%

# y_true = []
# y_pred = []

# for example in val_set:
#     prediction = grader(
#         passage_text=example.passage_text,
#         question_text=example.question_text,
#         student_answer=example.student_answer,
#     )
#     y_true.append(example.label)
#     y_pred.append(prediction.label)

# accuracy = sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_true)
# f1 = f1_score(y_true, y_pred)

# print(f"Accuracy: {accuracy}")
# print(f"F1 score: {f1:.2f}")
# print(
#     f"Number of correct predictions: {sum([yt == yp for yt, yp in zip(y_true, y_pred)])} out of {len(y_true)}"
# )

# %%

baseline_evaluator = dspy.Evaluate(
    devset=val_set, metric=metric, display_progress=True, display_table=5, num_threads=4
)

baseline_evaluator(grader)

# %%
# config = dict(
#     max_bootstrapped_demos=4,
#     max_labeled_demos=4,
#     num_candidate_programs=4,
#     num_threads=4,
# )

# teleprompter = dspy.BootstrapFewShotWithRandomSearch(metric=metric, **config)
# grader_optimized = teleprompter.compile(grader, trainset=train_set)

config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    # num_candidate_programs=4,
    # num_threads=4,
)

teleprompter = dspy.BootstrapFewShot(metric=metric, **config)
grader_optimized = teleprompter.compile(grader, trainset=train_set)

baseline_evaluator(grader_optimized)

# Major improvement from 51 to 68

# %%

dspy.settings.experimental = True


optimizer = dspy.BootstrapFinetune(
    num_threads=16,
    metric=metric,
)  # if you *do* have labels, pass metric=your_metric here!
grader_ft = optimizer.compile(grader, trainset=train_set)

grader_ft.get_lm().launch()

# %%
baseline_evaluator(grader_ft)
