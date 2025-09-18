# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field
import outlines
from enum import Enum
from sklearn.metrics import f1_score
from tqdm import tqdm

# Config
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

seed = 42
train_size = 100
eval_size = 10
dataset_name = "Short-Answer-Feedback/saf_communication_networks_english"
hub_model_id = (
    f"saurluca/{MODEL_NAME.split('/')[-1]}-finetuned-saf-communication-networks-english"
)


# Load tokenizer and model
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)


# define output
class Rating(Enum):
    incorrect = 0
    partially_correct = 0.5
    correct = 1


class GradingResult(BaseModel):
    reasoning: str = Field(
        ..., description="Brief explanation for the grade no more then 10 words"
    )
    score: Rating = Field(
        ..., description="Grade as a number: incorrect, partially_correct, or correct"
    )


prompt_template = """question: {question} \n student answer: {answer} \n reference answer: {reference}"""


# %%

print(f"Loading dataset from {dataset_name}...")
dataset = load_dataset(dataset_name)

# Column names: verification_feedback, answer_feedback, score, question, id, reference_answer, provided_answer

# shuffle dataset
small_train = dataset["train"].shuffle(seed=seed)
small_eval = dataset["validation"].shuffle(seed=seed)

# select subset of data
if train_size is not None:
    small_train = small_train.select(range(train_size))
if eval_size is not None:
    small_eval = small_eval.select(range(eval_size))

reverse_rating = {"Incorrect": 0.0, "Partially correct": 0.5, "Correct": 1.0}

new_small_eval = []

# convert verification_feedback to rating string with rever
for example in tqdm(small_eval):
    example["verification_feedback"] = reverse_rating[example["verification_feedback"]]
    new_small_eval.append(example)

small_eval = new_small_eval

print(new_small_eval[0])

# %%

# multi example evaluation

predictions = []
true_scores = []
mistakes = 0

for i in tqdm(range(len(small_eval))):
    example = small_eval[i]

    grading_result = model(
        prompt_template.format(
            question=example["question"],
            answer=example["answer_feedback"],
            reference=example["reference_answer"],
        ),
        GradingResult,
        max_new_tokens=512,
    )
    try:
        grading_result = GradingResult.model_validate_json(grading_result)
    except Exception as e:
        print(f"Error grading example {i}: {e}")
        mistakes += 1
        continue
    predictions.append(str(float(grading_result.score.value)))
    true_scores.append(str(example["verification_feedback"]))

print(f"Mistakes: {mistakes}")

print(f"Predictions: {predictions}")
print(f"True scores: {true_scores}")

print(f"F1 score: {f1_score(true_scores, predictions, average='macro')}")

# %%
