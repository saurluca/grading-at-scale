# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from utils import mse as utils_mse, isfloat as utils_isfloat, extract_model_pred
import numpy as np
import re

# Config
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda"
use_gradient_checkpointing = False
seed = 42
train_size = 100
eval_size = 50
dataset_name = "Short-Answer-Feedback/saf_communication_networks_english"
hub_model_id = "saurluca/gemma-3-270m-finetuned-saf-communication-networks-english"

print(f"Loading tokenizer and model from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Use eager attention implementation for Gemma3 models as recommended
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
    use_cache=(
        not use_gradient_checkpointing
    ),  # Disable cache for gradient checkpointing compatibility
).to(device)

# Fix for models that might not have padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing to save memory
if use_gradient_checkpointing:
    model.gradient_checkpointing_enable()

prompt_template = """You are a helpful assistant that grades student answers to questions. 
Return only the grade. Valid values are 0, 0.5 or 1. 1 is the highest grade.
The question: {question} \n the student answer: {answer} \n the reference answer: {reference}."""

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

# %%


def tokenize_function(examples):
    """Tokenize the text and prepare inputs for the model"""

    prompts = []
    scores = []

    for i in range(len(examples["question"])):
        prompts.append(
            prompt_template.format(
                question=examples["question"][i],
                answer=examples["answer_feedback"][i],
                reference=examples["reference_answer"][i],
            )
        )
        scores.append(examples["score"][i])

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors=None,
    )

    # Store scores for MSE loss calculation
    tokenized["scores"] = scores

    return tokenized


print("\nTokenizing datasets...")
tokenized_train = small_train.map(
    tokenize_function, batched=True, remove_columns=small_train.column_names
)
tokenized_eval = small_eval.map(
    tokenize_function, batched=True, remove_columns=small_eval.column_names
)


# %% RUN SMALL EXAMPLE

example = tokenized_eval[0]
print("Example input:")
print(example)

# Convert input_ids to tensor and add batch dimension
input_ids = torch.tensor([example["input_ids"]]).to(device)
print(f"\nInput shape: {input_ids.shape}")

# Model generation
outputs = model.generate(
    input_ids,
    max_new_tokens=4,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


# Decode the full generated sequence
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nFull generated response: '{full_response}'")

# Decode only the new tokens (what the model generated)
new_tokens = outputs[0][input_ids.shape[1] :]
new_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"\nOnly new tokens: '{new_response}'")

# Show the prompt template used
print(f"\nPrompt template used:")
print(f"True score: {example.get('scores', 'N/A')}")


# %%


def extract_score_from_text(text):
    """Extract numerical score from model output text. Return None if not a clean number in {0,0.5,1}."""
    text = text.strip()
    try:
        # Assume model returns just the number
        val = float(text)
        if val in (0.0, 0.5, 1.0):
            return val
        return None
    except Exception:
        pass
    # Fallback regex patterns
    patterns = [r"^\s*(0|0\.5|1)\s*$"]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            if val in (0.0, 0.5, 1.0):
                return val
    return None


# `%%


def evaluate_model(model, tokenizer, eval_dataset, device):
    """Evaluate model using simple generation + utils helpers.
    - Generate short outputs per example
    - Clean predictions with utils.extract_model_pred
    - Compute MSE (and invalid count) with utils.mse
    - Compute macro F1 by binning values into classes {0, 0.5, 1}
    """

    def bin_score_to_class(val: float) -> int:
        if val < 0.25:
            return 0
        if val < 0.75:
            return 1
        return 2

    model.eval()
    pred_texts = []
    true_texts = []

    with torch.no_grad():
        for i in range(len(eval_dataset)):
            input_ids = torch.tensor([eval_dataset[i]["input_ids"]]).to(device)
            outputs = model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            pred_texts.append(response)
            true_texts.append(str(eval_dataset[i]["scores"]))

    cleaned_preds = extract_model_pred(pred_texts)

    # Convert to numpy arrays for proper indexing
    cleaned_preds_array = np.array(cleaned_preds)
    true_texts_array = np.array(true_texts)

    mse_val, invalid_count = utils_mse(cleaned_preds_array, true_texts_array)

    y_true = []
    y_pred = []
    for t_str, p_str in zip(true_texts, cleaned_preds):
        t_val = float(t_str)
        true_c = bin_score_to_class(t_val)
        if utils_isfloat(p_str):
            p_val = float(p_str)
            pred_c = bin_score_to_class(p_val)
        else:
            pred_c = (true_c + 1) % 3
        y_true.append(true_c)
        y_pred.append(pred_c)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Eval MSE: {mse_val:.4f} | invalid: {invalid_count}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["0", "0.5", "1"], zero_division=0
        )
    )
    return f1, y_pred, y_true


# %%


# Custom trainer with MSE loss
class MSETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Precompute token ids for score strings
        def first_token_id(s):
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) > 0 else None

        self.score_id_map = {
            0.0: first_token_id("0"),
            0.5: first_token_id("0.5"),
            1.0: first_token_id("1"),
        }
        # Fallbacks for spaces
        if self.score_id_map[0.0] is None:
            self.score_id_map[0.0] = first_token_id(" 0")
        if self.score_id_map[0.5] is None:
            self.score_id_map[0.5] = first_token_id(" 0.5")
        if self.score_id_map[1.0] is None:
            self.score_id_map[1.0] = first_token_id(" 1")
        # Keep only available ids
        self.available_scores = [
            v for v in self.score_id_map.keys() if self.score_id_map[v] is not None
        ]
        self.available_ids = [self.score_id_map[v] for v in self.available_scores]
        self.available_values_tensor = torch.tensor(
            self.available_scores, dtype=torch.float32, device=self.model.device
        )

    @property
    def tokenizer(self):
        # Access global tokenizer created above
        return globals()["tokenizer"]

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"].to(self.model.device)
        scores = torch.tensor(
            inputs["scores"], dtype=torch.float32, device=self.model.device
        )
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [B, T, V]
        last_token_logits = logits[:, -1, :]  # [B, V]
        probs = torch.softmax(last_token_logits, dim=-1)  # [B, V]

        if len(self.available_ids) == 0:
            # No identifiable tokens; return loss = 1
            loss = torch.ones((), device=self.model.device)
            return (loss, outputs) if return_outputs else loss

        # Gather probabilities for our score tokens
        ids_tensor = torch.tensor(
            self.available_ids, dtype=torch.long, device=self.model.device
        )
        selected_probs = probs.index_select(dim=1, index=ids_tensor)  # [B, K]
        total_prob = selected_probs.sum(dim=1)  # [B]

        # Expected score conditioned on selecting one of the score tokens
        weighted_sum = (selected_probs * self.available_values_tensor.unsqueeze(0)).sum(
            dim=1
        )  # [B]
        eps = 1e-8
        predicted_scores = weighted_sum / (total_prob + eps)  # [B]

        # Per-sample squared error
        squared_error = (predicted_scores - scores) ** 2  # [B]

        # If the model doesn't put probability mass on any of the score tokens, count as wrong (MSE=1)
        invalid_mask = total_prob < 1e-4
        squared_error = torch.where(
            invalid_mask, torch.ones_like(squared_error), squared_error
        )

        loss = squared_error.mean()
        return (loss, outputs) if return_outputs else loss


# %%

# Evaluate model before training
print("\nEvaluating model before training...")
f1_before, _, _ = evaluate_model(model, tokenizer, tokenized_eval, device)
print(f"F1 Score before training: {f1_before:.4f}")

# %%

# Clear GPU cache before training
torch.cuda.empty_cache()

# Training arguments optimized for memory usage
training_args = TrainingArguments(
    output_dir="output",
    eval_strategy="epoch",
    per_device_train_batch_size=12,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    dataloader_pin_memory=True,
    gradient_checkpointing=use_gradient_checkpointing,
    learning_rate=5e-4,
    num_train_epochs=3,
)

# Create custom trainer
trainer = MSETrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Start training
print("\nStarting training...")
trainer.train()
print("\nTraining completed successfully!")

# %%

# Evaluate model after training
print("\nEvaluating model after training...")
f1_after, predictions, true_scores = evaluate_model(
    model, tokenizer, tokenized_eval, device
)
print(f"F1 Score after training: {f1_after:.4f}")

# Print some example predictions
print("\nExample predictions:")
for i in range(min(5, len(predictions))):
    print(f"True: {true_scores[i]}, Predicted: {predictions[i]}")

# %%

# print("\nPushing model to Hugging Face Hub...")
# model.push_to_hub(hub_model_id)
# tokenizer.push_to_hub(hub_model_id)
