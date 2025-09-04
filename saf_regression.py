# %%
import torch
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from torch.nn import functional as F
import os
import json

model_id = "meta-llama/Llama-3.2-1B-Instruct"
seed = 42
train_size = 2000
eval_size = 300


# Load the model configuration
config = AutoConfig.from_pretrained(model_id)

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set padding token for the tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create a custom model class for regression
class RegressionModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Create a separate regression head that doesn't interfere with base model
        self.regression_head = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=1,
            bias=False
        )
        
        # Freeze the base model layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Only train the regression head
        self.regression_head.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get the last hidden state from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]  # Get the last token's hidden state
        
        # Pass through the regression head
        logits = self.regression_head(last_hidden_state)
        
        loss = None
        if labels is not None:
            # Ensure proper tensor dimensions and use MSE loss for regression
            # Squeeze logits to match labels shape, but keep batch dimension
            if logits.dim() > 1:
                logits = logits.squeeze(-1)  # Remove last dimension if it's 1
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)  # Add batch dimension if scalar
            loss = F.mse_loss(logits, labels.float())
        
        return {"loss": loss, "logits": logits}
    
    def save_pretrained(self, save_directory, **kwargs):
        """Custom save method to avoid shared tensor issues"""
        # Create a temporary state dict without shared tensors
        state_dict = self.state_dict()
        
        # Remove the base model's lm_head from state dict to avoid conflicts
        keys_to_remove = [k for k in state_dict.keys() if 'base_model.lm_head' in k]
        for key in keys_to_remove:
            del state_dict[key]
        
        # Ensure all tensors have proper dimensions (not scalar)
        for key, tensor in state_dict.items():
            if tensor.dim() == 0:  # If tensor is scalar
                state_dict[key] = tensor.unsqueeze(0)  # Add dimension
        
        # Save the model
        os.makedirs(save_directory, exist_ok=True)
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        config_dict = {
            "model_type": "regression_llama",
            "hidden_size": config.hidden_size,
            "architectures": ["RegressionModel"]
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f)

# Create the regression model
regression_model = RegressionModel(model)

# %%

# 1 is correct, 0 is incorrect
print(f"Loading dataset from SciEntsBank_2way...")
dataset = DatasetDict.load_from_disk('SciEntsBank_2way')

# shuffle dataset
small_train = dataset["train"].shuffle(seed=seed)
small_eval = dataset["test_ud"].shuffle(seed=seed)

# select subset of data
if train_size is not None:
    small_train = small_train.select(range(train_size))
if eval_size is not None:
    small_eval = small_eval.select(range(eval_size))

# %%

# Data preprocessing function
def preprocess_function(examples):
    prompts = []
    for i in range(len(examples['question'])):
        prompt = f"You are a teacher grading a student answers to questions. Return a 1 if the student answer is correct, 0 if it is incorrect."
        prompt += f"\n question: {examples['question'][i]} \n student answer: {examples['student_answer'][i]} \n reference answer: {examples['reference_answer'][i]}"
        prompts.append(prompt)
    
    # Tokenize the prompts
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    )
    
    # Convert labels to float for regression
    labels = torch.tensor([float(label) for label in examples['label']])
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# Preprocess the datasets
tokenized_train = small_train.map(preprocess_function, batched=True, remove_columns=small_train.column_names)
tokenized_eval = small_eval.map(preprocess_function, batched=True, remove_columns=small_eval.column_names)

# %%

torch.cuda.empty_cache()

# Custom metric for regression
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert predictions to binary (0 or 1) for accuracy
    binary_predictions = (predictions.squeeze() > 0.5).astype(int)
    # Calculate accuracy
    accuracy = (binary_predictions == labels).mean()
    # Calculate MSE
    mse = ((predictions.squeeze() - labels) ** 2).mean()
    
    return {
        "accuracy": accuracy,
        "mse": mse
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    eval_strategy="epoch",
    push_to_hub=False,
    learning_rate=0.01,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=1,
    save_safetensors=False,  # Disable safetensors to avoid shared tensor issues
)

# Create trainer
trainer = Trainer(
    model=regression_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# %%

torch.cuda.empty_cache()

# Test on a single example
example = small_eval[0]
prompt = f"You are a teacher grading a student answers to questions. Return a 1 if the student answer is correct, 0 if it is incorrect."
prompt += f"\n question: {example['question']} \n student answer: {example['student_answer']} \n reference answer: {example['reference_answer']}"

tokenized_example = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")

# Get prediction
with torch.no_grad():
    outputs = regression_model(**tokenized_example)
    prediction = torch.sigmoid(outputs["logits"]).item()

print(f"Generated prediction: {prediction:.4f}")
print(f"Expected: 1 if correct, 0 if incorrect")
print(f"Actual answer: {example['label']}")

