# %%
import torch
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import numpy as np
from transformers import TrainingArguments, Trainer
from torch.nn import functional as F
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score

model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "swiss-ai/Apertus-8B-Instruct-2509"
seed = 42
train_size = 500
eval_size = 100


# Load the model configuration
config = AutoConfig.from_pretrained(model_id)

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Disable KV cache during training to avoid extra memory usage
model.config.use_cache = False

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
        # Use the base transformer to avoid computing large LM logits
        base_outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        last_hidden_state = base_outputs.last_hidden_state[:, -1, :]
        
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
small_eval = dataset["test_ua"].shuffle(seed=seed)

# select subset of data
if train_size is not None:
    small_train = small_train.select(range(train_size))
if eval_size is not None:
    small_eval = small_eval.select(range(eval_size))

# %%

# Data preprocessing function
def preprocess_function(examples):
    prompts = []
    # fill in missing values of prompt
    for i in range(len(examples['question'])):
        prompt = f"""You are a teacher grading a student answers to questions. Return a 1 if the student answer is correct, 0 if it is incorrect.
        \n question: {examples['question'][i]} \n student answer: {examples['student_answer'][i]} \n reference answer: {examples['reference_answer'][i]}"""
        prompts.append(prompt)
    
    # Tokenize the prompts with consistent max_length and padding
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding='max_length',  # Ensure all sequences are padded to max_length
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
    
    # Apply sigmoid to convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-predictions.squeeze()))
    
    # Convert probabilities to binary predictions (0 or 1) for accuracy
    binary_predictions = (probs > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = (binary_predictions == labels).mean()
    
    # Calculate MSE on the raw logits vs labels
    mse = ((predictions.squeeze() - labels) ** 2).mean()
    
    # Clear variables to prevent memory leaks
    del predictions, labels, binary_predictions, probs
    torch.cuda.empty_cache()
    
    return {
        "accuracy": accuracy,
        "mse": mse
    }


trainable_params = [
    param for param in regression_model.parameters() if param.requires_grad
]

# Create a custom optimizer with only the trainable parameters
optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)


# 

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    eval_strategy="epoch",
    push_to_hub=False,
    # learning_rate=5e-5,  # Much more reasonable for fine-tuning
    per_device_train_batch_size=1,  # Smaller batch size for stability
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=1,
    gradient_accumulation_steps=1,
    save_safetensors=False,  # Disable safetensors to avoid shared tensor issues
)

# Print model parameter information
print("\n==== Model Parameter Information ====")
total_params = sum(p.numel() for p in regression_model.parameters())
trainable_params = sum(p.numel() for p in regression_model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {frozen_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Print details of trainable parameters
print("\n==== Trainable Parameter Details ====")
for name, param in regression_model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.numel():,} parameters, shape: {param.shape}")

print("\n" + "="*50)

# Create trainer
trainer = Trainer(
    model=regression_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
)

# Train the model
trainer.train()

# %%
# Final evaluation on eval set
print("\n==== Final Evaluation on Eval Set ====")
pred_output = trainer.predict(tokenized_eval)

preds = pred_output.predictions.squeeze()
labels = pred_output.label_ids.squeeze()

# MSE with raw regression outputs
final_mse = float(np.mean((preds - labels) ** 2))

# Classification metrics: apply sigmoid then threshold at 0.5
probs = 1.0 / (1.0 + np.exp(-preds))
binary_preds = (probs >= 0.5).astype(int)
int_labels = labels.astype(int)

# Compute precision, recall, and F1
final_precision = precision_score(int_labels, binary_preds, average='binary')
final_recall = recall_score(int_labels, binary_preds, average='binary')
final_f1 = f1_score(int_labels, binary_preds, average='binary')

print(f"MSE: {final_mse:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"F1: {final_f1:.4f}")
