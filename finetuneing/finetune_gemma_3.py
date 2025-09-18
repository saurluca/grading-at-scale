# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Config
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "google/gemma-3-270m"
device = "cuda"
use_gradient_checkpointing = True
seed = 42
train_size = 100
eval_size = 100
hub_model_id = "saurluca/gemma-3-270m-finetuned-yelp-review-classifier"


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


# Define the prompt template (simplified for models without chat templates)
prompt = """You are a helpful assistant that classifies Yelp reviews based on their expected number of stars.

The review is:
{review}

Only return the number of stars, no other text. Answer should be a number between 0 and 4.
Answer:"""

# %%

print(f"Loading dataset from yelp_review_full...")
dataset = load_dataset("yelp_review_full")

# shuffle and reduce the dataset to the size of train_size and eval_size
small_train = dataset["train"].shuffle(seed=seed).select(range(train_size))
small_eval = dataset["test"].shuffle(seed=seed).select(range(eval_size))

# %%


def tokenize_function(examples):
    """Tokenize the text and prepare inputs for the model"""
    prompts = [prompt.format(review=text) for text in examples["text"]]

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=512,  # Reduced from 512 to save memory
        return_tensors=None,
    )

    # Create labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


print("\nTokenizing datasets...")
tokenized_train = small_train.map(
    tokenize_function, batched=True, remove_columns=small_train.column_names
)
tokenized_eval = small_eval.map(
    tokenize_function, batched=True, remove_columns=small_eval.column_names
)

# %%

# Clear GPU cache before training
torch.cuda.empty_cache()

# Training arguments optimized for memory usage
training_args = TrainingArguments(
    output_dir="yelp_review_classifier",
    eval_strategy="epoch",
    per_device_train_batch_size=1,  # Keep batch size at 1
    per_device_eval_batch_size=1,  # Keep eval batch size at 1
    gradient_accumulation_steps=4,  # Reduced from 8 to 1 to save memory
    fp16=True,  # Enable mixed precision to save memory
    dataloader_pin_memory=True,  # Disable pin memory to save GPU memory
    gradient_checkpointing=True,  # Enable gradient checkpointing
    # optim="adamw_torch",            # Use PyTorch optimizer for better memory management
    learning_rate=5e-5,
    num_train_epochs=1,  # Reduced to 1 epoch for testing
    # save_total_limit=1,             # Only keep 1 checkpoint to save disk space
    remove_unused_columns=False,  # Important for custom datasets
    ddp_find_unused_parameters=False,  # Disable unused parameter detection
)

# Create trainer
trainer = Trainer(
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

print("\nPushing model to Hugging Face Hub...")
model.push_to_hub(hub_model_id)
tokenizer.push_to_hub(hub_model_id)
