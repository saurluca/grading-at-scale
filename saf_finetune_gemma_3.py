# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Config
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "google/gemma-3-270m"
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
    use_cache=(not use_gradient_checkpointing)  # Disable cache for gradient checkpointing compatibility
).to(device)

# Fix for models that might not have padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing to save memory
if use_gradient_checkpointing:
    model.gradient_checkpointing_enable()

prompt_template = "justify and grade: question: {question} student: {answer} reference: {reference}"
    
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
    for i in range(len(examples["question"])):
        prompts.append(prompt_template.format(
            question=examples["question"][i], 
            answer=examples["answer_feedback"][i], 
            reference=examples["reference_answer"][i]
        ))
    
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=512,  # Reduced from 512 to save memory
        return_tensors=None
    )
    
    # Create labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("\nTokenizing datasets...")
tokenized_train = small_train.map(tokenize_function, batched=True, remove_columns=small_train.column_names)
tokenized_eval = small_eval.map(tokenize_function, batched=True, remove_columns=small_eval.column_names)

# %%

# use MSE
def metric(predictions, references):
    pass

    


# %%

# Clear GPU cache before training
torch.cuda.empty_cache()

# Training arguments optimized for memory usage
training_args = TrainingArguments(
    output_dir="output",
    eval_strategy="epoch",
    per_device_train_batch_size=12,  # Keep batch size at 1
    per_device_eval_batch_size=16,   # Keep eval batch size at 1
    gradient_accumulation_steps=1,  # Reduced from 8 to 1 to save memory
    # fp16=True,                       # Enable mixed precision to save memory
    dataloader_pin_memory=True,     # Disable pin memory to save GPU memory
    gradient_checkpointing=use_gradient_checkpointing,     # Enable gradient checkpointing
    learning_rate=5e-4,
    num_train_epochs=3,             # Reduced to 1 epoch for testing
    remove_unused_columns=False,    # Important for custom datasets
    ddp_find_unused_parameters=False, # Disable unused parameter detection
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

# print("\nPushing model to Hugging Face Hub...")
# model.push_to_hub(hub_model_id)
# tokenizer.push_to_hub(hub_model_id)