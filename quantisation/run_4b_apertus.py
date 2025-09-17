from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "saurluca/Apertus-8B-Instruct-2509-bnb-4bit"
# model_name = "swiss-ai/Apertus-8B-Instruct-2509"
tokenizer_name = "swiss-ai/Apertus-8B-Instruct-2509"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    trust_remote_code=True,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=quantization_config,
).to(device)

# prepare the model input
prompt = "Give me a brief explanation of gravity in simple terms."
messages_think = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

# Generate the output
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

# Get and decode the output
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)