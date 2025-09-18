from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
import torch


# model_name = "saurluca/Apertus-8B-Instruct-2509-bnb-int8"
model_name = "swiss-ai/Apertus-8B-Instruct-2509"
tokenizer_name = "swiss-ai/Apertus-8B-Instruct-2509"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"e
device = "cuda"  # for GPU usage or "cpu" for CPU usage

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

quantization_config = QuantoConfig(weights="int8")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device,
    quantization_config=quantization_config,
)

model = torch.compile(model)

# prepare the model input
prompt = "Give me a brief explanation of gravity in simple terms."
messages_think = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(
    model.device
)

print("Model inputs")

# Generate the output
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

print("Generated ids")

# Get and decode the output
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
