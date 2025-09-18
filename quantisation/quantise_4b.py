from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_quant_type="nf4"
)

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    # "bigscience/bloom-1b7",
    "swiss-ai/Apertus-8B-Instruct-2509",
    quantization_config=quantization_config,
)

print("Getting memory footprint")
memory_bytes = model.get_memory_footprint()
memory_gb = memory_bytes / (1024**3)
print(f"Memory footprint: {memory_gb:.2f} GB")


print("Pushing to hub")
model.push_to_hub("saurluca/Apertus-8B-Instruct-2509-bnb-4bit")
