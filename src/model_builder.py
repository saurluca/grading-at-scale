import os
from dotenv import load_dotenv
import dspy

load_dotenv()


model_configs = {
    "gpt-4o-mini": {
        "model": "azure/gpt-4o-mini",
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_base": os.getenv("AZURE_API_BASE"),
        "api_version": "2024-12-01-preview",
    },
    "gpt-4o": {
        "model": "azure/gpt-4o",
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_base": os.getenv("AZURE_API_BASE"),
        "api_version": "2024-12-01-preview",
    },
    "llama3.2:3b": {
        "model": "ollama_chat/llama3.2:3b",
        "api_key": "",
        "api_base": os.getenv("OLLAMA_API_BASE"),
    },
    "llama3.2:1b": {
        "model": "ollama_chat/llama3.2:1b",
        "api_key": "",
        "api_base": os.getenv("OLLAMA_API_BASE"),
    },
    "qwen3:0.6b": {
        "model": "hosted_vllm/Qwen/Qwen3-0.6B",
        "api_key": "",
        "api_base": os.getenv("VLLM_API_BASE"),
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model": "hosted_vllm/meta-llama/Llama-3.2-1B-Instruct",
        "api_key": "",
        "api_base": os.getenv("VLLM_API_BASE"),
    },
    "openai-community/gpt2-large": {
        "model": "hosted_vllm/openai-community/gpt2-large",
        "api_key": "",
        "api_base": os.getenv("VLLM_API_BASE"),
    },
    "openai-community/gpt2": {
        "model": "hosted_vllm/openai-community/gpt2",
        "api_key": "",
        "api_base": os.getenv("VLLM_API_BASE"),
    },
    "google/flan-t5-base": {
        "model": "hosted_vllm/google/flan-t5-base",
        "api_key": "",
        "api_base": os.getenv("VLLM_API_BASE"),
    },
    "Qwen/Qwen3-0.6B": {
        "model": "hosted_vllm/Qwen/Qwen3-0.6B",
        "api_key": "",
        "api_base": os.getenv("VLLM_API_BASE"),
    },
}


def build_lm(
    model_name: str,
    cache: bool = True,
    temperature: float | None = None,
    max_tokens: int = 512,
):
    config = model_configs[model_name]
    lm_kwargs = {
        "model": config.get("model"),
        "api_key": config.get("api_key"),
        "api_base": config.get("api_base"),
        "max_tokens": max_tokens,
        "cache": cache,
    }
    if temperature is not None:
        lm_kwargs["temperature"] = temperature
    # Only add api_version if present in config
    if "api_version" in config:
        lm_kwargs["api_version"] = config.get("api_version")
    if "api_base" in config:
        lm_kwargs["api_base"] = config.get("api_base")
    

    return dspy.LM(**lm_kwargs)
