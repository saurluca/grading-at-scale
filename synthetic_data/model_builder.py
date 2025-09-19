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
    "apertus-8b": {
        "model": "openrouter/swiss-ai/apertus-8b-instruct",
        "api_key": os.getenv("SWISS_API_KEY"),
        "api_base": "https://api.publicai.co/v1",
    },
    "apertus-70b": {
        "model": "openrouter/swiss-ai/apertus-70b-instruct",
        "api_key": os.getenv("SWISS_API_KEY"),
        "api_base": "https://api.publicai.co/v1",
    },
    "DeepSeek-V3.1": {
        "model": "huggingface/fireworks-ai/deepseek-ai/DeepSeek-V3.1",
        "api_key": os.getenv("HF_TOKEN"),
    },
    "gpt-oss-120b": {
        "model": "huggingface/cerebras/openai/gpt-oss-120b",
        "api_key": os.getenv("HF_TOKEN"),
    },
    "llama-4": {
        "model": "openrouter/meta-llama/llama-4-scout:free",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "gpt-5-mini": {
        "model": "openrouter/openai/gpt-5-mini",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "gpt-oss-20b": {
        "model": "openrouter/openai/gpt-oss-20b:free",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "aion-llama": {
        "model": "openrouter/aion-labs/aion-rp-llama-3.1-8b",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
}


def build_lm(
    model_name: str, cache: bool = True, temperature: float = 1.0, max_tokens: int = 512
):
    config = model_configs[model_name]
    lm_kwargs = {
        "model": config.get("model"),
        "api_key": config.get("api_key"),
        "api_base": config.get("api_base"),
        "max_tokens": max_tokens,
        "cache": cache,
        "temperature": temperature,
    }
    # Only add api_version if present in config
    if "api_version" in config:
        lm_kwargs["api_version"] = config.get("api_version")
    if "api_base" in config:
        lm_kwargs["api_base"] = config.get("api_base")

    return dspy.LM(**lm_kwargs)
