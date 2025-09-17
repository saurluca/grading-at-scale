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
    "apertus-8b-instruct-2509": {
        "model": "openrouter/swiss-ai/Apertus-8B-Instruct-2509",
        "api_key": os.getenv("SWISS_API_KEY"),
        "api_base": "https://api.publicai.co/v1",
        # No api_version
    },
    "apertus-70b-instruct-2509": {
        "model": "openrouter/swiss-ai/Apertus-70B-Instruct-2509",
        "api_key": os.getenv("SWISS_API_KEY"),
        "api_base": "https://api.publicai.co/v1",
        # No api_version
    },
}

def build_lm(model_name: str, cache: bool = True, temperature: float = 0.5):
    config = model_configs[model_name]
    lm_kwargs = {
        "model": config.get("model"),
        "api_key": config.get("api_key"),
        "api_base": config.get("api_base"),
        "cache": cache,
        "temperature": temperature,
    }
    # Only add api_version if present in config
    if "api_version" in config:
        lm_kwargs["api_version"] = config.get("api_version")
    return dspy.LM(**lm_kwargs)