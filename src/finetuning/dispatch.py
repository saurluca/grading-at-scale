import os
import sys
import time
import random
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.finetuning import lora  # noqa: E402


def main() -> None:
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")

    # Check if TRAINING_CONFIG_PATH is set (for direct runs)
    training_cfg_path = os.environ.get("TRAINING_CONFIG_PATH")
    if training_cfg_path and Path(training_cfg_path).exists():
        training_cfg = OmegaConf.load(training_cfg_path)
    else:
        # Default to training.yaml
        training_path = PROJECT_ROOT / "configs" / "training.yaml"
        training_cfg = OmegaConf.load(training_path)

    cfg = OmegaConf.merge(base_cfg, training_cfg)

    dispatcher = getattr(cfg, "dispatcher", {})
    models_list = dispatcher.get("models")
    seeds_list = dispatcher.get("seeds")

    if not models_list:
        raise ValueError("dispatcher.models must be provided (list of model names)")

    if seeds_list:
        seeds = [int(s) for s in seeds_list]
        num_runs = len(seeds)
    else:
        num_runs = int(dispatcher.get("num_runs", 1))
        # Always use random seeds if seeds not provided
        random.seed(int(time.time()))
        seeds = [random.randint(0, 2**31 - 1) for _ in range(num_runs)]

    run_dir = PROJECT_ROOT / "configs" / ".runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Always use models list from dispatcher
    model_names = [str(m) for m in models_list]

    # Get model-specific parameters if available
    model_specific_params = getattr(cfg, "model_specific_params", {})

    for model_name in model_names:
        safe_model = model_name.replace("/", "_").replace(":", "_")
        for i, seed in enumerate(seeds):
            # Start with base config updates
            per_run_updates = {
                "project": {"seed": int(seed)},
                "model": {"base": model_name},
            }
            
            # Apply model-specific parameters if available
            if model_name in model_specific_params:
                model_params = model_specific_params[model_name]
                # Initialize training updates dict
                training_updates = {}
                
                if "batch_size" in model_params:
                    training_updates["batch_size"] = model_params["batch_size"]
                if "learning_rate" in model_params:
                    training_updates["learning_rate"] = model_params["learning_rate"]
                
                if training_updates:
                    per_run_updates["training"] = training_updates
                    print(
                        f"[dispatcher] Using model-specific params for {model_name}: "
                        f"batch_size={model_params.get('batch_size', {}).get('train', 'default')}, "
                        f"lr={model_params.get('learning_rate', 'default')}"
                    )
            
            per_run_cfg = OmegaConf.merge(cfg, per_run_updates)
            out_path = (
                run_dir / f"dispatcher_run_{safe_model}_{int(time.time())}_{i}.yaml"
            )
            OmegaConf.save(config=per_run_cfg, f=str(out_path))

            os.environ["TRAINING_CONFIG_PATH"] = str(out_path)
            print(
                f"[dispatcher] Model={model_name} | Run {i + 1}/{len(seeds)}: seed={seed}, cfg={out_path}"
            )
            lora.main()


if __name__ == "__main__":
    main()
