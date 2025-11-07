import os
import time
import random
import sys
from importlib import import_module
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Add finetuning directory to path so import_module can find lora, base, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def select_main(task_type: str):
    mapping = {
        "lora-classification": "lora",
        "lora-regression": "lora_regression",
        "vanilla-classification": "base",
        "lora-classification-gridsearch": "lora_gridsearch",
    }
    if task_type not in mapping:
        raise ValueError(f"Unknown task_type: {task_type}")
    return import_module(mapping[task_type]).main


def main() -> None:
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    
    # Check if TRAINING_CONFIG_PATH is set (for direct runs or grid search)
    training_cfg_path = os.environ.get("TRAINING_CONFIG_PATH")
    if training_cfg_path and Path(training_cfg_path).exists():
        training_cfg = OmegaConf.load(training_cfg_path)
    else:
        # Default to training.yaml
        training_path = PROJECT_ROOT / "configs" / "training.yaml"
        training_cfg = OmegaConf.load(training_path)
    
    cfg = OmegaConf.merge(base_cfg, training_cfg)

    task_type = str(getattr(cfg, "task_type", "lora-classification"))
    
    # For grid search, skip dispatcher logic and run directly
    if task_type == "lora-classification-gridsearch":
        runner = select_main(task_type)
        runner()
        return

    dispatcher = getattr(cfg, "dispatcher", {})
    models_list = dispatcher.get("models")
    seeds_list = dispatcher.get("seeds")

    if seeds_list:
        seeds = [int(s) for s in seeds_list]
        num_runs = len(seeds)
    else:
        num_runs = int(dispatcher.get("num_runs", 1))
        seed_strategy = str(dispatcher.get("seed_strategy", "same"))
        base_seed = int(getattr(cfg.project, "seed", 42))

        if seed_strategy == "random":
            random.seed(int(time.time()))
            seeds = [random.randint(0, 2**31 - 1) for _ in range(num_runs)]
        else:
            seeds = [base_seed for _ in range(num_runs)]

    run_dir = PROJECT_ROOT / "configs" / ".runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = select_main(task_type)

    # Determine which model names to run: either an explicit list from dispatcher.models
    # or fall back to the single base model in cfg.model.base
    if models_list:
        model_names = [str(m) for m in models_list]
    else:
        model_names = [str(cfg.model.base)]

    for model_name in model_names:
        safe_model = model_name.replace("/", "_").replace(":", "_")
        for i, seed in enumerate(seeds):
            per_run_cfg = OmegaConf.merge(
                cfg,
                {
                    "project": {"seed": int(seed)},
                    "model": {"base": model_name},
                },
            )
            out_path = (
                run_dir / f"dispatcher_run_{safe_model}_{int(time.time())}_{i}.yaml"
            )
            OmegaConf.save(config=per_run_cfg, f=str(out_path))

            os.environ["TRAINING_CONFIG_PATH"] = str(out_path)
            print(
                f"[dispatcher] Model={model_name} | Run {i + 1}/{len(seeds)}: seed={seed}, task={task_type}, cfg={out_path}"
            )
            runner()


if __name__ == "__main__":
    main()
