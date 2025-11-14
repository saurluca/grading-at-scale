import os
import time
import random
from pathlib import Path

from omegaconf import OmegaConf

from src.finetuning import lora


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
                f"[dispatcher] Model={model_name} | Run {i + 1}/{len(seeds)}: seed={seed}, cfg={out_path}"
            )
            lora.main()


if __name__ == "__main__":
    main()
