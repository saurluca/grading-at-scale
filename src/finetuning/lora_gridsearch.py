# %%
import os
from pathlib import Path
import sys
import mlflow
import random
import time
from itertools import product
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.common import (  # noqa: E402
    setup_training_args,
    setup_trainer,
    load_and_preprocess_data,
    tokenize_dataset,
    detailed_evaluation,
    setup_model_and_tokenizer,
)
from src.mlflow_config import setup_mlflow  # noqa: E402


def run_single_training(
    cfg, cache_dir, output_dir_base, combination_idx, total_combinations
):
    """Run a single training run with given config."""
    model_name: str = str(cfg.model.base)

    # Create unique output directory for this combination
    output_dir = os.path.join(output_dir_base, f"combination_{combination_idx}")
    os.makedirs(output_dir, exist_ok=True)

    # Use separate train/val/test files
    dataset_base_path = PROJECT_ROOT / "data" / cfg.dataset.dataset_name
    train_csv = str(dataset_base_path / getattr(cfg.dataset, "train_file", "train.csv"))
    val_csv = str(dataset_base_path / getattr(cfg.dataset, "val_file", "val.csv"))
    test_csv = str(dataset_base_path / getattr(cfg.dataset, "test_file", "test.csv"))
    dataset_csv = train_csv  # For logging purposes

    # Load and preprocess data
    raw_data, label_order, label2id, id2label = load_and_preprocess_data(
        cache_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
    )

    # Setup model and tokenizer
    tokenizer, base_model = setup_model_and_tokenizer(
        model_name, label2id, id2label, cache_dir
    )

    # Setup LoRA model
    print("Setting up LoRA configuration and applying to base model...")
    lora_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=cfg.lora.target_modules,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()

    # Tokenize dataset
    include_ref_ans = bool(getattr(cfg.tokenization, "include_reference_answer", False))
    tokenized_data = tokenize_dataset(raw_data, tokenizer, include_ref_ans)

    # Setup training arguments and trainer
    training_args = setup_training_args(cfg, output_dir)
    trainer = setup_trainer(model, training_args, tokenized_data, tokenizer, cfg)

    # Training
    print("Starting training...")
    eval_set_name = "validation" if "val" in tokenized_data else "test"
    print(f"Evaluating on {eval_set_name} set before training...")
    initial_metrics = trainer.evaluate()
    print(f"Metrics before training: {initial_metrics}")

    trainer.train()

    # Get best validation metrics (from training history)
    # The trainer stores the best metrics
    best_metrics = (
        trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None
    )

    # Evaluate on validation set to get final metrics
    final_eval_metrics = trainer.evaluate()

    # Perform detailed evaluation on test set
    print("\nPerforming detailed evaluation on test dataset...")
    detailed_metrics = detailed_evaluation(trainer, tokenized_data["test"], label_order)

    return {
        "initial_metrics": initial_metrics,
        "final_eval_metrics": final_eval_metrics,
        "detailed_metrics": detailed_metrics,
        "model": model,
        "tokenizer": tokenizer,
    }


def run_grid_search_with_seed(cfg, seed: int) -> dict:
    """Run a single grid search with a specific seed."""
    # Set seed in config
    cfg.project.seed = seed

    # Extract grid search parameters
    grid_search = cfg.grid_search
    learning_rates = list(grid_search.learning_rate)
    lora_rs = list(grid_search.lora_r)
    lora_alpha_ratios = list(grid_search.lora_alpha_ratios)
    lora_dropouts = list(grid_search.lora_dropout)
    batch_sizes = list(grid_search.batch_size)
    optimization_metric = str(grid_search.optimization_metric)

    # Generate all combinations
    combinations = list(
        product(
            learning_rates,
            lora_rs,
            lora_alpha_ratios,
            lora_dropouts,
            batch_sizes,
        )
    )

    total_combinations = len(combinations)
    print(f"\n{'=' * 60}")
    print(f"Grid Search (Seed: {seed}): {total_combinations} combinations to evaluate")
    print(f"{'=' * 60}")
    print(f"Learning rates: {learning_rates}")
    print(f"LoRA r: {lora_rs}")
    print(f"LoRA alpha ratios: {lora_alpha_ratios}")
    print(f"LoRA dropout: {lora_dropouts}")
    print(f"Training batch sizes: {batch_sizes}")
    print(f"Optimization metric: {optimization_metric}")
    print(f"Seed: {seed}")
    print(f"{'=' * 60}\n")

    model_name: str = str(cfg.model.base)
    # Make output directory unique per seed
    base_output_dir = str(PROJECT_ROOT / cfg.output.dir)
    output_dir_base = os.path.join(base_output_dir, f"seed_{seed}")
    cache_dir: str | None = str(cfg.paths.hf_cache_dir) if "paths" in cfg else None

    os.makedirs(output_dir_base, exist_ok=True)
    if cache_dir:
        cache_path = (
            os.path.join(PROJECT_ROOT, cache_dir)
            if not os.path.isabs(cache_dir)
            else cache_dir
        )
        os.makedirs(cache_path, exist_ok=True)
    else:
        cache_path = None

    best_combination = None
    best_metric_value = None
    all_results = []

    # Create nested MLflow run for this seed
    run_name = f"gridsearch_seed_{seed}_{model_name.split('/')[-1]}"
    with mlflow.start_run(run_name=run_name, nested=True):
        # Log grid search parameters for this seed
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_name": str(cfg.dataset.dataset_name),
                "optimization_metric": optimization_metric,
                "seed": seed,
                "total_combinations": total_combinations,
                "learning_rates": str(learning_rates),
                "lora_rs": str(lora_rs),
                "lora_alpha_ratios": str(lora_alpha_ratios),
                "lora_dropouts": str(lora_dropouts),
                "batch_sizes": str(batch_sizes),
            }
        )

        # Run each combination
        for idx, (lr, r, alpha_ratio, dropout, batch_size) in enumerate(
            combinations
        ):
            print(f"\n{'=' * 60}")
            print(f"Combination {idx + 1}/{total_combinations}")
            print(f"{'=' * 60}")
            print(f"Learning rate: {lr}")
            print(f"LoRA r: {r}")
            print(f"LoRA alpha ratio: {alpha_ratio}")
            print(f"LoRA alpha: {r * alpha_ratio}")
            print(f"LoRA dropout: {dropout}")
            print(f"Training batch size: {batch_size}")
            print(f"{'=' * 60}\n")

            # Create modified config for this combination
            combination_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
            combination_cfg.training.learning_rate = float(lr)
            combination_cfg.lora.r = int(r)
            combination_cfg.lora.alpha = int(r * alpha_ratio)
            combination_cfg.lora.dropout = float(dropout)
            combination_cfg.training.batch_size.train = int(batch_size)

            # Create nested MLflow run for this combination
            run_name = f"lr_{lr}_r_{r}_alpha_{r * alpha_ratio:.1f}_drop_{dropout}_bs_{batch_size}"
            with mlflow.start_run(run_name=run_name, nested=True):
                try:
                    # Log hyperparameters
                    mlflow.log_params(
                        {
                            "learning_rate": lr,
                            "lora_r": r,
                            "lora_alpha_ratio": alpha_ratio,
                            "lora_alpha": int(r * alpha_ratio),
                            "lora_dropout": dropout,
                            "batch_size": batch_size,
                            "combination_idx": idx + 1,
                            "seed": seed,
                        }
                    )

                    # Run training
                    result = run_single_training(
                        combination_cfg,
                        cache_path if cache_path else None,
                        output_dir_base,
                        idx,
                        total_combinations,
                    )

                    # Log initial metrics
                    mlflow.log_metrics(
                        {
                            f"initial_{k}": v
                            for k, v in result["initial_metrics"].items()
                        }
                    )

                    # Log final evaluation metrics
                    mlflow.log_metrics(
                        {
                            f"eval_{k}": v
                            for k, v in result["final_eval_metrics"].items()
                        }
                    )

                    # Log detailed metrics
                    mlflow.log_metrics(result["detailed_metrics"])

                    # Extract optimization metric from evaluation metrics
                    metric_key = f"eval_{optimization_metric}"
                    if metric_key in result["final_eval_metrics"]:
                        metric_value = result["final_eval_metrics"][metric_key]
                    else:
                        # Fallback to detailed metrics
                        metric_key = optimization_metric
                        if metric_key in result["detailed_metrics"]:
                            metric_value = result["detailed_metrics"][metric_key]
                        else:
                            print(
                                f"Warning: Optimization metric '{optimization_metric}' not found. Using eval_loss."
                            )
                            metric_value = result["final_eval_metrics"].get(
                                "eval_loss", 0.0
                            )

                    # Track best combination
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_combination = {
                            "idx": idx + 1,
                            "learning_rate": lr,
                            "lora_r": r,
                            "lora_alpha_ratio": alpha_ratio,
                            "lora_alpha": int(r * alpha_ratio),
                            "lora_dropout": dropout,
                            "batch_size": batch_size,
                            "metric_value": metric_value,
                            "run_id": mlflow.active_run().info.run_id,
                        }
                        mlflow.set_tag("best_combination", "true")
                    else:
                        mlflow.set_tag("best_combination", "false")

                    all_results.append(
                        {
                            "combination_idx": idx + 1,
                            "learning_rate": lr,
                            "lora_r": r,
                            "lora_alpha": int(r * alpha_ratio),
                            "lora_dropout": dropout,
                            "batch_size": batch_size,
                            "metric_value": metric_value,
                        }
                    )

                    print(f"\n✓ Combination {idx + 1}/{total_combinations} completed")
                    print(f"  {optimization_metric}: {metric_value:.4f}")

                except Exception as e:
                    print(
                        f"\n✗ Error in combination {idx + 1}/{total_combinations}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    mlflow.log_param("error", str(e))

        # Log best combination summary
        if best_combination:
            print(f"\n{'=' * 60}")
            print("GRID SEARCH SUMMARY")
            print(f"{'=' * 60}")
            print(f"Total combinations evaluated: {total_combinations}")
            print(f"\nBest combination (based on {optimization_metric}):")
            print(f"  Combination index: {best_combination['idx']}")
            print(f"  Learning rate: {best_combination['learning_rate']}")
            print(f"  LoRA r: {best_combination['lora_r']}")
            print(f"  LoRA alpha ratio: {best_combination['lora_alpha_ratio']}")
            print(f"  LoRA alpha: {best_combination['lora_alpha']}")
            print(f"  LoRA dropout: {best_combination['lora_dropout']}")
            print(f"  Training batch size: {best_combination['batch_size']}")
            print(f"  {optimization_metric}: {best_combination['metric_value']:.4f}")
            print(f"  MLflow run ID: {best_combination['run_id']}")
            print(f"{'=' * 60}\n")

            # Log best combination to parent run
            mlflow.log_params(
                {
                    "best_learning_rate": best_combination["learning_rate"],
                    "best_lora_r": best_combination["lora_r"],
                    "best_lora_alpha_ratio": best_combination["lora_alpha_ratio"],
                    "best_lora_alpha": best_combination["lora_alpha"],
                    "best_lora_dropout": best_combination["lora_dropout"],
                    "best_batch_size": best_combination["batch_size"],
                    f"best_{optimization_metric}": best_combination["metric_value"],
                    "best_combination_idx": best_combination["idx"],
                    "best_run_id": best_combination["run_id"],
                }
            )

        print(f"\n\nGrid search completed for seed {seed}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return {
        "seed": seed,
        "best_combination": best_combination,
        "best_metric_value": best_metric_value,
        "all_results": all_results,
        "total_combinations": total_combinations,
    }


def main() -> None:
    print("Loading grid search config...")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    training_cfg_path = os.environ.get(
        "TRAINING_CONFIG_PATH",
        str(PROJECT_ROOT / "configs" / "hyperparameter_search.yaml"),
    )
    grid_search_cfg = OmegaConf.load(training_cfg_path)

    # Merge configs
    cfg = OmegaConf.merge(base_cfg, grid_search_cfg)

    # Extract dispatcher configuration
    dispatcher = getattr(cfg, "dispatcher", {})
    seeds_list = dispatcher.get("seeds")
    
    if seeds_list:
        seeds = [int(s) for s in seeds_list]
        num_runs = len(seeds)
    else:
        num_runs = int(dispatcher.get("num_runs", 1))
        # Always use random seeds if seeds not provided
        random.seed(int(time.time()))
        seeds = [random.randint(0, 2**31 - 1) for _ in range(num_runs)]

    print(f"\n{'=' * 60}")
    print(f"MULTI-RUN GRID SEARCH")
    print(f"{'=' * 60}")
    print(f"Number of runs: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"{'=' * 60}\n")

    # Setup MLflow tracking URI from config
    setup_mlflow(cfg, PROJECT_ROOT)

    # Start MLflow experiment with parent run
    experiment_name = getattr(cfg.mlflow, "experiment_name", "lora_gridsearch")
    mlflow.set_experiment(experiment_name)

    model_name: str = str(cfg.model.base)
    optimization_metric = str(cfg.grid_search.optimization_metric)
    
    all_run_results = []
    overall_best_combination = None
    overall_best_metric_value = None

    with mlflow.start_run(run_name=f"gridsearch_multi_{model_name.split('/')[-1]}"):
        # Log overall grid search parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_name": str(cfg.dataset.dataset_name),
                "optimization_metric": optimization_metric,
                "num_runs": num_runs,
                "seeds": str(seeds),
            }
        )

        # Run grid search for each seed
        for run_idx, seed in enumerate(seeds):
            print(f"\n{'#' * 60}")
            print(f"RUN {run_idx + 1}/{num_runs} - Seed: {seed}")
            print(f"{'#' * 60}\n")

            try:
                # Create a copy of config for this seed run to avoid side effects
                seed_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
                run_result = run_grid_search_with_seed(seed_cfg, seed)
                all_run_results.append(run_result)

                # Track overall best combination across all seeds
                if run_result["best_combination"]:
                    metric_value = run_result["best_metric_value"]
                    if overall_best_metric_value is None or metric_value > overall_best_metric_value:
                        overall_best_metric_value = metric_value
                        overall_best_combination = {
                            **run_result["best_combination"],
                            "seed": seed,
                            "run_idx": run_idx + 1,
                        }

                print(f"\n✓ Run {run_idx + 1}/{num_runs} (seed {seed}) completed")

            except Exception as e:
                print(f"\n✗ Error in run {run_idx + 1}/{num_runs} (seed {seed}): {e}")
                import traceback
                traceback.print_exc()
                mlflow.log_param(f"error_run_{run_idx + 1}", str(e))

        # Log overall summary
        if overall_best_combination:
            print(f"\n{'=' * 60}")
            print("OVERALL GRID SEARCH SUMMARY (ACROSS ALL SEEDS)")
            print(f"{'=' * 60}")
            print(f"Total runs: {num_runs}")
            print(f"Seeds used: {seeds}")
            print(f"\nBest combination overall (based on {optimization_metric}):")
            print(f"  Run index: {overall_best_combination['run_idx']}")
            print(f"  Seed: {overall_best_combination['seed']}")
            print(f"  Combination index: {overall_best_combination['idx']}")
            print(f"  Learning rate: {overall_best_combination['learning_rate']}")
            print(f"  LoRA r: {overall_best_combination['lora_r']}")
            print(f"  LoRA alpha ratio: {overall_best_combination['lora_alpha_ratio']}")
            print(f"  LoRA alpha: {overall_best_combination['lora_alpha']}")
            print(f"  LoRA dropout: {overall_best_combination['lora_dropout']}")
            print(f"  Training batch size: {overall_best_combination['batch_size']}")
            print(f"  {optimization_metric}: {overall_best_combination['metric_value']:.4f}")
            print(f"  MLflow run ID: {overall_best_combination['run_id']}")
            print(f"{'=' * 60}\n")

            # Log overall best combination to parent run
            mlflow.log_params(
                {
                    "overall_best_seed": overall_best_combination["seed"],
                    "overall_best_run_idx": overall_best_combination["run_idx"],
                    "overall_best_learning_rate": overall_best_combination["learning_rate"],
                    "overall_best_lora_r": overall_best_combination["lora_r"],
                    "overall_best_lora_alpha_ratio": overall_best_combination["lora_alpha_ratio"],
                    "overall_best_lora_alpha": overall_best_combination["lora_alpha"],
                    "overall_best_lora_dropout": overall_best_combination["lora_dropout"],
                    "overall_best_batch_size": overall_best_combination["batch_size"],
                    f"overall_best_{optimization_metric}": overall_best_combination["metric_value"],
                    "overall_best_combination_idx": overall_best_combination["idx"],
                    "overall_best_run_id": overall_best_combination["run_id"],
                }
            )

            # Log best results per seed
            for run_result in all_run_results:
                if run_result["best_combination"]:
                    seed = run_result["seed"]
                    mlflow.log_params(
                        {
                            f"seed_{seed}_best_{optimization_metric}": run_result["best_metric_value"],
                            f"seed_{seed}_best_learning_rate": run_result["best_combination"]["learning_rate"],
                            f"seed_{seed}_best_lora_r": run_result["best_combination"]["lora_r"],
                        }
                    )

        print("\n\nMulti-run grid search completed")
        print(f"Parent MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting grid search...")
    main()
