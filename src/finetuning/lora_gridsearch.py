# %%
import os
from pathlib import Path
import sys
import warnings
import mlflow
import pandas as pd
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


def run_single_training(cfg, cache_dir, output_dir_base, combination_idx, total_combinations):
    """Run a single training run with given config."""
    model_name: str = str(cfg.model.base)
    
    # Create unique output directory for this combination
    output_dir = os.path.join(output_dir_base, f"combination_{combination_idx}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if using split files or single file
    use_split_files = bool(getattr(cfg.dataset, "use_split_files", False))
    
    if use_split_files:
        dataset_base_path = PROJECT_ROOT / "data" / cfg.dataset.dataset_name
        train_csv = str(
            dataset_base_path / getattr(cfg.dataset, "train_file", "train.csv")
        )
        val_csv = str(dataset_base_path / getattr(cfg.dataset, "val_file", "val.csv"))
        test_csv = str(
            dataset_base_path / getattr(cfg.dataset, "test_file", "test.csv")
        )
        dataset_csv = train_csv  # For logging purposes
    else:
        dataset_csv = str(PROJECT_ROOT / cfg.dataset.csv_path)
        train_csv = None
        val_csv = None
        test_csv = None
    
    # Extract topics from config
    topics = getattr(cfg.dataset, "topics", None)
    
    # Load and preprocess data
    raw_data, label_order, label2id, id2label = load_and_preprocess_data(
        dataset_csv,
        cache_dir,
        int(getattr(cfg.project, "seed", 42)),
        test_size=getattr(cfg.dataset, "test_size", 0.4),
        topics=topics,
        use_split_files=use_split_files,
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
    include_ref_ans = bool(
        getattr(cfg.tokenization, "include_reference_answer", False)
    )
    tokenized_data = tokenize_dataset(
        raw_data, tokenizer, include_ref_ans
    )
    
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
    best_metrics = trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None
    
    # Evaluate on validation set to get final metrics
    final_eval_metrics = trainer.evaluate()
    
    # Perform detailed evaluation on test set
    print("\nPerforming detailed evaluation on test dataset...")
    detailed_metrics = detailed_evaluation(
        trainer, tokenized_data["test"], label_order
    )
    
    return {
        "initial_metrics": initial_metrics,
        "final_eval_metrics": final_eval_metrics,
        "detailed_metrics": detailed_metrics,
        "model": model,
        "tokenizer": tokenizer,
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
    
    # Extract grid search parameters
    grid_search = cfg.grid_search
    learning_rates = list(grid_search.learning_rate)
    lora_rs = list(grid_search.lora_r)
    lora_alpha_ratios = list(grid_search.lora_alpha_ratios)
    lora_dropouts = list(grid_search.lora_dropout)
    gradient_accumulation_steps_list = list(grid_search.gradient_accumulation_steps)
    optimization_metric = str(grid_search.optimization_metric)
    seed = int(grid_search.seed)
    
    # Set seed in config
    cfg.project.seed = seed
    
    # Generate all combinations
    combinations = list(product(
        learning_rates,
        lora_rs,
        lora_alpha_ratios,
        lora_dropouts,
        gradient_accumulation_steps_list
    ))
    
    total_combinations = len(combinations)
    print(f"\n{'='*60}")
    print(f"Grid Search: {total_combinations} combinations to evaluate")
    print(f"{'='*60}")
    print(f"Learning rates: {learning_rates}")
    print(f"LoRA r: {lora_rs}")
    print(f"LoRA alpha ratios: {lora_alpha_ratios}")
    print(f"LoRA dropout: {lora_dropouts}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps_list}")
    print(f"Optimization metric: {optimization_metric}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    model_name: str = str(cfg.model.base)
    output_dir_base: str = str(PROJECT_ROOT / cfg.output.dir)
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
    
    # Setup MLflow tracking URI from config
    setup_mlflow(cfg, PROJECT_ROOT)
    
    # Start MLflow experiment with parent run
    experiment_name = getattr(cfg.mlflow, "experiment_name", "lora_gridsearch")
    mlflow.set_experiment(experiment_name)
    
    best_combination = None
    best_metric_value = None
    all_results = []
    
    with mlflow.start_run(run_name=f"gridsearch_{model_name.split('/')[-1]}"):
        # Log grid search parameters
        mlflow.log_params({
            "model_name": model_name,
            "dataset_name": str(cfg.dataset.dataset_name),
            "optimization_metric": optimization_metric,
            "seed": seed,
            "total_combinations": total_combinations,
            "learning_rates": str(learning_rates),
            "lora_rs": str(lora_rs),
            "lora_alpha_ratios": str(lora_alpha_ratios),
            "lora_dropouts": str(lora_dropouts),
            "gradient_accumulation_steps": str(gradient_accumulation_steps_list),
        })
        
        # Run each combination
        for idx, (lr, r, alpha_ratio, dropout, grad_acc_steps) in enumerate(combinations):
            print(f"\n{'='*60}")
            print(f"Combination {idx + 1}/{total_combinations}")
            print(f"{'='*60}")
            print(f"Learning rate: {lr}")
            print(f"LoRA r: {r}")
            print(f"LoRA alpha ratio: {alpha_ratio}")
            print(f"LoRA alpha: {r * alpha_ratio}")
            print(f"LoRA dropout: {dropout}")
            print(f"Gradient accumulation steps: {grad_acc_steps}")
            print(f"Effective batch size: {8 * grad_acc_steps}")
            print(f"{'='*60}\n")
            
            # Create modified config for this combination
            combination_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
            combination_cfg.training.learning_rate = float(lr)
            combination_cfg.lora.r = int(r)
            combination_cfg.lora.alpha = int(r * alpha_ratio)
            combination_cfg.lora.dropout = float(dropout)
            combination_cfg.training.gradient_accumulation_steps = int(grad_acc_steps)
            
            # Create nested MLflow run for this combination
            run_name = f"lr_{lr}_r_{r}_alpha_{r * alpha_ratio:.1f}_drop_{dropout}_gradacc_{grad_acc_steps}"
            with mlflow.start_run(run_name=run_name, nested=True):
                try:
                    # Log hyperparameters
                    mlflow.log_params({
                        "learning_rate": lr,
                        "lora_r": r,
                        "lora_alpha_ratio": alpha_ratio,
                        "lora_alpha": int(r * alpha_ratio),
                        "lora_dropout": dropout,
                        "gradient_accumulation_steps": grad_acc_steps,
                        "effective_batch_size": 8 * grad_acc_steps,
                        "combination_idx": idx + 1,
                        "seed": seed,
                    })
                    
                    # Run training
                    result = run_single_training(
                        combination_cfg, cache_path if cache_path else None, output_dir_base, idx, total_combinations
                    )
                    
                    # Log initial metrics
                    mlflow.log_metrics({f"initial_{k}": v for k, v in result["initial_metrics"].items()})
                    
                    # Log final evaluation metrics
                    mlflow.log_metrics({f"eval_{k}": v for k, v in result["final_eval_metrics"].items()})
                    
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
                            print(f"Warning: Optimization metric '{optimization_metric}' not found. Using eval_loss.")
                            metric_value = result["final_eval_metrics"].get("eval_loss", 0.0)
                    
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
                            "gradient_accumulation_steps": grad_acc_steps,
                            "effective_batch_size": 8 * grad_acc_steps,
                            "metric_value": metric_value,
                            "run_id": mlflow.active_run().info.run_id,
                        }
                        mlflow.set_tag("best_combination", "true")
                    else:
                        mlflow.set_tag("best_combination", "false")
                    
                    all_results.append({
                        "combination_idx": idx + 1,
                        "learning_rate": lr,
                        "lora_r": r,
                        "lora_alpha": int(r * alpha_ratio),
                        "lora_dropout": dropout,
                        "gradient_accumulation_steps": grad_acc_steps,
                        "metric_value": metric_value,
                    })
                    
                    print(f"\n✓ Combination {idx + 1}/{total_combinations} completed")
                    print(f"  {optimization_metric}: {metric_value:.4f}")
                    
                except Exception as e:
                    print(f"\n✗ Error in combination {idx + 1}/{total_combinations}: {e}")
                    import traceback
                    traceback.print_exc()
                    mlflow.log_param("error", str(e))
        
        # Log best combination summary
        if best_combination:
            print(f"\n{'='*60}")
            print("GRID SEARCH SUMMARY")
            print(f"{'='*60}")
            print(f"Total combinations evaluated: {total_combinations}")
            print(f"\nBest combination (based on {optimization_metric}):")
            print(f"  Combination index: {best_combination['idx']}")
            print(f"  Learning rate: {best_combination['learning_rate']}")
            print(f"  LoRA r: {best_combination['lora_r']}")
            print(f"  LoRA alpha ratio: {best_combination['lora_alpha_ratio']}")
            print(f"  LoRA alpha: {best_combination['lora_alpha']}")
            print(f"  LoRA dropout: {best_combination['lora_dropout']}")
            print(f"  Gradient accumulation steps: {best_combination['gradient_accumulation_steps']}")
            print(f"  Effective batch size: {best_combination['effective_batch_size']}")
            print(f"  {optimization_metric}: {best_combination['metric_value']:.4f}")
            print(f"  MLflow run ID: {best_combination['run_id']}")
            print(f"{'='*60}\n")
            
            # Log best combination to parent run
            mlflow.log_params({
                "best_learning_rate": best_combination['learning_rate'],
                "best_lora_r": best_combination['lora_r'],
                "best_lora_alpha_ratio": best_combination['lora_alpha_ratio'],
                "best_lora_alpha": best_combination['lora_alpha'],
                "best_lora_dropout": best_combination['lora_dropout'],
                "best_gradient_accumulation_steps": best_combination['gradient_accumulation_steps'],
                "best_effective_batch_size": best_combination['effective_batch_size'],
                f"best_{optimization_metric}": best_combination['metric_value'],
                "best_combination_idx": best_combination['idx'],
                "best_run_id": best_combination['run_id'],
            })
        
        print("\n\nGrid search completed")
        print(f"Parent MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting grid search...")
    main()

