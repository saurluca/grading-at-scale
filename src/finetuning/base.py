import os
from pathlib import Path
import sys

import mlflow
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Force the process to see only a single GPU (GPU 0) to avoid DataParallel
# Must be set before importing any modules that might initialize CUDA/torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.common import (  # noqa: E402
    load_and_preprocess_data,
    setup_model_and_tokenizer,
    setup_training_args,
    setup_trainer,
    tokenize_dataset,
    detailed_evaluation,
)
from src.mlflow_config import setup_mlflow  # noqa: E402


def main() -> None:
    print("Loading config...")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    training_cfg_path = os.environ.get(
        "TRAINING_CONFIG_PATH",
        str(PROJECT_ROOT / "configs" / "training.yaml"),
    )
    training_cfg = OmegaConf.load(training_cfg_path)
    cfg = OmegaConf.merge(base_cfg, training_cfg)

    # Determine if using split files or single file
    use_split_files = bool(getattr(cfg.dataset, "use_split_files", False))

    if use_split_files:
        # Use separate train/val/test files
        dataset_base_path = PROJECT_ROOT / "data" / cfg.dataset.dataset_name
        train_csv = str(
            dataset_base_path / getattr(cfg.dataset, "train_file", "train.csv")
        )
        val_csv = str(dataset_base_path / getattr(cfg.dataset, "val_file", "val.csv"))
        test_csv = str(
            dataset_base_path / getattr(cfg.dataset, "test_file", "test.csv")
        )
        dataset_csv = train_csv  # For logging purposes
        print(f"Using split files - train: {train_csv}, val: {val_csv}, test: {test_csv}")
    else:
        # Use single file to split at runtime
        dataset_csv = str(PROJECT_ROOT / cfg.dataset.csv_path)
        train_csv = None
        val_csv = None
        test_csv = None
        print(f"Using single file to split at runtime: {dataset_csv}")

    model_name: str = str(cfg.model.base)
    output_dir: str = str(PROJECT_ROOT / cfg.output.dir)
    cache_dir: str | None = str(cfg.paths.hf_cache_dir) if "paths" in cfg else None

    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        # Ensure cache directory is at project root
        cache_path = (
            os.path.join(PROJECT_ROOT, cache_dir)
            if not os.path.isabs(cache_dir)
            else cache_dir
        )
        os.makedirs(cache_path, exist_ok=True)

    # Setup MLflow tracking URI from config
    setup_mlflow(cfg, PROJECT_ROOT)

    # Start MLflow experiment
    experiment_name = "vanilla_training"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"vanilla_training_{model_name.split('/')[-1]}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_name": str(cfg.dataset.dataset_name),
                "dataset_csv": dataset_csv,
                "use_split_files": use_split_files,
                "output_dir": output_dir,
                "num_train_epochs": float(cfg.training.num_epochs),
                "per_device_train_batch_size": int(cfg.training.batch_size.train),
                "per_device_eval_batch_size": int(cfg.training.batch_size.eval),
                "learning_rate": float(cfg.training.learning_rate),
                "weight_decay": float(cfg.training.weight_decay),
                "seed": int(getattr(cfg.project, "seed", 42)),
                "include_reference_answer": bool(
                    getattr(cfg.tokenization, "include_reference_answer", False)
                ),
            }
        )

        # Load and preprocess data
        raw_data, label_order, label2id, id2label = load_and_preprocess_data(
            dataset_csv,
            cache_dir,
            int(getattr(cfg.project, "seed", 42)),
            test_size=cfg.dataset.test_size,
            topics=getattr(cfg.dataset, "topics", None),
            use_split_files=use_split_files,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
        )

        # Log dataset info
        dataset_info = {
            "train_size": len(raw_data["train"]),
            "test_size": len(raw_data["test"]),
            "total_size": len(raw_data["train"]) + len(raw_data["test"]),
        }
        if "val" in raw_data:
            dataset_info["val_size"] = len(raw_data["val"])
            dataset_info["total_size"] = len(raw_data["train"]) + len(raw_data["val"]) + len(raw_data["test"])
        mlflow.log_params(dataset_info)

        # Setup model and tokenizer
        tokenizer, model = setup_model_and_tokenizer(
            model_name, label2id, id2label, cache_dir
        )

        # Tokenize dataset
        include_ref_ans = bool(
            getattr(cfg.tokenization, "include_reference_answer", False)
        )
        include_chunk = bool(getattr(cfg.tokenization, "include_chunk_text", False))
        tokenized_data = tokenize_dataset(
            raw_data, tokenizer, include_ref_ans, include_chunk
        )

        # Setup training arguments and trainer
        training_args = setup_training_args(cfg, output_dir)
        
        # Log eval_strategy and eval_steps after setup (since they may be overridden)
        eval_strategy_to_log = training_args.eval_strategy
        mlflow.log_param("eval_strategy", eval_strategy_to_log)
        if eval_strategy_to_log == "steps":
            mlflow.log_param("eval_steps", training_args.eval_steps)
        
        trainer = setup_trainer(model, training_args, tokenized_data, tokenizer, cfg)

        # Training
        print("Starting training...")
        # Evaluate on validation set before training (if available, otherwise test set)
        eval_set_name = "validation" if "val" in tokenized_data else "test"
        print(f"Evaluating on {eval_set_name} set before training...")
        metrics = trainer.evaluate()
        print(f"Metrics before training: {metrics}")
        mlflow.log_metrics({f"initial_{k}": v for k, v in metrics.items()})

        trainer.train()

        # The best model is automatically loaded at the end of training (load_best_model_at_end=True)
        # Perform detailed evaluation on test set
        print("\nPerforming detailed evaluation on test dataset...")
        detailed_metrics = detailed_evaluation(
            trainer, tokenized_data["test"], label_order
        )

        # Log detailed evaluation metrics to MLflow
        mlflow.log_metrics(detailed_metrics)

        # Optionally save the fine-tuned model and tokenizer
        model.save_pretrained(Path(output_dir) / f"model-{model_name}")
        tokenizer.save_pretrained(output_dir)

        # Log model artifacts
        mlflow.log_artifacts(Path(output_dir) / f"model-{model_name}", "model")
        mlflow.log_artifacts(output_dir, "tokenizer")

        # Log the full training configuration as an artifact
        mlflow.log_artifact(PROJECT_ROOT / "configs" / "training.yaml", "config")


if __name__ == "__main__":
    print("Starting...")
    main()
