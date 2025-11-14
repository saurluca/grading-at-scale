# %%
import os
from pathlib import Path
import sys
import warnings
import mlflow
import pandas as pd
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


def main() -> None:
    print("Loading config...")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    training_cfg_path = os.environ.get(
        "TRAINING_CONFIG_PATH",
        str(PROJECT_ROOT / "configs" / "training.yaml"),
    )
    training_cfg = OmegaConf.load(training_cfg_path)
    cfg = OmegaConf.merge(base_cfg, training_cfg)

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
    print(f"Loading datasets - train: {train_csv}, val: {val_csv}, test: {test_csv}")

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
    else:
        cache_path = None

    # Setup MLflow tracking URI from config
    setup_mlflow(cfg, PROJECT_ROOT)

    # Start MLflow experiment
    experiment_name = getattr(cfg.mlflow, "experiment_name", "peft_lora_training")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"lora_{model_name.split('/')[-1]}"):
        # Log the raw dataset as an MLflow Dataset
        try:
            raw_df = pd.read_csv(dataset_csv, delimiter=";")
            # Ensure labels column is properly typed to avoid MLflow schema warnings
            if "labels" in raw_df.columns:
                raw_df["labels"] = raw_df["labels"].astype("int64")
            ds_name = str(getattr(cfg.dataset, "dataset_name", Path(dataset_csv).stem))
            # Suppress MLflow dataset source resolution warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="mlflow")
                ml_dataset = mlflow.data.from_pandas(
                    raw_df,
                    source=dataset_csv,
                    name=ds_name,
                )
                mlflow.log_input(ml_dataset, context="training")
        except Exception as e:
            print(f"Warning: Failed to log MLflow Dataset input: {e}")

        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_name": str(cfg.dataset.dataset_name),
                "train_csv": train_csv,
                "val_csv": val_csv,
                "test_csv": test_csv,
                "output_dir": output_dir,
                "lora_r": int(cfg.lora.r),
                "lora_alpha": int(cfg.lora.alpha),
                "lora_dropout": float(cfg.lora.dropout),
                "target_modules": str(list(cfg.lora.target_modules)),
                "num_train_epochs": float(cfg.training.num_epochs),
                "per_device_train_batch_size": int(cfg.training.batch_size.train),
                "per_device_eval_batch_size": int(cfg.training.batch_size.eval),
                "learning_rate": float(cfg.training.learning_rate),
                "weight_decay": float(cfg.training.weight_decay),
                "gradient_accumulation_steps": int(
                    getattr(cfg.training, "gradient_accumulation_steps", 1)
                ),
                "seed": int(getattr(cfg.project, "seed", 42)),
                "save_model": bool(getattr(cfg.output, "save_model", True)),
                "include_reference_answer": bool(
                    getattr(cfg.tokenization, "include_reference_answer", False)
                ),
            }
        )

        # Load and preprocess data
        raw_data, label_order, label2id, id2label = load_and_preprocess_data(
            cache_dir,
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
        tokenizer, base_model = setup_model_and_tokenizer(
            model_name, label2id, id2label, cache_path
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

        # dump the raw train, val, and test datasets
        raw_data["train"].to_csv(f"{output_dir}/train.csv", index=False, sep=";")
        if "val" in raw_data:
            raw_data["val"].to_csv(f"{output_dir}/val.csv", index=False, sep=";")
        raw_data["test"].to_csv(f"{output_dir}/test.csv", index=False, sep=";")

        # Tokenize dataset
        include_ref_ans = bool(
            getattr(cfg.tokenization, "include_reference_answer", False)
        )
        tokenized_data = tokenize_dataset(
            raw_data, tokenizer, include_ref_ans
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

        # Log the full training configuration as an artifact
        mlflow.log_artifact(PROJECT_ROOT / "configs" / "training.yaml", "config")

        # Log the LoRA adapter using MLflow transformers integration
        if cfg.output.save_model_locally:
            print("\nSaving LoRA adapter to local path")
            adapter_path = (
                Path(output_dir)
                / f"adapter-{model_name.split('/')[-1]}-{cfg.dataset.dataset_name}"
            )
            model.save_pretrained(adapter_path)

            mlflow.log_artifacts(
                Path(output_dir)
                / f"model_outputs-{model_name}-{cfg.dataset.dataset_name}",
                "model_outputs",
            )
            mlflow.log_artifacts(str(adapter_path), "adapter")

            try:
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": model,
                        "tokenizer": tokenizer,
                    },
                    name="lora_adapter",
                    task="text-classification",
                )
                print(f"Adapter saved to: {adapter_path}")
            except Exception as e:
                print(f"Warning: Could not log LoRA adapter to MLflow: {e}")
        else:
            print("LoRA adapter saving to MLflow skipped (save_model=false in config)")

        # Push adapter to Hugging Face Hub
        if cfg.output.push_to_hub:
            print("\n Saving model to huggingface")
            try:
                repo_name = (
                    f"{model_name.split('/')[-1]}-lora-{cfg.dataset.dataset_name}"
                )
                model.push_to_hub(repo_name)
                print(f"Adapter successfully pushed to Hugging Face Hub: {repo_name}")
                mlflow.log_param("hf_hub_repo", repo_name)
            except Exception as e:
                print(f"Warning: Could not push adapter to Hugging Face Hub: {e}")
                mlflow.log_param("hf_hub_error", str(e))

        print("\n\nTraining completed")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting...")
    main()
