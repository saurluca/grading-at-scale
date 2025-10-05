import os
from pathlib import Path

import mlflow
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf

from common import (
    setup_training_args,
    setup_trainer,
    load_and_preprocess_data,
    tokenize_dataset,
    detailed_evaluation,
    setup_model_and_tokenizer,
)


def setup_lora_model(base_model, cfg):
    print("Setting up LoRA configuration and applying to base model...")
    lora_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.lora_alpha),
        lora_dropout=float(cfg.lora.lora_dropout),
        # target_modules=list(cfg.lora.target_modules),
        target_modules="all-linear",
        task_type=TaskType.SEQ_CLS,
        init_lora_weights=str(cfg.lora.init_lora_weights),
    )
    return get_peft_model(base_model, lora_cfg)


def main() -> None:
    print("Loading config...")
    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "peft_lora.yaml"
    )

    dataset_csv: str = str(cfg.dataset_csv)
    model_name: str = str(cfg.model_name)
    output_dir: str = str(cfg.output_dir)
    cache_dir: str | None = str(cfg.hf_cache_dir) if "hf_cache_dir" in cfg else None

    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Start MLflow experiment
    experiment_name = "peft_lora_training"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"lora_training_{model_name.split('/')[-1]}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_csv": dataset_csv,
                "output_dir": output_dir,
                "lora_r": int(cfg.lora.r),
                "lora_alpha": int(cfg.lora.lora_alpha),
                "lora_dropout": float(cfg.lora.lora_dropout),
                "target_modules": str(list(cfg.lora.target_modules)),
                "num_train_epochs": float(cfg.training.num_train_epochs),
                "per_device_train_batch_size": int(
                    cfg.training.per_device_train_batch_size
                ),
                "per_device_eval_batch_size": int(
                    cfg.training.per_device_eval_batch_size
                ),
                "learning_rate": float(cfg.training.learning_rate),
                "weight_decay": float(cfg.training.weight_decay),
                "eval_strategy": str(cfg.training.eval_strategy),
                "seed": int(getattr(cfg, "seed", 42)),
                "use_unseen_questions": bool(
                    getattr(cfg.training, "use_unseen_questions", False)
                ),
                "save_model": bool(getattr(cfg, "save_model", True)),
            }
        )

        # Load and preprocess data
        raw_data, label_order, label2id, id2label = load_and_preprocess_data(
            dataset_csv,
            cache_dir,
            int(getattr(cfg, "seed", 42)),
            test_size=cfg.training.test_size,
            use_unseen_questions=bool(
                getattr(cfg.training, "use_unseen_questions", False)
            ),
        )

        # Log dataset info
        mlflow.log_params(
            {
                "train_size": len(raw_data["train"]),
                "test_size": len(raw_data["test"]),
                "total_size": len(raw_data["train"]) + len(raw_data["test"]),
            }
        )

        # Setup model and tokenizer
        tokenizer, base_model = setup_model_and_tokenizer(
            model_name, label2id, id2label, cache_dir
        )

        # Setup LoRA model
        model = setup_lora_model(base_model, cfg)

        model.print_trainable_parameters()

        # dump the raw train and test datasets 1
        raw_data["train"].to_csv(f"{output_dir}/train.csv", index=False, sep=";")
        raw_data["test"].to_csv(f"{output_dir}/test.csv", index=False, sep=";")

        # Tokenize dataset
        tokenized_data = tokenize_dataset(raw_data, tokenizer)

        # Setup training arguments and trainer
        training_args = setup_training_args(cfg, output_dir)
        trainer, loss_callback = setup_trainer(
            model, training_args, tokenized_data, tokenizer
        )

        # Training
        print("Starting training...")
        trainer.train()
        metrics = trainer.evaluate()

        # Log final loss information
        if loss_callback.train_losses:
            print(f"\nFinal Training Loss: {loss_callback.train_losses[-1]:.4f}")
            mlflow.log_metric("final_train_loss", loss_callback.train_losses[-1])

        if loss_callback.eval_losses:
            print(f"Final Evaluation Loss: {loss_callback.eval_losses[-1]:.4f}")
            mlflow.log_metric("final_eval_loss", loss_callback.eval_losses[-1])

        # Log final metrics
        mlflow.log_metrics(metrics)

        # Perform detailed evaluation
        print("\nPerforming detailed evaluation on test dataset...")
        detailed_metrics = detailed_evaluation(
            trainer, tokenized_data["test"], label_order
        )

        # Log detailed evaluation metrics to MLflow
        mlflow.log_metrics(detailed_metrics)

        # Save adapter and tokenizer
        model.save_pretrained(Path(output_dir) / f"adapter-{model_name}")
        tokenizer.save_pretrained(Path(output_dir) / f"tokenizer-{model_name}")

        # Log model artifacts
        mlflow.log_artifacts(
            Path(output_dir) / f"model_outputs-{model_name}", "model_outputs"
        )
        mlflow.log_artifacts(Path(output_dir) / f"tokenizer-{model_name}", "tokenizer")
        mlflow.log_artifacts(Path(output_dir) / f"adapter-{model_name}", "adapter")

        # Log the trained model using MLflow transformers integration
        save_model = bool(getattr(cfg, "save_model", True))
        if save_model:
            try:
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": model,
                        "tokenizer": tokenizer,
                    },
                    name="model",
                    task="text-classification",
                )
                print("Model logged to MLflow successfully")
            except Exception as e:
                print(f"Warning: Could not log model to MLflow: {e}")
        else:
            print("Model saving to MLflow skipped (save_model=false in config)")

        print("Training complete. Eval metrics:", metrics)
        print(f"Adapter saved to: {Path(output_dir) / f'adapter-{model_name}'}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting...")
    main()
