# %%
import os
from pathlib import Path
import sys
import mlflow
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


def setup_lora_model(base_model, cfg):
    print("Setting up LoRA configuration and applying to base model...")
    lora_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=cfg.lora.target_modules,
        # target_modules="all-linear",
        task_type=TaskType.SEQ_CLS,
        init_lora_weights=str(cfg.lora.init_weights),
    )
    return get_peft_model(base_model, lora_cfg)


def main() -> None:
    print("Loading config...")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    training_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "training.yaml")
    cfg = OmegaConf.merge(base_cfg, training_cfg)

    dataset_csv: str = str(PROJECT_ROOT / cfg.dataset.csv_path)
    print(f"dataset_csv: {dataset_csv}")
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

    # Start MLflow experiment
    experiment_name = getattr(cfg.mlflow, "experiment_name", "peft_lora_training")
    
    # Add quantization info to experiment name if enabled
    if "quantization" in cfg and cfg.quantization.get("load_in_4bit", False):
        experiment_name = experiment_name.replace("peft_lora_training", "peft_lora_training_4bit")
    
    mlflow.set_experiment(experiment_name)

    run_name_suffix = "qlora" if ("quantization" in cfg and cfg.quantization.get("load_in_4bit", False)) else "lora"
    with mlflow.start_run(run_name=f"{run_name_suffix}_training_{model_name.split('/')[-1]}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_csv": dataset_csv,
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
                "gradient_accumulation_steps": int(getattr(cfg.training, "gradient_accumulation_steps", 1)),
                "eval_strategy": str(cfg.training.eval_strategy),
                "seed": int(getattr(cfg.project, "seed", 42)),
                "use_unseen_questions": bool(
                    getattr(cfg.dataset, "use_unseen_questions", False)
                ),
                "save_model": bool(getattr(cfg.output, "save_model", True)),
            }
        )

        # Load and preprocess data
        raw_data, label_order, label2id, id2label = load_and_preprocess_data(
            dataset_csv,
            cache_dir,
            int(getattr(cfg.project, "seed", 42)),
            test_size=cfg.dataset.test_size,
            use_unseen_questions=bool(
                getattr(cfg.dataset, "use_unseen_questions", False)
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

        # Setup model and tokenizer (with optional quantization)
        quantization_config = None
        if "quantization" in cfg and cfg.quantization.get("load_in_4bit", False):
            print("Quantization enabled in config")
            quantization_config = {
                "load_in_4bit": cfg.quantization.load_in_4bit,
                "bnb_4bit_compute_dtype": cfg.quantization.get("bnb_4bit_compute_dtype", "float16"),
                "bnb_4bit_quant_type": cfg.quantization.get("bnb_4bit_quant_type", "nf4"),
                "bnb_4bit_use_double_quant": cfg.quantization.get("bnb_4bit_use_double_quant", True),
            }
            mlflow.log_params({
                "quant_bits": 4,
                "quant_type": quantization_config["bnb_4bit_quant_type"],
                "quant_enabled": True,
            })
        else:
            print("Quantization disabled or not configured")
            mlflow.log_param("quant_enabled", False)
        
        tokenizer, base_model = setup_model_and_tokenizer(
            model_name, label2id, id2label, cache_path, quantization_config
        )

        # Setup LoRA model
        model = setup_lora_model(base_model, cfg)

        model.print_trainable_parameters()

        # dump the raw train and test datasets 1
        raw_data["train"].to_csv(f"{output_dir}/train.csv", index=False, sep=";")
        raw_data["test"].to_csv(f"{output_dir}/test.csv", index=False, sep=";")

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

        # Save adapter locally
        adapter_path = (
            Path(output_dir)
            / f"adapter-{model_name.split('/')[-1]}-{cfg.dataset.dataset_name}"
        )
        model.save_pretrained(adapter_path)

        # Push adapter to Hugging Face Hub
        save_model = bool(getattr(cfg.output, "save_model", True))

        if save_model:
            print("\n Trying to save model to huggingface")

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
        # Log model artifacts (adapter only)
        mlflow.log_artifacts(
            Path(output_dir) / f"model_outputs-{model_name}", "model_outputs"
        )
        mlflow.log_artifacts(str(adapter_path), "adapter")

        # Log the LoRA adapter using MLflow transformers integration
        save_model = bool(getattr(cfg.output, "save_model", True))
        print("")
        if save_model:
            try:
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": model,
                        "tokenizer": tokenizer,
                    },
                    name="lora_adapter",
                    task="text-classification",
                )
                print("LoRA adapter logged to MLflow successfully")
            except Exception as e:
                print(f"Warning: Could not log LoRA adapter to MLflow: {e}")
        else:
            print("LoRA adapter saving to MLflow skipped (save_model=false in config)")

        print("\n\nTraining complete. Eval metrics:", metrics)
        print(f"Adapter saved to: {adapter_path}")
        if save_model:
            print(f"Adapter pushed to HF Hub: {repo_name}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("Starting...")
    main()
