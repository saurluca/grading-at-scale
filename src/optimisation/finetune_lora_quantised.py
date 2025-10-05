import os
from pathlib import Path

import mlflow
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    ApertusForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from omegaconf import OmegaConf
from common import (
    setup_training_args,
    setup_trainer,
    load_and_preprocess_data,
    tokenize_dataset,
    detailed_evaluation,
)


def setup_quantized_lora_model(
    model_name: str, num_labels: int, label2id, id2label, cache_dir: str | None, cfg
):
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    if model_name == "swiss-ai/Apertus-8B-Instruct-2509":
        base_model = ApertusForCausalLM.from_pretrained(model_name)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            quantization_config=quant_config,
            device_map="auto",
            cache_dir=cache_dir,
        )

    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.lora_alpha),
        lora_dropout=float(cfg.lora.lora_dropout),
        target_modules=list(cfg.lora.target_modules),
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_cfg)
    return model


def main() -> None:
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

    experiment_name = "peft_lora_training_4bit"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"qlora_training_{model_name.split('/')[-1]}"):
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
                "quant_bits": 4,
                "quant_type": "nf4",
            }
        )

        raw_data, label_order, label2id, id2label = load_and_preprocess_data(
            dataset_csv,
            cache_dir,
            int(getattr(cfg, "seed", 42)),
            test_size=cfg.training.test_size,
        )

        mlflow.log_params(
            {
                "train_size": len(raw_data["train"]),
                "test_size": len(raw_data["test"]),
                "total_size": len(raw_data["train"]) + len(raw_data["test"]),
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model = setup_quantized_lora_model(
            model_name=model_name,
            num_labels=3,
            label2id=label2id,
            id2label=id2label,
            cache_dir=cache_dir,
            cfg=cfg,
        )

        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        model.print_trainable_parameters()

        tokenized_data = tokenize_dataset(raw_data, tokenizer)

        training_args = setup_training_args(cfg, output_dir)
        trainer, loss_callback = setup_trainer(
            model, training_args, tokenized_data, tokenizer
        )

        trainer.train()
        metrics = trainer.evaluate()

        if loss_callback.train_losses:
            mlflow.log_metric("final_train_loss", loss_callback.train_losses[-1])
        if loss_callback.eval_losses:
            mlflow.log_metric("final_eval_loss", loss_callback.eval_losses[-1])

        mlflow.log_metrics(metrics)

        detailed_metrics = detailed_evaluation(
            trainer, tokenized_data["test"], label_order
        )
        mlflow.log_metrics(detailed_metrics)


if __name__ == "__main__":
    main()
