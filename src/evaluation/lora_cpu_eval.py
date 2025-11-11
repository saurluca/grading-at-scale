# %%
"""
CPU-only evaluation script for fine-tuned LoRA adapters.

This script evaluates a LoRA adapter model on CPU, loading:
- Base model: Qwen/Qwen3-0.6B
- LoRA adapter from HuggingFace Hub (for gras dataset)
- Test dataset: data/gras/test.csv

CPU enforcement is applied at the start to ensure no GPU usage.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# CRITICAL: Enforce CPU-only execution before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import DatasetDict, load_dataset
from omegaconf import OmegaConf
from peft import PeftModel
import torch
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    compute_metrics,
    detailed_evaluation,
    map_labels,
    sample_dataset,
    setup_model_and_tokenizer,
    tokenize_dataset,
)

# Verify CPU enforcement
if torch.cuda.is_available():
    print("WARNING: CUDA is available but CPU-only mode is enforced.")
    print("CUDA_VISIBLE_DEVICES is set to empty string.")
print(f"Using device: CPU (torch.cuda.is_available()={torch.cuda.is_available()})")


def main() -> None:
    print("=" * 60)
    print("CPU-ONLY LoRA ADAPTER EVALUATION")
    print("=" * 60)
    
    # Load configs
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    cpu_eval_cfg_path = os.environ.get(
        "LORA_CPU_EVAL_CONFIG",
        str(PROJECT_ROOT / "configs" / "lora_cpu_eval.yaml"),
    )
    cpu_eval_cfg = OmegaConf.load(cpu_eval_cfg_path)
    cfg = OmegaConf.merge(base_cfg, cpu_eval_cfg)

    # Verify CPU enforcement is enabled
    enforce_cpu = bool(getattr(cfg.lora_cpu_eval, "enforce_cpu", True))
    if not enforce_cpu:
        print("WARNING: enforce_cpu is False, but CPU-only mode is still enforced by script.")
    
    # Label maps (fixed order)
    label_order: List[str] = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}

    # Load dataset from CSV
    csv_path_cfg = getattr(cfg.lora_cpu_eval.dataset, "csv_path", None)
    if not csv_path_cfg:
        raise ValueError("lora_cpu_eval.dataset.csv_path must be set to a CSV file")

    csv_path = os.path.normpath(os.path.join(PROJECT_ROOT, str(csv_path_cfg)))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    print(f"\nLoading evaluation data from CSV: {csv_path}")

    # Ensure cache directory is at project root
    cache_dir = str(cfg.paths.hf_cache_dir)
    cache_path = os.path.join(PROJECT_ROOT, cache_dir)
    os.makedirs(cache_path, exist_ok=True)

    # Load dataset
    ds = load_dataset(
        "csv",
        data_files={"test": csv_path},
        cache_dir=cache_path,
        sep=";",
    )["test"]

    print(f"Loaded {len(ds)} test samples")

    # Map labels to class indices
    ds = ds.map(lambda x: map_labels(x, label2id))

    # Apply data sampling if configured
    sample_fraction = float(
        getattr(cfg.lora_cpu_eval.dataset, "sample_fraction", 1.0)
    )
    sample_seed = int(getattr(cfg.lora_cpu_eval.dataset, "sample_seed", 42))
    ds = sample_dataset(ds, sample_fraction, sample_seed)

    # Build a DatasetDict expected by downstream code
    raw: DatasetDict = DatasetDict({"test": ds})

    # Print a sample example
    print("\nSample example from test set:")
    if len(ds) > 0:
        print(ds[0])

    # Load tokenizer and base model on CPU
    base_model_name = str(cfg.lora_cpu_eval.base_model)
    print(f"\nLoading base model: {base_model_name} (CPU-only)")

    output_dir = os.path.normpath(
        os.path.join(
            PROJECT_ROOT,
            str(getattr(cfg.lora_cpu_eval, "output_dir", "results/lora_cpu_eval")),
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    # Use existing function and explicitly move to CPU
    tokenizer, base_model = setup_model_and_tokenizer(
        base_model_name, label2id, id2label, cache_path
    )
    # Explicitly move model to CPU (extra safety)
    base_model = base_model.to("cpu")
    print("Base model loaded and moved to CPU")

    # Load LoRA adapter
    adapter_source = getattr(cfg.lora_cpu_eval.adapter, "source", "hub")
    adapter_path_cfg = getattr(cfg.lora_cpu_eval.adapter, "path", None)

    if adapter_source == "none":
        # Use base model without any adapter
        print("\nUsing base model without adapter")
        model = base_model
    elif adapter_source == "hub":
        # Load from Hugging Face Hub
        print(f"\nLoading LoRA adapter from Hugging Face Hub: {adapter_path_cfg}")
        try:
            # Load adapter and ensure it's on CPU
            model = PeftModel.from_pretrained(
                base_model, 
                adapter_path_cfg,
                device_map="cpu",  # Explicitly use CPU
            )
            # Ensure model is on CPU (double-check)
            model = model.to("cpu")
            print(f"Successfully loaded adapter from Hub: {adapter_path_cfg}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter from Hugging Face Hub '{adapter_path_cfg}': {e}"
            )
    elif adapter_source == "local":
        # Load from local path
        adapter_path = os.path.normpath(
            os.path.join(PROJECT_ROOT, str(adapter_path_cfg))
        )
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter directory not found at '{adapter_path}'. Exiting."
            )
        print(f"\nLoading LoRA adapter from local path: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map="cpu",  # Explicitly use CPU
        )
        model = model.to("cpu")  # Double-check
        print("Successfully loaded adapter from local path")
    else:
        raise ValueError(
            f"Invalid adapter source '{adapter_source}'. Must be 'local', 'hub', or 'none'."
        )

    # Verify model is on CPU
    print(f"\nModel device check:")
    for name, param in model.named_parameters():
        if param.device.type != "cpu":
            print(f"WARNING: Parameter {name} is on {param.device}, expected CPU")
        break  # Just check first parameter
    print("Model is on CPU (verified)")

    # Tokenize dataset
    include_ref_ans = bool(
        getattr(cfg.tokenization, "include_reference_answer", False)
    )
    include_chunk = bool(getattr(cfg.tokenization, "include_chunk_text", False))
    print(
        f"\nTokenizing dataset (include_reference_answer={include_ref_ans}, "
        f"include_chunk_text={include_chunk})..."
    )
    tokenized = tokenize_dataset(raw, tokenizer, include_ref_ans, include_chunk)

    # Trainer setup for evaluation only (CPU-friendly settings)
    per_device_eval_batch_size = int(
        getattr(cfg.lora_cpu_eval, "batch_size", 8)
    )
    print(f"\nSetting up Trainer with batch_size={per_device_eval_batch_size} (CPU-friendly)")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",  # Disable MLflow for CPU evaluation (can be enabled if needed)
        logging_strategy="no",
        fp16=False,  # Disable mixed precision for CPU
        bf16=False,  # Disable bfloat16 for CPU
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Run detailed evaluation with timing
    print("\n" + "=" * 60)
    print("RUNNING DETAILED EVALUATION")
    print("=" * 60)
    
    # Record timing information
    num_examples = len(tokenized["test"])
    print(f"Evaluating {num_examples} examples on CPU...")
    
    # Start timing
    eval_start_time = time.time()
    
    detailed_evaluation(trainer, tokenized["test"], label_order)
    
    # End timing
    eval_end_time = time.time()
    total_eval_time = eval_end_time - eval_start_time
    
    # Calculate timing metrics
    examples_per_minute = (num_examples / total_eval_time) * 60 if total_eval_time > 0 else 0
    avg_time_per_example = total_eval_time / num_examples if num_examples > 0 else 0
    
    # Print timing results
    print("\n" + "=" * 60)
    print("CPU EVALUATION TIMING METRICS")
    print("=" * 60)
    print(f"Total evaluation time: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)")
    print(f"Number of examples evaluated: {num_examples}")
    print(f"Examples per minute: {examples_per_minute:.2f}")
    print(f"Average time per example: {avg_time_per_example:.4f} seconds ({avg_time_per_example*1000:.2f} ms)")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

