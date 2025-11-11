# %%
import os
from pathlib import Path
import sys
import itertools
from typing import Dict, List, Tuple
import numpy as np
import mlflow
import pandas as pd
from datasets import load_dataset, ClassLabel, DatasetDict, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.common import (  # noqa: E402
    setup_training_args,
    setup_trainer,
    tokenize_dataset,
    detailed_evaluation,
    setup_model_and_tokenizer,
    map_labels,
)
from src.mlflow_config import setup_mlflow  # noqa: E402

# %%
def split_by_topics(
    full_dataset,
    train_topics: List[str],
    test_topics: List[str],
    seed: int = 42,
    val_size: float = 0.2,
    out_of_fold_samples: int = 0,
) -> Tuple[DatasetDict, List[str], Dict[str, int], Dict[int, str]]:
    """
    Split dataset by topics: train on train_topics, test on test_topics.
    Within each topic, splits by task_id to avoid question overlap.
    
    Args:
        full_dataset: Full dataset with all topics
        train_topics: List of topics to use for training
        test_topics: List of topics to use for testing
        seed: Random seed for reproducibility
        val_size: Fraction of train data to use for validation
        out_of_fold_samples: Number of samples from test topics to include in training
        
    Returns:
        DatasetDict with train, val, test splits
        label_order, label2id, id2label
    """
    print(f"\n{'='*60}")
    print(f"Splitting: Train on {train_topics}, Test on {test_topics}")
    print(f"{'='*60}")
    
    # Labels mapping
    label_order = ["incorrect", "partial", "correct"]
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_order)}
    id2label: Dict[int, str] = {i: name for name, i in label2id.items()}
    
    # Map labels if not already mapped
    if "labels" not in full_dataset.column_names:
        # If no labels column, check for "label" column
        if "label" in full_dataset.column_names:
            full_dataset = full_dataset.map(lambda x: map_labels(x, label2id))
        else:
            raise ValueError("Dataset must have either 'labels' or 'label' column")
    elif len(full_dataset) > 0 and isinstance(full_dataset[0]["labels"], str):
        # Labels are strings, need to map them
        full_dataset = full_dataset.map(lambda x: map_labels(x, label2id))
    
    # Ensure labels are ClassLabel type
    if "labels" in full_dataset.column_names:
        full_dataset = full_dataset.cast_column("labels", ClassLabel(names=label_order))
    
    # Filter data by topics
    train_data_indices = [
        i for i, ex in enumerate(full_dataset) if ex["topic"] in train_topics
    ]
    test_data_indices = [
        i for i, ex in enumerate(full_dataset) if ex["topic"] in test_topics
    ]
    
    train_data = full_dataset.select(train_data_indices)
    test_data = full_dataset.select(test_data_indices)
    
    print(f"Train topics data: {len(train_data)} samples")
    print(f"Test topics data: {len(test_data)} samples")
    
    # Split train data by task_id to create train/val split
    # Group task_ids by topic for stratified splitting
    train_df = train_data.to_pandas()
    task_id_to_topic = train_df.groupby("task_id")["topic"].first().to_dict()
    topics_to_task_ids = {}
    for task_id, topic in task_id_to_topic.items():
        if topic not in topics_to_task_ids:
            topics_to_task_ids[topic] = []
        topics_to_task_ids[topic].append(task_id)
    
    # Split task_ids by topic to maintain topic proportions
    rng = np.random.default_rng(seed)
    train_task_ids = []
    val_task_ids = []
    
    for topic, task_ids in sorted(topics_to_task_ids.items()):
        # Shuffle task_ids for this topic
        shuffled_task_ids = task_ids.copy()
        rng.shuffle(shuffled_task_ids)
        
        # Calculate split sizes for this topic
        n_total = len(shuffled_task_ids)
        
        if n_total == 1:
            # Only 1 question: put in train (can't split)
            n_train = 1
            n_val = 0
        else:
            # For 2+ questions, use proportional split
            n_val = max(1, round(n_total * val_size))
            n_train = n_total - n_val
        
        topic_train = shuffled_task_ids[:n_train]
        topic_val = shuffled_task_ids[n_train:] if n_val > 0 else []
        
        train_task_ids.extend(topic_train)
        if topic_val:
            val_task_ids.extend(topic_val)
        
        print(
            f"  Topic '{topic}': {n_total} questions -> train={len(topic_train)}, val={len(topic_val)}"
        )
    
    train_task_ids = set(train_task_ids)
    val_task_ids = set(val_task_ids)
    
    print(
        f"Total task_ids: train={len(train_task_ids)}, val={len(val_task_ids)}"
    )
    
    # Filter train data based on task_id assignment
    train_indices = [
        i
        for i, task_id in enumerate(train_data["task_id"])
        if task_id in train_task_ids
    ]
    val_indices = [
        i
        for i, task_id in enumerate(train_data["task_id"])
        if task_id in val_task_ids
    ]
    
    # If no validation set created, use a small portion of train as val
    if not val_indices and len(train_indices) > 10:
        # Use 10% of train as val
        n_val = max(1, len(train_indices) // 10)
        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]
        print(f"Warning: No validation set created, using {len(val_indices)} samples from train")
    elif not val_indices:
        # If train is too small, use train as val (trainer will handle this)
        val_indices = train_indices.copy()
        print(f"Warning: Train set too small, using train as validation set")
    
    # Test data uses all task_ids from test topics
    test_indices = list(range(len(test_data)))
    
    # Optionally include samples from test topics in training (for few-shot experiments)
    oof_samples_added = 0
    if out_of_fold_samples > 0:
        print(f"\nIncluding {out_of_fold_samples} samples from test topics in training...")
        
        # Convert test_data to pandas for easier sampling
        test_df = test_data.to_pandas()
        
        # Sample stratified by label to get balanced representation
        oof_indices = []
        labels_in_test = test_df["labels"].unique()
        
        # Calculate samples per label (try to balance)
        samples_per_label = max(1, out_of_fold_samples // len(labels_in_test))
        remaining_samples = out_of_fold_samples - (samples_per_label * len(labels_in_test))
        
        rng = np.random.default_rng(seed)
        
        for label in labels_in_test:
            label_indices = test_df[test_df["labels"] == label].index.tolist()
            if len(label_indices) == 0:
                continue
            
            # Shuffle and take samples
            shuffled = label_indices.copy()
            rng.shuffle(shuffled)
            
            # Take samples_per_label, plus one extra if we have remaining_samples
            n_to_take = samples_per_label + (1 if remaining_samples > 0 else 0)
            if remaining_samples > 0:
                remaining_samples -= 1
            
            n_to_take = min(n_to_take, len(shuffled))
            oof_indices.extend(shuffled[:n_to_take])
            
            if len(oof_indices) >= out_of_fold_samples:
                oof_indices = oof_indices[:out_of_fold_samples]
                break
        
        # If we still need more samples, fill randomly
        if len(oof_indices) < out_of_fold_samples:
            remaining_indices = [i for i in range(len(test_data)) if i not in oof_indices]
            rng.shuffle(remaining_indices)
            needed = out_of_fold_samples - len(oof_indices)
            oof_indices.extend(remaining_indices[:needed])
        
        # Convert back to dataset indices (test_data indices)
        oof_samples_added = len(oof_indices)
        
        # Add out-of-fold samples to training set
        oof_samples_dataset = test_data.select(oof_indices)
        
        # Remove out-of-fold samples from test set
        remaining_test_indices = [i for i in range(len(test_data)) if i not in oof_indices]
        
        # Combine train data with out-of-fold samples
        train_with_oof = concatenate_datasets([
            train_data.select(train_indices),
            oof_samples_dataset
        ])
        
        print(f"  Added {oof_samples_added} samples from test topics to training")
        print(f"  Test set reduced from {len(test_data)} to {len(remaining_test_indices)} samples")
        
        raw = DatasetDict(
            {
                "train": train_with_oof,
                "val": train_data.select(val_indices),
                "test": test_data.select(remaining_test_indices),
            }
        )
    else:
        raw = DatasetDict(
            {
                "train": train_data.select(train_indices),
                "val": train_data.select(val_indices),
                "test": test_data.select(test_indices),
            }
        )
    
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(raw['train'])} samples" + (f" (including {oof_samples_added} from test topics)" if oof_samples_added > 0 else ""))
    print(f"  Val: {len(raw['val'])} samples")
    print(f"  Test: {len(raw['test'])} samples")
    
    return raw, label_order, label2id, id2label


def extract_topic_weighted_f1(
    trainer, test_dataset, test_topics: List[str]
) -> Dict[str, float]:
    """
    Extract weighted F1 score per topic from test set.
    
    Returns:
        Dict mapping topic -> weighted_f1 score
    """
    # Get predictions
    predictions = trainer.predict(test_dataset)
    
    # Handle case where predictions.predictions might be a tuple/list
    logits = predictions.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    
    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids
    
    # Get topic information from test dataset
    if "topic" not in test_dataset.column_names:
        return {}
    
    test_topics_list = test_dataset["topic"]
    unique_topics = list(set(test_topics_list))
    
    topic_f1_scores = {}
    for topic in unique_topics:
        if topic not in test_topics:
            continue
            
        # Find indices where the topic matches
        topic_indices = [i for i, t in enumerate(test_topics_list) if t == topic]
        
        if len(topic_indices) == 0:
            topic_f1_scores[topic] = 0.0
            continue
        
        # Get predictions and labels for this topic
        topic_y_true = np.array([y_true[i] for i in topic_indices])
        topic_y_pred = np.array([y_pred[i] for i in topic_indices])
        
        # Calculate weighted F1 for this topic
        _, _, topic_f1_weighted, _ = precision_recall_fscore_support(
            topic_y_true, topic_y_pred, average="weighted", zero_division=0
        )
        
        topic_f1_scores[topic] = topic_f1_weighted
    
    return topic_f1_scores


def main() -> None:
    print("Loading config...")
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    training_cfg_path = os.environ.get(
        "TRAINING_CONFIG_PATH",
        str(PROJECT_ROOT / "configs" / "training.yaml"),
    )
    training_cfg = OmegaConf.load(training_cfg_path)
    cfg = OmegaConf.merge(base_cfg, training_cfg)
    
    # Setup MLflow tracking URI from config
    setup_mlflow(cfg, PROJECT_ROOT)
    
    # Start MLflow experiment for topic k-fold cross-validation
    experiment_name = "topic_kfold_cross_validation"
    mlflow.set_experiment(experiment_name)
    
    # Load full dataset
    dataset_csv = str(PROJECT_ROOT / "data" / cfg.dataset.dataset_name / "full.csv")
    print(f"Loading full dataset from {dataset_csv}...")
    
    cache_dir: str | None = str(cfg.paths.hf_cache_dir) if "paths" in cfg else None
    if cache_dir:
        cache_path = (
            os.path.join(PROJECT_ROOT, cache_dir)
            if not os.path.isabs(cache_dir)
            else cache_dir
        )
        os.makedirs(cache_path, exist_ok=True)
    else:
        cache_path = None
    
    full_dataset = load_dataset(
        "csv",
        data_files={"data": dataset_csv},
        cache_dir=cache_path,
        sep=";",
    )["data"]
    
    # Extract unique topics
    unique_topics = sorted(list(set(full_dataset["topic"])))
    print(f"\nFound {len(unique_topics)} unique topics: {unique_topics}")
    
    # Get k-fold configuration
    kfold_config = getattr(cfg, "kfold", {})
    train_topics_count = int(getattr(kfold_config, "train_topics", 3))  # Default to 3
    
    # Validate configuration
    if train_topics_count not in [2, 3]:
        print(f"Warning: train_topics must be 2 or 3, got {train_topics_count}. Defaulting to 3.")
        train_topics_count = 3
    
    if len(unique_topics) < 4:
        print(f"Warning: Expected at least 4 topics, found {len(unique_topics)}")
    
    test_topics_count = len(unique_topics) - train_topics_count
    print(f"\nK-fold configuration: Train on {train_topics_count} topics, Test on {test_topics_count} topics")
    
    # Get out-of-fold samples configuration
    kfold_config = getattr(cfg, "kfold", {})
    out_of_fold_samples = int(getattr(kfold_config, "out_of_fold_samples", 0))
    
    if out_of_fold_samples > 0:
        print(f"Out-of-fold samples: {out_of_fold_samples} samples from test topics will be included in training")
    else:
        print(f"Out-of-fold samples: Disabled (0)")
    
    # Generate all combinations of train_topics_count topics for training
    train_topic_combinations = list(itertools.combinations(unique_topics, train_topics_count))
    print(f"\nGenerated {len(train_topic_combinations)} topic combinations for training:")
    for i, combo in enumerate(train_topic_combinations, 1):
        test_combo = [t for t in unique_topics if t not in combo]
        print(f"  {i}. Train: {combo}, Test: {test_combo}")
    
    # Storage for results
    results = []
    topic_performance = {topic: [] for topic in unique_topics}
    
    model_name: str = str(cfg.model.base)
    seed = int(getattr(cfg.project, "seed", 42))
    
    # Start parent run for the entire k-fold experiment
    with mlflow.start_run(run_name=f"topic_kfold_{model_name.split('/')[-1]}"):
        # Log experiment-level parameters
        mlflow.log_params({
            "experiment_type": "topic_kfold_cross_validation",
            "model_name": model_name,
            "dataset_name": str(cfg.dataset.dataset_name),
            "num_folds": len(train_topic_combinations),
            "num_topics": len(unique_topics),
            "topics": ", ".join(unique_topics),
            "kfold_train_topics": train_topics_count,
            "kfold_test_topics": test_topics_count,
            "kfold_out_of_fold_samples": out_of_fold_samples,
        })
        
        # Process each fold
        for fold_idx, train_topics in enumerate(train_topic_combinations, 1):
            train_topics = list(train_topics)
            test_topics = [t for t in unique_topics if t not in train_topics]
            
            print(f"\n{'#'*80}")
            print(f"FOLD {fold_idx}/{len(train_topic_combinations)}")
            print(f"Train topics: {train_topics}")
            print(f"Test topics: {test_topics}")
            print(f"{'#'*80}")
            
            # Create MLflow run for this fold
            run_name = f"fold_{fold_idx}_{'_'.join(train_topics)}_vs_{'_'.join(test_topics)}"
            with mlflow.start_run(run_name=run_name, nested=True):
                # Log dataset input
                try:
                    raw_df = pd.read_csv(dataset_csv, delimiter=";")
                    if "labels" in raw_df.columns:
                        raw_df["labels"] = raw_df["labels"].astype("int64")
                    ds_name = str(getattr(cfg.dataset, "dataset_name", Path(dataset_csv).stem))
                    import warnings
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
                
                # Log parameters for this fold
                mlflow.log_params({
                    "fold": fold_idx,
                    "model_name": model_name,
                    "dataset_name": str(cfg.dataset.dataset_name),
                    "dataset_csv": dataset_csv,
                    "train_topics": ", ".join(train_topics),
                    "test_topics": ", ".join(test_topics),
                    "out_of_fold_samples": out_of_fold_samples,
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
                    "seed": seed + fold_idx,
                    "include_reference_answer": bool(
                        getattr(cfg.tokenization, "include_reference_answer", False)
                    ),
                    "include_chunk_text": bool(
                        getattr(cfg.tokenization, "include_chunk_text", False)
                    ),
                })
                
                # Split data by topics
                raw_data, label_order, label2id, id2label = split_by_topics(
                    full_dataset,
                    train_topics=train_topics,
                    test_topics=test_topics,
                    seed=seed + fold_idx,  # Different seed per fold
                    val_size=0.2,
                    out_of_fold_samples=out_of_fold_samples,
                )
                
                # Log dataset sizes
                mlflow.log_params({
                    "train_size": len(raw_data["train"]),
                    "val_size": len(raw_data["val"]),
                    "test_size": len(raw_data["test"]),
                })
                
                # Setup model and tokenizer
                tokenizer, base_model = setup_model_and_tokenizer(
                    model_name, label2id, id2label, cache_path
                )
                
                # Setup LoRA model
                print("Setting up LoRA configuration...")
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
                include_chunk = bool(getattr(cfg.tokenization, "include_chunk_text", False))
                tokenized_data = tokenize_dataset(
                    raw_data, tokenizer, include_ref_ans, include_chunk
                )
                
                # Setup training arguments and trainer
                output_dir = str(PROJECT_ROOT / cfg.output.dir / f"topic_kfold_fold_{fold_idx}")
                os.makedirs(output_dir, exist_ok=True)
                
                training_args = setup_training_args(cfg, output_dir)
                trainer = setup_trainer(model, training_args, tokenized_data, tokenizer, cfg)
                
                # Training
                print("Starting training...")
                trainer.train()
                
                # Evaluation
                print("\nPerforming evaluation on test dataset...")
                detailed_metrics = detailed_evaluation(
                    trainer, tokenized_data["test"], label_order
                )
                
                # Log all detailed metrics
                mlflow.log_metrics(detailed_metrics)
                
                overall_weighted_f1 = detailed_metrics.get("weighted_f1", 0.0)
                
                # Extract per-topic weighted F1 scores
                topic_f1_scores = extract_topic_weighted_f1(
                    trainer, tokenized_data["test"], test_topics
                )
                
                # Log per-topic metrics
                for topic, f1 in topic_f1_scores.items():
                    mlflow.log_metric(f"topic_{topic}_weighted_f1", f1)
                
                # Store results
                fold_result = {
                    "fold": fold_idx,
                    "train_topics": train_topics,
                    "test_topics": test_topics,
                    "overall_weighted_f1": overall_weighted_f1,
                    "topic_f1_scores": topic_f1_scores,
                }
                results.append(fold_result)
                
                # Accumulate per-topic performance
                for topic in test_topics:
                    if topic in topic_f1_scores:
                        topic_performance[topic].append(topic_f1_scores[topic])
                
                print(f"\nFold {fold_idx} Results:")
                print(f"  Overall Weighted F1: {overall_weighted_f1:.4f}")
                for topic, f1 in topic_f1_scores.items():
                    print(f"  {topic} Weighted F1: {f1:.4f}")
        
        # Print summary results
        print(f"\n{'='*80}")
        print("SUMMARY RESULTS")
        print(f"{'='*80}\n")
        
        # Per-split results table
        print("Per-Split Results:")
        print("-" * 80)
        print(f"{'Fold':<6} {'Train Topics':<30} {'Test Topics':<30} {'Overall F1':<12}")
        print("-" * 80)
        for result in results:
            train_str = ", ".join(result["train_topics"])
            test_str = ", ".join(result["test_topics"])
            print(
                f"{result['fold']:<6} {train_str:<30} {test_str:<30} {result['overall_weighted_f1']:<12.4f}"
            )
        
        # Per-topic averaged performance
        print(f"\n{'='*80}")
        print("Averaged Performance per Topic (when tested):")
        print("-" * 80)
        print(f"{'Topic':<20} {'Avg Weighted F1':<20} {'Std Dev':<15} {'# Folds':<10}")
        print("-" * 80)
        
        topic_summary_metrics = {}
        for topic in sorted(unique_topics):
            if len(topic_performance[topic]) > 0:
                avg_f1 = np.mean(topic_performance[topic])
                std_f1 = np.std(topic_performance[topic])
                n_folds = len(topic_performance[topic])
                print(
                    f"{topic:<20} {avg_f1:<20.4f} {std_f1:<15.4f} {n_folds:<10}"
                )
                topic_summary_metrics[f"topic_{topic}_avg_weighted_f1"] = avg_f1
                topic_summary_metrics[f"topic_{topic}_std_weighted_f1"] = std_f1
                topic_summary_metrics[f"topic_{topic}_n_folds"] = n_folds
            else:
                print(f"{topic:<20} {'N/A':<20} {'N/A':<15} {0:<10}")
        
        # Overall statistics
        print(f"\n{'='*80}")
        print("Overall Statistics:")
        print("-" * 80)
        all_overall_f1 = [r["overall_weighted_f1"] for r in results]
        mean_overall_f1 = np.mean(all_overall_f1)
        std_overall_f1 = np.std(all_overall_f1)
        min_overall_f1 = np.min(all_overall_f1)
        max_overall_f1 = np.max(all_overall_f1)
        
        print(f"Mean Overall Weighted F1 across all folds: {mean_overall_f1:.4f}")
        print(f"Std Overall Weighted F1 across all folds: {std_overall_f1:.4f}")
        print(f"Min Overall Weighted F1: {min_overall_f1:.4f}")
        print(f"Max Overall Weighted F1: {max_overall_f1:.4f}")
        
        # Log summary to MLflow (within parent run)
        with mlflow.start_run(run_name="summary", nested=True):
            # Log overall statistics
            mlflow.log_metrics({
                "mean_overall_weighted_f1": mean_overall_f1,
                "std_overall_weighted_f1": std_overall_f1,
                "min_overall_weighted_f1": min_overall_f1,
                "max_overall_weighted_f1": max_overall_f1,
            })
            
            # Log per-topic summary metrics
            mlflow.log_metrics(topic_summary_metrics)
            
            # Log per-fold results as a table artifact
            summary_df = pd.DataFrame([
                {
                    "fold": r["fold"],
                    "train_topics": ", ".join(r["train_topics"]),
                    "test_topics": ", ".join(r["test_topics"]),
                    "overall_weighted_f1": r["overall_weighted_f1"],
                    **{f"topic_{t}_weighted_f1": r["topic_f1_scores"].get(t, 0.0) for t in unique_topics}
                }
                for r in results
            ])
            mlflow.log_table(data=summary_df, artifact_file="per_fold_results.json")
        
        print("\n\nK-fold cross-validation completed!")


if __name__ == "__main__":
    print("Starting topic-based k-fold cross-validation...")
    main()

