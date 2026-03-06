"""
Filter the 'logic' topic from the GRAS dataset and re-split into train/val/test.

Reads data/gras/full.csv, removes all rows with topic=='logic',
saves filtered full.csv to data/gras_no_logic/, then performs a
stratified 60/20/20 split by task_id (identical logic to split_data.py).

Usage:
    uv run src/data_prep/prepare_gras_no_logic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
seed = cfg.project.seed

EXCLUDE_TOPIC = "logic"
INPUT_PATH = PROJECT_ROOT / "data" / "gras" / "full.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gras_no_logic"

label_order = ["incorrect", "partial", "correct"]
label2id = {name: i for i, name in enumerate(label_order)}


def map_label(label_raw):
    try:
        label_num = float(label_raw)
        if label_num.is_integer():
            return int(label_num)
        raise ValueError(f"Label '{label_raw}' is not an integer class index.")
    except (ValueError, TypeError):
        label_val = str(label_raw).strip().lower()
        if label_val in label2id:
            return label2id[label_val]
        raise ValueError(f"Label '{label_raw}' not in {label2id}")


def stratified_split(df, seed):
    """Split by task_id with stratification by topic and label distribution."""
    task_id_to_topic = df.groupby("task_id")["topic"].first().to_dict()
    topics_to_task_ids = {}
    for task_id, topic in task_id_to_topic.items():
        topics_to_task_ids.setdefault(topic, []).append(task_id)

    task_id_label_counts = {}
    for task_id in df["task_id"].unique():
        counts = df[df["task_id"] == task_id]["labels"].value_counts().to_dict()
        task_id_label_counts[task_id] = {i: counts.get(i, 0) for i in range(3)}

    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []

    for topic, task_ids in sorted(topics_to_task_ids.items()):
        shuffled = task_ids.copy()
        rng.shuffle(shuffled)
        n_total = len(shuffled)

        topic_label_counts = np.array([0, 0, 0])
        for tid in shuffled:
            topic_label_counts += np.array([task_id_label_counts[tid][i] for i in range(3)])

        target_counts = {
            "train": np.round(topic_label_counts * 0.6).astype(int),
            "val": np.round(topic_label_counts * 0.2).astype(int),
            "test": np.round(topic_label_counts * 0.2).astype(int),
        }
        for i in range(3):
            diff = topic_label_counts[i] - sum(target_counts[s][i] for s in target_counts)
            target_counts["train"][i] += diff

        if n_total <= 2:
            n_train, n_val, n_test = n_total, 0, 0
        elif n_total == 3:
            n_train, n_val, n_test = 1, 1, 1
        else:
            n_test = max(1, round(n_total * 0.2))
            n_val = max(1, round(n_total * 0.2))
            n_train = n_total - n_test - n_val

        splits = {
            "train": ([], np.array([0, 0, 0]), n_train),
            "val": ([], np.array([0, 0, 0]), n_val),
            "test": ([], np.array([0, 0, 0]), n_test),
        }

        for tid in shuffled:
            task_labels = np.array([task_id_label_counts[tid][i] for i in range(3)])
            if n_total <= 2:
                best_split = "train"
            else:
                best_split = "train"
                best_distance = float("inf")
                for name, (task_list, label_counts, max_q) in splits.items():
                    if len(task_list) >= max_q:
                        continue
                    new_counts = label_counts + task_labels
                    distance = np.sum(np.abs(new_counts - target_counts[name]))
                    if distance < best_distance:
                        best_distance = distance
                        best_split = name

            task_list, label_counts, _ = splits[best_split]
            task_list.append(tid)
            splits[best_split] = (task_list, label_counts + task_labels, splits[best_split][2])

        train_ids.extend(splits["train"][0])
        val_ids.extend(splits["val"][0])
        test_ids.extend(splits["test"][0])

        print(
            f"  Topic '{topic}': {n_total} questions -> "
            f"train={len(splits['train'][0])}, val={len(splits['val'][0])}, test={len(splits['test'][0])}"
        )

    return set(train_ids), set(val_ids), set(test_ids)


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, sep=";")
    print(f"  Total rows: {len(df)}, topics: {sorted(df['topic'].unique())}")

    before = len(df)
    df = df[df["topic"] != EXCLUDE_TOPIC].copy()
    print(f"  Removed {before - len(df)} rows with topic='{EXCLUDE_TOPIC}'")
    print(f"  Remaining: {len(df)} rows, topics: {sorted(df['topic'].unique())}")

    df["labels"] = df["labels"].apply(map_label)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_out = OUTPUT_DIR / "full.csv"
    df.to_csv(full_out, index=False, sep=";")
    print(f"  Saved filtered full.csv to {full_out}")

    print("Splitting into train/val/test (60/20/20) by task_id ...")
    train_ids, val_ids, test_ids = stratified_split(df, seed)

    df_train = df[df["task_id"].isin(train_ids)]
    df_val = df[df["task_id"].isin(val_ids)]
    df_test = df[df["task_id"].isin(test_ids)]

    print(f"\nFinal sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out = OUTPUT_DIR / f"{name}.csv"
        split_df.to_csv(out, index=False, sep=";")
        print(f"  Saved {name} ({len(split_df)} rows) -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
