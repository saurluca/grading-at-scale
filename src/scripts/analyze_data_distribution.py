#!/usr/bin/env python3
"""
Analyze the distribution of questions, answers, and labels per topic and data split.

This script reads train.csv, val.csv, and test.csv from the data/gras directory
and prints statistics about:
- Questions (unique task_ids)
- Answers (total rows)
- Correct answers (label == 2)
- Partially correct answers (label == 1)
- Incorrect answers (label == 0)
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "gras"


def analyze_split(df: pd.DataFrame, split_name: str) -> dict:
    """Analyze a single data split and return statistics per topic."""
    stats = defaultdict(
        lambda: {
            "questions": set(),
            "answers": 0,
            "correct_answers": 0,
            "partially_correct_answers": 0,
            "incorrect_answers": 0,
        }
    )

    for _, row in df.iterrows():
        topic = row["topic"]
        task_id = row["task_id"]
        label = int(row["labels"])

        stats[topic]["questions"].add(task_id)
        stats[topic]["answers"] += 1

        if label == 2:
            stats[topic]["correct_answers"] += 1
        elif label == 1:
            stats[topic]["partially_correct_answers"] += 1
        elif label == 0:
            stats[topic]["incorrect_answers"] += 1

    # Convert sets to counts
    result = {}
    for topic, data in stats.items():
        result[topic] = {
            "questions": len(data["questions"]),
            "answers": data["answers"],
            "correct_answers": data["correct_answers"],
            "partially_correct_answers": data["partially_correct_answers"],
            "incorrect_answers": data["incorrect_answers"],
        }

    return result


def print_statistics(all_stats: dict, all_dfs: dict):
    """Print statistics in a formatted table."""
    splits = ["train", "val", "test"]

    # Get all topics across all splits
    all_topics = set()
    for split_stats in all_stats.values():
        all_topics.update(split_stats.keys())
    all_topics = sorted(all_topics)

    # Print header
    print("\n" + "=" * 120)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 120)

    # Print per split per topic
    for split in splits:
        print(f"\n{split.upper()} SPLIT:")
        print("-" * 120)
        print(
            f"{'Topic':<20} {'Questions':<12} {'Answers':<12} {'Correct':<12} {'Partial':<12} {'Incorrect':<12}"
        )
        print("-" * 120)

        split_stats = all_stats.get(split, {})
        for topic in all_topics:
            if topic in split_stats:
                s = split_stats[topic]
                print(
                    f"{topic:<20} {s['questions']:<12} {s['answers']:<12} "
                    f"{s['correct_answers']:<12} {s['partially_correct_answers']:<12} "
                    f"{s['incorrect_answers']:<12}"
                )
            else:
                print(f"{topic:<20} {'0':<12} {'0':<12} {'0':<12} {'0':<12} {'0':<12}")

        # Total for this split
        split_df = all_dfs.get(split)
        if split_df is not None:
            total = {
                "questions": split_df["task_id"].nunique(),
                "answers": sum(s["answers"] for s in split_stats.values()),
                "correct_answers": sum(
                    s["correct_answers"] for s in split_stats.values()
                ),
                "partially_correct_answers": sum(
                    s["partially_correct_answers"] for s in split_stats.values()
                ),
                "incorrect_answers": sum(
                    s["incorrect_answers"] for s in split_stats.values()
                ),
            }
        else:
            total = {
                "questions": 0,
                "answers": 0,
                "correct_answers": 0,
                "partially_correct_answers": 0,
                "incorrect_answers": 0,
            }

        print("-" * 120)
        print(
            f"{'TOTAL':<20} {total['questions']:<12} {total['answers']:<12} "
            f"{total['correct_answers']:<12} {total['partially_correct_answers']:<12} "
            f"{total['incorrect_answers']:<12}"
        )

    # Print totals per topic across all splits
    print("\n\nTOTALS PER TOPIC (across all splits):")
    print("-" * 120)
    print(
        f"{'Topic':<20} {'Questions':<12} {'Answers':<12} {'Correct':<12} {'Partial':<12} {'Incorrect':<12}"
    )
    print("-" * 120)

    topic_totals = defaultdict(
        lambda: {
            "questions": set(),
            "answers": 0,
            "correct_answers": 0,
            "partially_correct_answers": 0,
            "incorrect_answers": 0,
        }
    )

    # Collect all data for topic totals
    for split in splits:
        split_df = all_dfs.get(split)
        if split_df is not None:
            for _, row in split_df.iterrows():
                topic = row["topic"]
                task_id = row["task_id"]
                label = int(row["labels"])

                topic_totals[topic]["questions"].add(task_id)
                topic_totals[topic]["answers"] += 1

                if label == 2:
                    topic_totals[topic]["correct_answers"] += 1
                elif label == 1:
                    topic_totals[topic]["partially_correct_answers"] += 1
                elif label == 0:
                    topic_totals[topic]["incorrect_answers"] += 1

    for topic in all_topics:
        if topic in topic_totals:
            t = topic_totals[topic]
            print(
                f"{topic:<20} {len(t['questions']):<12} {t['answers']:<12} "
                f"{t['correct_answers']:<12} {t['partially_correct_answers']:<12} "
                f"{t['incorrect_answers']:<12}"
            )
        else:
            print(f"{topic:<20} {'0':<12} {'0':<12} {'0':<12} {'0':<12} {'0':<12}")

    # Grand total - use unique task_ids across all splits to avoid double-counting
    all_dfs_list = [df for df in all_dfs.values() if df is not None]
    if all_dfs_list:
        combined_df = pd.concat(all_dfs_list, ignore_index=True)
        grand_total = {
            "questions": combined_df[
                "task_id"
            ].nunique(),  # Unique task_ids across all splits
            "answers": sum(t["answers"] for t in topic_totals.values()),
            "correct_answers": sum(t["correct_answers"] for t in topic_totals.values()),
            "partially_correct_answers": sum(
                t["partially_correct_answers"] for t in topic_totals.values()
            ),
            "incorrect_answers": sum(
                t["incorrect_answers"] for t in topic_totals.values()
            ),
        }
    else:
        grand_total = {
            "questions": 0,
            "answers": 0,
            "correct_answers": 0,
            "partially_correct_answers": 0,
            "incorrect_answers": 0,
        }

    print("-" * 120)
    print(
        f"{'GRAND TOTAL':<20} {grand_total['questions']:<12} {grand_total['answers']:<12} "
        f"{grand_total['correct_answers']:<12} {grand_total['partially_correct_answers']:<12} "
        f"{grand_total['incorrect_answers']:<12}"
    )
    print("=" * 120 + "\n")


def verify_no_overlap(all_dfs: dict):
    """Verify that task_ids don't overlap between splits."""
    splits = ["train", "val", "test"]
    task_id_sets = {}

    for split in splits:
        if split in all_dfs and all_dfs[split] is not None:
            task_id_sets[split] = set(all_dfs[split]["task_id"].unique())
        else:
            task_id_sets[split] = set()

    overlaps = []
    if "train" in task_id_sets and "val" in task_id_sets:
        train_val_overlap = task_id_sets["train"] & task_id_sets["val"]
        if train_val_overlap:
            overlaps.append(f"Train-Val: {len(train_val_overlap)} overlapping task_ids")

    if "train" in task_id_sets and "test" in task_id_sets:
        train_test_overlap = task_id_sets["train"] & task_id_sets["test"]
        if train_test_overlap:
            overlaps.append(
                f"Train-Test: {len(train_test_overlap)} overlapping task_ids"
            )

    if "val" in task_id_sets and "test" in task_id_sets:
        val_test_overlap = task_id_sets["val"] & task_id_sets["test"]
        if val_test_overlap:
            overlaps.append(f"Val-Test: {len(val_test_overlap)} overlapping task_ids")

    if overlaps:
        print("\n⚠️  WARNING: Found overlapping task_ids between splits:")
        for overlap in overlaps:
            print(f"   {overlap}")
        print(
            "   This indicates the splits were not properly separated by questions.\n"
        )
    else:
        print("\n✓ Verified: No task_id overlap between splits.\n")


def main():
    """Main function to run the analysis."""
    splits = ["train", "val", "test"]
    all_stats = {}
    all_dfs = {}

    # Read and analyze each split
    for split in splits:
        csv_path = DATA_DIR / f"{split}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue

        df = pd.read_csv(csv_path, sep=";")
        all_dfs[split] = df
        all_stats[split] = analyze_split(df, split)

    # Verify no overlaps
    verify_no_overlap(all_dfs)

    # Print statistics
    print_statistics(all_stats, all_dfs)


if __name__ == "__main__":
    main()
