"""
Bar chart of Quadratic Weighted Kappa on the GRAS dataset (Experiment 1).

Queries MLflow for all runs in the experiment where both train and test
are on gras, then plots mean +/- std per model.

Usage:
    uv run src/plots/quadratic_weighted_kappa_plot.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.mlflow_config import setup_mlflow  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

EXPERIMENT_NAME = "paper_experiments"
OUTPUT_PATH = PROJECT_ROOT / "results" / "quadratic_weighted_kappa_chart.png"

MODEL_NAME_MAPPING = {
    "gpt-4o": "GPT-4o",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B Instruct",
    "google/flan-t5-large": "Flan-T5 Large",
    "openai-community/gpt2-large": "GPT-2 Large",
    "Qwen/Qwen3-0.6B": "Qwen3 0.6B",
}

MODEL_ORDER = [
    "gpt-4o",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/flan-t5-large",
    "openai-community/gpt2-large",
    "Qwen/Qwen3-0.6B",
]


def query_exp1_runs() -> pd.DataFrame:
    """Fetch Experiment 1 runs (train GRAS, test GRAS) from MLflow."""
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    setup_mlflow(cfg, PROJECT_ROOT)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' not found")

    all_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )

    rows = []

    # Fine-tuned models: dataset_name=gras, test on gras test.csv
    ft_mask = (all_runs["params.dataset_name"] == "gras") & (
        all_runs["params.test_set_name"] == "gras"
    )
    for _, run in all_runs[ft_mask].iterrows():
        rows.append(
            {
                "model": run["params.model_name"],
                "quadratic_weighted_kappa": run["metrics.quadratic_weighted_kappa"],
            }
        )

    # GPT-4o runs (dspy_eval)
    gpt_mask = (all_runs["params.model"] == "gpt-4o") & (
        all_runs["params.test_csv"].str.contains("gras", na=False)
    )
    for _, run in all_runs[gpt_mask].iterrows():
        rows.append(
            {
                "model": "gpt-4o",
                "quadratic_weighted_kappa": run["metrics.quadratic_weighted_kappa"],
            }
        )

    return pd.DataFrame(rows)


def main():
    df = query_exp1_runs()
    if df.empty:
        print("No runs found. Make sure experiments have been executed.")
        return

    stats = (
        df.groupby("model")["quadratic_weighted_kappa"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["model", "mean_kappa", "std_kappa"]
    stats["std_kappa"] = stats["std_kappa"].fillna(0)

    stats["model_display"] = (
        stats["model"].map(MODEL_NAME_MAPPING).fillna(stats["model"])
    )
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    stats["order"] = stats["model"].map(order_map).fillna(999)
    df_sorted = stats.sort_values("order").reset_index(drop=True)

    # Journal: max width 20 cm, 300 DPI (saved below)
    MAX_WIDTH_CM = 20
    INCH_PER_CM = 2.54
    width_in = MAX_WIDTH_CM / INCH_PER_CM
    height_in = width_in * (3 / 12)  # preserve 12:3 aspect ratio
    plt.rcParams["figure.figsize"] = (width_in, height_in)
    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    colors = [
        "#2E86AB" if m == "GPT-4o" else "#A23B72" for m in df_sorted["model_display"]
    ]

    bars = ax.barh(
        df_sorted["model_display"],
        df_sorted["mean_kappa"],
        xerr=df_sorted["std_kappa"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    for i, (bar, value) in enumerate(zip(bars, df_sorted["mean_kappa"])):
        error_offset = (
            df_sorted.iloc[i]["std_kappa"] if df_sorted.iloc[i]["std_kappa"] > 0 else 0
        )
        ax.text(
            bar.get_width() + error_offset + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Quadratic Weighted Kappa", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")

    x_max = (df_sorted["mean_kappa"] + df_sorted["std_kappa"]).max()
    ax.set_xlim(0, min(1, x_max + 0.05))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved to: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
