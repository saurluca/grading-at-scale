"""
Script to create a bar chart of quadratic weighted kappa scores from evaluation results.
Orders models with GPT-4o at the top, then best to worst.
Includes error bars based on aggregated model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
# sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 3)
plt.rcParams["font.size"] = 11

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "gras_eval" / "quadratic_weighted_kappa (1).csv"
AGGREGATED_CSV_PATH = (
    PROJECT_ROOT / "results" / "gras_eval" / "quadratic_weighted_kappa (3).csv"
)
OUTPUT_PATH = PROJECT_ROOT / "results" / "quadratic_weighted_kappa_chart.png"

# Model name mapping for proper display
MODEL_NAME_MAPPING = {
    "openai/chatgpt-4o": "GPT-4o",
    "meta-llama/llama-3.2-1b-instruct": "Llama 3.2 1B Instruct",
    "qwen/qwen3-0.6b": "Qwen3 0.6B",
    "openai-community/gpt2-large": "GPT-2 Large",
    "google/flan-t5-large": "Flan-T5 Large",
}

# Mapping from aggregated CSV model names to original CSV model names
AGGREGATED_TO_ORIGINAL_MAPPING = {
    "lora_Llama-3.2-1B-Instruct": "meta-llama/llama-3.2-1b-instruct",
    "lora_Qwen3-0.6B": "qwen/qwen3-0.6b",
    "lora_gpt2-large": "openai-community/gpt2-large",
    "lora_flan-t5-large": "google/flan-t5-large",
    "openai/chatgpt-4o": "openai/chatgpt-4o",
}


def format_model_name(model_id: str) -> str:
    """Format model ID to display name."""
    return MODEL_NAME_MAPPING.get(model_id, model_id.replace("/", " ").title())


def map_aggregated_to_original(aggregated_name: str) -> str:
    """Map aggregated CSV model name to original CSV model name."""
    return AGGREGATED_TO_ORIGINAL_MAPPING.get(aggregated_name, aggregated_name)


def main():
    # Read main CSV
    df = pd.read_csv(CSV_PATH)

    # Filter out empty rows
    df = df.dropna(subset=["quadratic_weighted_kappa"])

    # Convert kappa to float
    df["quadratic_weighted_kappa"] = df["quadratic_weighted_kappa"].astype(float)

    # Read aggregated CSV for error bars
    df_agg = pd.read_csv(AGGREGATED_CSV_PATH)
    df_agg = df_agg.dropna(subset=["quadratic_weighted_kappa"])
    df_agg["quadratic_weighted_kappa"] = df_agg["quadratic_weighted_kappa"].astype(
        float
    )

    # Map aggregated model names to original model names
    df_agg["group"] = df_agg["Run"].apply(map_aggregated_to_original)

    # Calculate statistics (mean and std) for each model from aggregated data
    agg_stats = (
        df_agg.groupby("group")["quadratic_weighted_kappa"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_stats.columns = ["group", "mean_kappa", "std_kappa"]

    # Merge with main dataframe
    df = df.merge(agg_stats, on="group", how="left")

    # Use mean from aggregated data if available, otherwise use original value
    df["kappa_value"] = df["mean_kappa"].fillna(df["quadratic_weighted_kappa"])
    # Fill NaN std with 0 (no error bar for single values)
    df["std_kappa"] = df["std_kappa"].fillna(0)

    # Format model names
    df["model_display"] = df["group"].apply(format_model_name)

    # Separate GPT-4o from others
    gpt4o_row = df[df["group"] == "openai/chatgpt-4o"].copy()
    other_rows = df[df["group"] != "openai/chatgpt-4o"].copy()

    # Sort others by kappa score (best to worst)
    other_rows = other_rows.sort_values("kappa_value", ascending=False)

    # Combine: GPT-4o first, then others sorted best to worst
    df_sorted = pd.concat([gpt4o_row, other_rows], ignore_index=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 3 ))

    # Create color palette - highlight GPT-4o
    colors = [
        "#2E86AB" if model == "GPT-4o" else "#A23B72"
        for model in df_sorted["model_display"]
    ]

    # Create bar chart with error bars
    bars = ax.barh(
        df_sorted["model_display"],
        df_sorted["kappa_value"],
        xerr=df_sorted["std_kappa"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted["kappa_value"])):
        width = bar.get_width()
        # Position label after error bar if present
        error_offset = (
            df_sorted.iloc[i]["std_kappa"] if df_sorted.iloc[i]["std_kappa"] > 0 else 0
        )
        ax.text(
            width + error_offset + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            ha="left",
            va="center",
            # fontweight="bold",
            fontsize=10,
        )

    # Customize axes
    ax.set_xlabel("Quadratic Weighted Kappa", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")

    # Set x-axis limits with some padding (accounting for error bars)
    x_max_with_error = (df_sorted["kappa_value"] + df_sorted["std_kappa"]).max()
    ax.set_xlim(0, min(1, x_max_with_error + 0.05))

    # Invert y-axis so GPT-4o is at the top
    ax.invert_yaxis()

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout
    plt.tight_layout()

    # Save figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved to: {OUTPUT_PATH}")

    plt.close()


if __name__ == "__main__":
    main()
