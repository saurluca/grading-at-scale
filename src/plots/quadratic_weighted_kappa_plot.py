import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.rcParams["figure.figsize"] = (12, 3)
plt.rcParams["font.size"] = 11

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "gras_eval" / "quadratic_weighted_kappa (5).csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "quadratic_weighted_kappa_chart.png"

# Mapping from CSV Run column values to standardized model groups
RUN_TO_MODEL_GROUP = {
    "lora_Llama-3.2-1B-Instruct": "meta-llama/llama-3.2-1b-instruct",
    "lora_Qwen3-0.6B": "qwen/qwen3-0.6b",
    "lora_gpt2-large": "openai-community/gpt2-large",
    "lora_flan-t5-large": "google/flan-t5-large",
}

# Model name mapping for proper display
MODEL_NAME_MAPPING = {
    "openai/chatgpt-4o": "GPT-4o",
    "meta-llama/llama-3.2-1b-instruct": "Llama 3.2 1B Instruct",
    "qwen/qwen3-0.6b": "Qwen3 0.6B",
    "openai-community/gpt2-large": "GPT-2 Large",
    "google/flan-t5-large": "Flan-T5 Large",
}

# Mmodel order
MODEL_ORDER = {
    "openai/chatgpt-4o": 1,
    "meta-llama/llama-3.2-1b-instruct": 2,
    "google/flan-t5-large": 3,
    "openai-community/gpt2-large": 4,
    "qwen/qwen3-0.6b": 5,
}


def get_model_group(run_name: str) -> str:
    """Map Run column value to standardized model group."""
    # Handle GPT-4o runs (all start with "gpt-4o")
    if run_name.startswith("gpt-4o"):
        return "openai/chatgpt-4o"
    # Handle other models
    return RUN_TO_MODEL_GROUP.get(run_name, run_name)


def format_model_name(model_id: str) -> str:
    """Format model ID to display name."""
    return MODEL_NAME_MAPPING.get(model_id, model_id.replace("/", " ").title())


def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Filter out empty rows
    df = df.dropna(subset=["quadratic_weighted_kappa"])

    # Convert kappa to float
    df["quadratic_weighted_kappa"] = df["quadratic_weighted_kappa"].astype(float)

    # Map Run column to standardized model groups
    df["model_group"] = df["Run"].apply(get_model_group)

    # Calculate statistics (mean and std) for each model group
    stats = (
        df.groupby("model_group")["quadratic_weighted_kappa"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["model_group", "mean_kappa", "std_kappa"]
    stats["std_kappa"] = stats["std_kappa"].fillna(0)  # Fill NaN std with 0

    # Format model names for display
    stats["model_display"] = stats["model_group"].apply(format_model_name)

    # Add order column and sort by custom order
    stats["order"] = stats["model_group"].map(MODEL_ORDER)
    # Fill NaN with a high number for models not in the order list
    stats["order"] = stats["order"].fillna(999)
    df_sorted = stats.sort_values("order").reset_index(drop=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 3))

    # Create color palette - highlight GPT-4o
    colors = [
        "#2E86AB" if model == "GPT-4o" else "#A23B72"
        for model in df_sorted["model_display"]
    ]

    # Create bar chart with error bars
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

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted["mean_kappa"])):
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
            fontsize=10,
        )

    # Customize axes
    ax.set_xlabel("Quadratic Weighted Kappa", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")

    # Set x-axis limits with some padding (accounting for error bars)
    x_max_with_error = (df_sorted["mean_kappa"] + df_sorted["std_kappa"]).max()
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
