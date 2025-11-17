"""
Script to create a parallel coordinate plot of grid search results.
Visualizes relationships between hyperparameters and quadratic weighted kappa performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# Set style
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "grid_search"
    / "quadratic_weighted_kappa-learning_rate-effective_batch_size-lora_alpha-lora_dropout-lora_r.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "results" / "parallel_coordinate_plot.png"

# Parameter columns to visualize
PARAM_COLUMNS = [
    "learning_rate",
    "effective_batch_size",
    "lora_dropout",
    "lora_alpha",
    "lora_r",
]

# Metric column to include as final axis
METRIC_COLUMN = "quadratic_weighted_kappa"


def clean_kappa_value(value):
    """Clean kappa value by removing quotes and leading apostrophes."""
    if pd.isna(value) or value == "":
        return np.nan
    # Convert to string and remove quotes/apostrophes
    value_str = str(value).strip()
    # Remove leading/trailing quotes
    if value_str.startswith('"') and value_str.endswith('"'):
        value_str = value_str[1:-1]
    # Remove leading apostrophe
    if value_str.startswith("'"):
        value_str = value_str[1:]
    # Try to convert to float
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return np.nan


def normalize_column(series):
    """Normalize a pandas Series to [0, 1] range."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Clean quadratic_weighted_kappa column
    df["quadratic_weighted_kappa"] = df["quadratic_weighted_kappa"].apply(clean_kappa_value)

    # Filter out rows with missing/invalid kappa values
    df = df.dropna(subset=["quadratic_weighted_kappa"])

    # Convert parameter columns to numeric
    for col in PARAM_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out rows with missing parameter values
    df = df.dropna(subset=PARAM_COLUMNS)

    if len(df) == 0:
        print("No valid data to plot!")
        return

    # Store original parameter ranges for axis labels
    param_ranges = {}
    for col in PARAM_COLUMNS:
        param_ranges[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
        }
    
    # Store metric range for axis labels
    param_ranges[METRIC_COLUMN] = {
        "min": df[METRIC_COLUMN].min(),
        "max": df[METRIC_COLUMN].max(),
    }

    # Normalize parameters for visualization
    df_normalized = df.copy()
    for col in PARAM_COLUMNS:
        df_normalized[f"{col}_norm"] = normalize_column(df[col])
    
    # Normalize metric for visualization
    df_normalized[f"{METRIC_COLUMN}_norm"] = normalize_column(df[METRIC_COLUMN])

    # Sort by kappa for better visualization (best on top)
    df_normalized = df_normalized.sort_values("quadratic_weighted_kappa", ascending=False)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # All columns including metric
    ALL_COLUMNS = PARAM_COLUMNS + [METRIC_COLUMN]
    
    # Number of axes (parameters + metric)
    n_axes = len(ALL_COLUMNS)
    x_positions = np.linspace(0, 1, n_axes)

    # Set up colormap for kappa values
    kappa_min = df_normalized["quadratic_weighted_kappa"].min()
    kappa_max = df_normalized["quadratic_weighted_kappa"].max()
    norm = Normalize(vmin=kappa_min, vmax=kappa_max)
    cmap = plt.cm.viridis

    # Draw lines for each run
    alpha = 0.13 if len(df_normalized) > 100 else 0.3
    for idx, row in df_normalized.iterrows():
        # Get normalized values for this row (parameters + metric)
        values = [row[f"{col}_norm"] for col in ALL_COLUMNS]
        kappa = row["quadratic_weighted_kappa"]
        color = cmap(norm(kappa))

        # Draw line connecting all parameter values and metric
        ax.plot(
            x_positions,
            values,
            color=color,
            alpha=alpha,
            linewidth=1.4,
        )

    # Draw vertical axes
    for i, col in enumerate(ALL_COLUMNS):
        x_pos = x_positions[i]
        ax.axvline(x=x_pos, color="black", linewidth=1.5, alpha=0.3)

        # Add axis label
        if col == METRIC_COLUMN:
            label = "Quadratic Weighted Kappa"
        else:
            label = col.replace("_", " ").title()
        ax.text(
            x_pos,
            -0.05,
            label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

        # Add value range labels
        min_val = param_ranges[col]["min"]
        max_val = param_ranges[col]["max"]

        # Format values appropriately
        if col == "learning_rate":
            min_str = f"{min_val:.4f}"
            max_str = f"{max_val:.4f}"
        elif col == "lora_dropout":
            min_str = f"{min_val:.2f}"
            max_str = f"{max_val:.2f}"
        elif col == METRIC_COLUMN:
            min_str = f"{min_val:.4f}"
            max_str = f"{max_val:.4f}"
        else:
            min_str = f"{min_val:.0f}"
            max_str = f"{max_val:.0f}"

        ax.text(
            x_pos,
            1.02,
            max_str,
            ha="center",
            va="bottom",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            x_pos,
            -0.02,
            min_str,
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )

    # Set axis limits
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # Remove default axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove spines except bottom
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Quadratic Weighted Kappa", fontsize=11, fontweight="bold", rotation=270, labelpad=20)

    # Tight layout
    plt.tight_layout()

    # Save figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Parallel coordinate plot saved to: {OUTPUT_PATH}")

    plt.close()


if __name__ == "__main__":
    main()

