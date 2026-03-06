"""
Table 2: Macro-F1 per domain on GRAS (Experiment 1).

Queries MLflow for runs where train=GRAS, test=GRAS, then produces a CSV
with overall and per-domain macro-F1 (mean +/- std) for each model.

Output columns: Model, Overall, AI, Neuroscience, Psychology
Each cell: "0.89 +/- 0.02"

Usage:
    uv run src/plots/macro_f1_table.py
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.mlflow_config import setup_mlflow  # noqa: E402

EXPERIMENT_NAME = "paper_experiments_no_logic"
OUTPUT_PATH = PROJECT_ROOT / "results" / "macro_f1_per_domain.csv"

MODEL_ORDER = [
    ("gpt-4o", "GPT-4o"),
    ("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct"),
    ("google/flan-t5-large", "Flan-T5 Large"),
    ("openai-community/gpt2-large", "GPT-2 Large"),
    ("Qwen/Qwen3-0.6B", "Qwen3 0.6B"),
]

METRIC_COLUMNS = {
    "Overall": "metrics.macro_f1",
    "AI": "metrics.ai_macro_f1",
    "Neuroscience": "metrics.neuro_macro_f1",
    "Psychology": "metrics.psychology_macro_f1",
}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


def main():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    setup_mlflow(cfg, PROJECT_ROOT)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' not found")

    all_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )

    # Collect runs per model
    model_runs: dict[str, pd.DataFrame] = {}

    for model_id, display_name in MODEL_ORDER:
        if model_id == "gpt-4o":
            mask = (
                (all_runs["params.model"] == "gpt-4o")
                & (all_runs["params.test_csv"].str.contains("gras_no_logic", na=False))
            )
        else:
            mask = (
                (all_runs["params.model_name"] == model_id)
                & (all_runs["params.dataset_name"] == "gras_no_logic")
                & (all_runs["params.test_set_name"] == "gras_no_logic")
            )
        model_runs[model_id] = all_runs[mask]

    rows = []
    for model_id, display_name in MODEL_ORDER:
        runs = model_runs[model_id]
        if runs.empty:
            print(f"WARNING: No runs found for {display_name}")
            rows.append({"Model": display_name, **{col: "N/A" for col in METRIC_COLUMNS}})
            continue

        row = {"Model": display_name}
        for col_name, metric_key in METRIC_COLUMNS.items():
            values = runs[metric_key].dropna()
            if values.empty:
                row[col_name] = "N/A"
            else:
                row[col_name] = fmt(values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0)
        rows.append(row)

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Table saved to: {OUTPUT_PATH}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
