"""
Table 3: Cross-domain transfer results.

Queries MLflow for Experiment 2 (GRAS->SciEntsBank) and Experiment 3
(SciEntsBank->GRAS), then produces a CSV with QWK and Macro-F1 for
each direction.

Column hierarchy:
  QWK:      D1->D2 (GRAS->SciEntsBank)  |  D2->D1 (SciEntsBank->GRAS)
  Macro-F1: D1->D2                       |  D2->D1

Each cell: "0.89 +/- 0.02"

Usage:
    uv run src/plots/cross_domain_table.py
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
OUTPUT_PATH = PROJECT_ROOT / "results" / "cross_domain_transfer.csv"

MODEL_ORDER = [
    ("gpt-4o", "GPT-4o"),
    ("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct"),
    ("google/flan-t5-large", "Flan-T5 Large"),
    ("openai-community/gpt2-large", "GPT-2 Large"),
    ("Qwen/Qwen3-0.6B", "Qwen3 0.6B"),
]


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


def get_runs_for_direction(all_runs: pd.DataFrame, model_id: str, direction: str) -> pd.DataFrame:
    """
    Filter runs for a model and transfer direction.
    direction: "d1_to_d2" = train GRAS test SciEntsBank (exp 2)
               "d2_to_d1" = train SciEntsBank test GRAS (exp 3)
    """
    if model_id == "gpt-4o":
        if direction == "d1_to_d2":
            return all_runs[
                (all_runs["params.model"] == "gpt-4o")
                & (all_runs["params.test_csv"].str.contains("SciEntsBank_3way", na=False))
            ]
        else:
            return all_runs[
                (all_runs["params.model"] == "gpt-4o")
                & (all_runs["params.test_csv"].str.contains("gras_no_logic", na=False))
            ]
    else:
        if direction == "d1_to_d2":
            return all_runs[
                (all_runs["params.model_name"] == model_id)
                & (all_runs["params.dataset_name"] == "gras_no_logic")
                & (all_runs["params.test_set_name"] == "SciEntsBank_3way")
            ]
        else:
            return all_runs[
                (all_runs["params.model_name"] == model_id)
                & (all_runs["params.dataset_name"] == "SciEntsBank_3way")
                & (all_runs["params.test_set_name"] == "gras_no_logic")
            ]


def extract_metric(runs: pd.DataFrame, metric: str) -> str:
    values = runs[metric].dropna()
    if values.empty:
        return "N/A"
    return fmt(values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0)


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

    rows = []
    for model_id, display_name in MODEL_ORDER:
        row = {"Model": display_name}

        for direction, label in [("d1_to_d2", "D1->D2"), ("d2_to_d1", "D2->D1")]:
            runs = get_runs_for_direction(all_runs, model_id, direction)
            if runs.empty:
                print(f"WARNING: No {label} runs for {display_name}")

            row[f"QWK {label}"] = extract_metric(runs, "metrics.quadratic_weighted_kappa")
            row[f"Macro-F1 {label}"] = extract_metric(runs, "metrics.macro_f1")

        rows.append(row)

    result = pd.DataFrame(rows)
    column_order = [
        "Model",
        "QWK D1->D2",
        "QWK D2->D1",
        "Macro-F1 D1->D2",
        "Macro-F1 D2->D1",
    ]
    result = result[column_order]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Table saved to: {OUTPUT_PATH}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
