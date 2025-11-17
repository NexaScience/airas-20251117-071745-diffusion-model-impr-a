import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FIG_DIR_NAME = "images"


def gather_results(results_root: Path) -> List[Dict]:
    all_results = []
    for run_dir in results_root.iterdir():
        metrics_path = run_dir / "metrics" / "results.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                res = json.load(f)
                res["run_dir"] = str(run_dir)
                all_results.append(res)
    return all_results


def plot_bar(df: pd.DataFrame, metric: str, results_dir: Path):
    plt.figure(figsize=(8, 4))
    sns.barplot(x="run_id", y=metric, data=df)
    for i, v in enumerate(df[metric]):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.title(metric)
    plt.xlabel("Run ID")
    plt.ylabel(metric)
    plt.tight_layout()
    fname = f"{metric}.pdf"
    (results_dir / FIG_DIR_NAME).mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / FIG_DIR_NAME / fname, bbox_inches="tight")
    plt.close()
    return fname


def plot_training_loss(df: pd.DataFrame, results_dir: Path):
    plt.figure(figsize=(8, 4))
    for _, row in df.iterrows():
        losses = row["epoch_losses"]
        plt.plot(range(1, len(losses) + 1), losses, label=row["run_id"])
        plt.annotate(f"{losses[-1]:.2f}", (len(losses), losses[-1]))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    fname = "training_loss.pdf"
    (results_dir / FIG_DIR_NAME).mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / FIG_DIR_NAME / fname, bbox_inches="tight")
    plt.close()
    return fname


def evaluate(results_root: Path):
    res = gather_results(results_root)
    if not res:
        raise RuntimeError(f"No results found in {results_root}")

    df = pd.DataFrame(res)

    # Output summary to stdout
    summary = df[["run_id", "fid", "latency_sec_per_sample", "executed_block_fraction"]].to_dict(orient="records")
    print(json.dumps({"comparison": summary}, indent=2))

    # Generate figures
    fig_names = []
    fig_names.append(plot_bar(df, "fid", results_root))
    fig_names.append(plot_bar(df, "latency_sec_per_sample", results_root))
    fig_names.append(plot_bar(df, "executed_block_fraction", results_root))
    fig_names.append(plot_training_loss(df, results_root))

    return fig_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    fig_files = evaluate(results_root)
    print(json.dumps({"figures": fig_files}))