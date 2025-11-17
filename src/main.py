"""Experiment orchestrator.

This script sequentially launches each run variation defined in a YAML config
file (smoke_test.yaml or full_experiment.yaml).  Each run is executed via a
sub-process to ensure a clean state, with stdout/stderr being tee-ed to both
terminal and disk.
"""

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def tee_stream(stream, log_file_path):
    """Forward bytes from a stream to both stdout/stderr and a log file."""
    with open(log_file_path, "wb") as log_f:
        for line in iter(stream.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            log_f.write(line)
            log_f.flush()
    stream.close()


def run_subprocess(cmd: List[str], run_dir: Path):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout_thread = threading.Thread(
        target=tee_stream, args=(proc.stdout, run_dir / "stdout.log"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=tee_stream, args=(proc.stderr, run_dir / "stderr.log"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    proc.wait()
    stdout_thread.join()
    stderr_thread.join()

    if proc.returncode != 0:
        raise RuntimeError(f"Sub-process failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main(cli_args):
    if cli_args.smoke_test == cli_args.full_experiment:
        raise ValueError("Specify exactly one of --smoke-test or --full-experiment")

    cfg_file = CONFIG_DIR / ("smoke_test.yaml" if cli_args.smoke_test else "full_experiment.yaml")
    with open(cfg_file, "r") as f:
        exp_cfg = yaml.safe_load(f)

    experiments = exp_cfg.get("experiments", [])
    if not experiments:
        raise RuntimeError("No experiments found in config file – please populate it.")

    results_root = Path(cli_args.results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        run_id = exp["run_id"]
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save single-run config to disk (train.py will read it)
        single_cfg_path = run_dir / "config.yaml"
        with open(single_cfg_path, "w") as f:
            yaml.dump(exp, f)

        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(single_cfg_path),
            "--results-dir",
            str(run_dir),
        ]
        print(f"\n=== Launching run {run_id} ===")
        run_subprocess(cmd, run_dir)
        print(f"=== Completed run {run_id} ===\n")

    # After all runs -> evaluation
    print("\nAll runs complete. Starting evaluation…\n")
    eval_cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--results-dir",
        str(results_root),
    ]
    run_subprocess(eval_cmd, results_root)
    print("Evaluation finished. Figures saved to", results_root / "images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-experiment", action="store_true")
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    main(args)