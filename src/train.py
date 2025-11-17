import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn, optim
from tqdm import tqdm

from . import preprocess
from . import model as model_lib


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_loop(cfg: Dict[str, Any], results_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ----------------------------------------------------------
    # 1. Data ----------------------------------------------------------------
    # ----------------------------------------------------------
    train_loader = preprocess.get_dataloader(cfg, split="train")
    fid_real_loader = preprocess.get_dataloader(cfg, split="fid")

    # ----------------------------------------------------------
    # 2. Model -------------------------------------------------
    # ----------------------------------------------------------
    model = model_lib.build_model(cfg["model"], cfg["training"].get("lambda", 0.0))
    model.to(device)

    # Only train gating parameters if LEES; otherwise full parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=cfg["training"].get("learning_rate", 1e-4))

    noise_scheduler = model_lib.NoiseScheduler()

    num_epochs = cfg["training"].get("epochs", 1)
    global_step = 0
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_loader:
                x0 = batch.to(device)
                bsz = x0.size(0)

                t = torch.rand(bsz, device=device).view(-1, 1, 1, 1)
                noisy, noise = noise_scheduler.add_noise(x0, t)

                optimizer.zero_grad()
                pred = model(noisy, t)
                recon_loss = nn.functional.mse_loss(pred, noise)
                compute_pen = model.compute_cost()  # mean fraction of executed blocks
                lam = cfg["training"].get("lambda", 0.0)
                loss = recon_loss + lam * compute_pen
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1
                pbar.set_postfix({"loss": loss.item(), "comp": compute_pen.item()})
                pbar.update(1)

        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)

    # ----------------------------------------------------------
    # 3. Sampling + Metrics ------------------------------------
    # ----------------------------------------------------------
    num_samples = cfg.get("generation", {}).get("num_samples", 100)
    num_steps = cfg.get("generation", {}).get("num_steps", 50)

    start_time = time.time()
    generated = model.sample(num_samples=num_samples,
                             num_steps=num_steps,
                             scheduler=noise_scheduler,
                             device=device)
    latency = (time.time() - start_time) / num_samples  # sec / image

    # Move to CPU for metric computation
    generated_cpu = generated.clamp(0, 1).cpu()

    # FID ------------------------------------------------------
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_metric = FrechetInceptionDistance(feature=64).to(device)
        # accumulate real
        for real_batch in fid_real_loader:
            fid_metric.update(real_batch.to(device), real=True)
        # accumulate fake
        fid_metric.update(generated.to(device), real=False)
        fid_score = fid_metric.compute().item()
    except Exception as e:
        print("[WARN] FID computation failed – falling back to dummy metric:", e)
        fid_score = float('nan')

    results = {
        "run_id": cfg["run_id"],
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "fid": fid_score,
        "latency_sec_per_sample": latency,
        "executed_block_fraction": model.compute_cost().item()
    }

    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics" / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also dump samples for qualitative inspection (small subset)
    sample_path = results_dir / "samples"
    sample_path.mkdir(exist_ok=True, parents=True)
    torch.save(generated_cpu[:16], sample_path / "samples.pt")

    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to single run YAML config file for this variation.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to save outputs for this run.")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_results_dir = Path(args.results_dir)
    train_loop(cfg, run_results_dir)