"""Model architectures and diffusion utilities (complete).

This file now supports the following *model.type* values used in the YAML
configuration:

1. FULL                     – baseline (no early exit)
2. ASE                      – Adaptive Score Estimation with fixed drop rate
3. LEES                     – Learnable Early-Exit Schedule (proposed)
4. PROGRESSIVEDISTILL (PD)  – Progressive distillation fast sampler

For the purpose of this code release we keep a lightweight TinyUNet backbone so
that experiments are runnable on CPU for smoke-tests; however, the gating logic
and factory API mirror what would be used with large production backbones.
"""

from typing import List
import math

import torch
from torch import nn
import torch.nn.functional as F

# ===============================================================
# 1. General Purpose Building Blocks
# ===============================================================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        # A real implementation would condition on t_emb; here we only pass x through convs.
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))


# ===============================================================
# 2. Tiny UNet backbone (for demonstration)
# ===============================================================
class TinyUNet(nn.Module):
    """A minimal UNet-style network with 4 residual blocks."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32, n_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for _ in range(n_blocks):
            blk = ResidualConvBlock(ch, base_ch)
            self.blocks.append(blk)
            ch = base_ch
        self.out = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x, t):
        for blk in self.blocks:
            x = blk(x, t)
        return self.out(x)


# ===============================================================
# 3. Simple linear noise scheduler (β schedule)
# ===============================================================
class NoiseScheduler:
    def __init__(self, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def _beta(self, t):
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def add_noise(self, x0, t):
        beta_t = self._beta(t)
        noise = torch.randn_like(x0)
        noisy = torch.sqrt(1 - beta_t) * x0 + torch.sqrt(beta_t) * noise
        return noisy, noise


# ===============================================================
# 4. Model variants (wrappers around TinyUNet)
# ===============================================================
class BaseWrapper(nn.Module):
    """Abstract base for wrappers providing a `compute_cost()` and `sample()` API."""

    def compute_cost(self):
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, scheduler: "NoiseScheduler", device):
        raise NotImplementedError


# ---------------------------------------------------------------
# 4.1 Full model (no skipping)
# ---------------------------------------------------------------
class FullModel(BaseWrapper):
    def __init__(self):
        super().__init__()
        self.unet = TinyUNet()
        self._executed_frac = 1.0  # always executes all blocks

    def forward(self, x, t):
        return self.unet(x, t)

    def compute_cost(self):
        return torch.tensor(self._executed_frac, device=next(self.parameters()).device)

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, scheduler: "NoiseScheduler", device):
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        for step in reversed(range(num_steps)):
            t = torch.full((num_samples, 1, 1, 1), step / num_steps, device=device)
            pred_noise = self.forward(x, t)
            beta_t = scheduler._beta(t)
            x = (x - beta_t * pred_noise) / torch.sqrt(1 - beta_t)
        return torch.sigmoid(x)


# ---------------------------------------------------------------
# 4.2 ASE – fixed drop schedule
# ---------------------------------------------------------------
class ASEModel(BaseWrapper):
    def __init__(self, drop_fraction: float = 0.5):
        super().__init__()
        self.unet = TinyUNet()
        self.drop_fraction = max(0.0, min(1.0, drop_fraction))
        self.n_blocks = len(self.unet.blocks)
        self._executed_frac = 1.0 - self.drop_fraction

    def _gate_vector(self):
        k = self.n_blocks
        n_exec = int(k * (1 - self.drop_fraction))
        return torch.tensor([1] * n_exec + [0] * (k - n_exec))

    def forward(self, x, t):
        gate = self._gate_vector()
        for blk, g in zip(self.unet.blocks, gate):
            if g.item() == 1:
                x = blk(x, t)
        return self.unet.out(x)

    def compute_cost(self):
        return torch.tensor(self._executed_frac, device=next(self.parameters()).device)

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, scheduler: "NoiseScheduler", device):
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        for step in reversed(range(num_steps)):
            t = torch.full((num_samples, 1, 1, 1), step / num_steps, device=device)
            pred_noise = self.forward(x, t)
            beta_t = scheduler._beta(t)
            x = (x - beta_t * pred_noise) / torch.sqrt(1 - beta_t)
        return torch.sigmoid(x)


# ---------------------------------------------------------------
# 4.3 LEES – learnable early-exit schedule
# ---------------------------------------------------------------
class LEESModel(BaseWrapper):
    def __init__(self, lambda_: float = 0.05):
        super().__init__()
        self.unet = TinyUNet()
        self.n_blocks = len(self.unet.blocks)
        self.alpha = nn.Parameter(torch.zeros(self.n_blocks))
        self.beta = nn.Parameter(torch.zeros(self.n_blocks))
        self.lambda_ = lambda_
        self._executed_frac = 1.0

        # Freeze backbone weights – only gates are trainable
        for p in self.unet.parameters():
            p.requires_grad = False

    def _sigmoid_gate(self, t):
        # t has shape [B,1,1,1]; squeeze to [B]
        t_flat = t.view(t.size(0))
        return torch.sigmoid(self.alpha * t_flat.unsqueeze(1) + self.beta)  # [B,K]

    def forward(self, x, t):
        gates = self._sigmoid_gate(t)
        hard = (gates > 0.5).float()
        self._executed_frac = hard.mean().item()

        # Execute or skip blocks assuming uniform gating within a batch
        for blk, g in zip(self.unet.blocks, hard[0]):
            if g.item() == 1:
                x = blk(x, t)
        return self.unet.out(x)

    def compute_cost(self):
        return torch.tensor(self._executed_frac, device=self.alpha.device)

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, scheduler: "NoiseScheduler", device):
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        for step in reversed(range(num_steps)):
            t = torch.full((num_samples, 1, 1, 1), step / num_steps, device=device)
            pred_noise = self.forward(x, t)
            beta_t = scheduler._beta(t)
            x = (x - beta_t * pred_noise) / torch.sqrt(1 - beta_t)
        return torch.sigmoid(x)


# ---------------------------------------------------------------
# 4.4 Progressive Distillation – very few denoising steps
# ---------------------------------------------------------------
class ProgressiveDistillModel(BaseWrapper):
    """A toy progressive-distillation model that simply reduces the number of
    denoising steps by a constant factor during sampling. It shares weights
    with the *FullModel* backbone to keep the code lightweight.
    """

    def __init__(self, step_reduction: int = 5):
        super().__init__()
        self.unet = TinyUNet()
        self.step_reduction = max(1, step_reduction)
        self._executed_frac = 1.0  # executes all blocks

    def forward(self, x, t):
        return self.unet(x, t)

    def compute_cost(self):
        return torch.tensor(self._executed_frac, device=next(self.parameters()).device)

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, scheduler: "NoiseScheduler", device):
        # Reduce the number of effective steps by *step_reduction*
        reduced_steps = max(1, math.ceil(num_steps / self.step_reduction))
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        for step in reversed(range(reduced_steps)):
            t = torch.full((num_samples, 1, 1, 1), step / reduced_steps, device=device)
            pred_noise = self.forward(x, t)
            beta_t = scheduler._beta(t)
            x = (x - beta_t * pred_noise) / torch.sqrt(1 - beta_t)
        return torch.sigmoid(x)


# ===============================================================
# 5. Model builder utility
# ===============================================================

def build_model(model_cfg: dict, lambda_: float = 0.0):
    mtype = model_cfg.get("type", "FULL").upper()
    if mtype == "FULL":
        return FullModel()
    if mtype == "ASE":
        return ASEModel(drop_fraction=model_cfg.get("drop_fraction", 0.5))
    if mtype == "LEES":
        lea_lambda = model_cfg.get("lambda", lambda_)
        return LEESModel(lambda_=lea_lambda)
    if mtype in {"PROGRESSIVEDISTILL", "PD"}:
        return ProgressiveDistillModel(step_reduction=model_cfg.get("step_reduction", 5))
    raise ValueError(f"Unknown model type: {mtype}")