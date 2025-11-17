"""Model architectures and diffusion utilities (complete).

This file provides baseline (Full), ASE (manual schedule) and LEES (learnable
schedule) model variants on top of a tiny UNet backbone so that experiments run
quickly while still exercising the full gating logic.  LEES supports *hard* and
*soft* gating via the `gating` flag.
"""

from typing import List

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
        # Tiny demo UNet – ignore t_emb (would normally be added)
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))


# ===============================================================
# 2. Tiny UNet backbone
# ===============================================================
class TinyUNet(nn.Module):
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
# 3. Diffusion noise scheduler (linear β)
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
# 4. Wrapper base-class
# ===============================================================
class BaseWrapper(nn.Module):
    def compute_cost(self):
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, scheduler: "NoiseScheduler", device):
        raise NotImplementedError


# ===============================================================
# 5. Full model (no skipping)
# ===============================================================
class FullModel(BaseWrapper):
    def __init__(self):
        super().__init__()
        self.unet = TinyUNet()
        self._executed_frac = 1.0

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


# ===============================================================
# 6. ASE model (manual schedule)
# ===============================================================
class ASEModel(BaseWrapper):
    def __init__(self, drop_fraction: float = 0.5):
        super().__init__()
        self.unet = TinyUNet()
        self.drop_fraction = float(drop_fraction)
        self.n_blocks = len(self.unet.blocks)

    def _gate_vector(self):
        k = self.n_blocks
        n_exec = int(k * (1 - self.drop_fraction))
        gate = torch.tensor([1] * n_exec + [0] * (k - n_exec), device=next(self.parameters()).device)
        return gate

    def forward(self, x, t):
        gate = self._gate_vector()
        for blk, g in zip(self.unet.blocks, gate):
            if g.item() == 1:
                x = blk(x, t)
        self._executed_frac = gate.float().mean().item()
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


# ===============================================================
# 7. LEES model (learnable schedule)
# ===============================================================
class LEESModel(BaseWrapper):
    def __init__(self, lambda_: float = 0.05, gating: str = "hard"):
        super().__init__()
        assert gating in {"hard", "soft"}, "gating must be 'hard' or 'soft'"
        self.gating = gating
        self.unet = TinyUNet()
        self.n_blocks = len(self.unet.blocks)

        # Learnable parameters α_k and β_k
        self.alpha = nn.Parameter(torch.zeros(self.n_blocks))
        self.beta = nn.Parameter(torch.zeros(self.n_blocks))
        self.lambda_ = float(lambda_)
        self._executed_frac = 1.0  # updated each forward

        # Freeze backbone weights
        for p in self.unet.parameters():
            p.requires_grad = False

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------
    def _sigmoid_gate(self, t):
        # t: [B, 1, 1, 1] – flatten to B×1 then broadcast
        gates = torch.sigmoid(self.alpha * t.view(-1, 1) + self.beta)  # [B, K]
        return gates

    # -----------------------------------------------------------
    # Forward
    # -----------------------------------------------------------
    def forward(self, x, t):
        gates = self._sigmoid_gate(t)  # [B, K]
        if self.gating == "hard":
            mask = (gates > 0.5).float()
        else:  # soft
            mask = gates  # continuous value between 0 and 1

        # Compute executed fraction for the *current* batch
        self._executed_frac = mask.mean().item()

        # Apply blocks according to mask
        # For efficiency we assume all samples in batch share same t ⇒ same gates
        for k, blk in enumerate(self.unet.blocks):
            g_scalar = mask[0, k].item()
            if self.gating == "hard":
                if g_scalar >= 1.0:  # executed
                    x = blk(x, t)
                else:  # skip (identity)
                    pass
            else:  # soft – always execute then interpolate
                out = blk(x, t)
                x = g_scalar * out + (1 - g_scalar) * x
        return self.unet.out(x)

    # -----------------------------------------------------------
    # Metrics helpers
    # -----------------------------------------------------------
    def compute_cost(self):
        return torch.tensor(self._executed_frac, device=self.alpha.device)

    # -----------------------------------------------------------
    # Sampling (DDPM ancestor – very naive but fine for demo)
    # -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, scheduler: "NoiseScheduler", device):
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        for step in reversed(range(num_steps)):
            t = torch.full((num_samples, 1, 1, 1), step / num_steps, device=device)
            pred_noise = self.forward(x, t)
            beta_t = scheduler._beta(t)
            x = (x - beta_t * pred_noise) / torch.sqrt(1 - beta_t)
        return torch.sigmoid(x)


# ===============================================================
# 8. Factory helper
# ===============================================================

def build_model(model_cfg: dict, lambda_: float = 0.0):
    mtype = model_cfg.get("type", "FULL").upper()
    if mtype == "FULL":
        return FullModel()

    if mtype == "ASE":
        drop_frac = float(model_cfg.get("drop_fraction", 0.5))
        return ASEModel(drop_fraction=drop_frac)

    if mtype == "LEES":
        lam = float(model_cfg.get("lambda", lambda_))
        gating = model_cfg.get("gating", "hard").lower()
        return LEESModel(lambda_=lam, gating=gating)

    raise ValueError(f"Unknown model type {mtype}")