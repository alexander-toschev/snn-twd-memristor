"""Memristor device model for weight update simulation.

Linear (write-noise) model:
    ΔG = ΔW · (1 + N(0, σ))   — multiplicative cycle-to-cycle noise
    G  ← clamp(G, g_min, g_max)  — device conductance limits
    G  ← quantize(G, n_bits)     — finite precision (optional)

Usage::

    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = MemristorWrapper(base_opt, sigma=0.1, g_min=-0.5, g_max=0.5, n_bits=8)

    # training loop:
    opt.zero_grad()
    loss.backward()
    opt.step()   # Adam update + noise injection + clamp + quantize

References:
    Gokmen & Vlasov (2016) Front. Neurosci.
    Boybat et al. (2018) Nature Comms.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class MemristorWrapper:
    """Wraps any PyTorch optimizer with a linear memristor device model.

    After each optimizer step the weight *delta* is corrupted by multiplicative
    Gaussian noise and then the resulting weight is clamped to [g_min, g_max]
    and optionally quantised to n_bits levels.

    Args:
        optimizer: base PyTorch optimizer (e.g. Adam, SGD).
        sigma: std of multiplicative write noise (0 = ideal, no noise).
        g_min: minimum conductance (lower clamp for weights).
        g_max: maximum conductance (upper clamp for weights).
        n_bits: number of conductance levels (None = no quantisation).
        apply_to_bias: whether to apply noise to bias parameters too.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        sigma: float = 0.0,
        g_min: float = -1.0,
        g_max: float = 1.0,
        n_bits: Optional[int] = None,
        apply_to_bias: bool = False,
    ):
        if g_min >= g_max:
            raise ValueError(f"g_min ({g_min}) must be < g_max ({g_max})")
        self.optimizer = optimizer
        self.sigma = sigma
        self.g_min = g_min
        self.g_max = g_max
        self.n_bits = n_bits
        self.apply_to_bias = apply_to_bias

        # pre-compute quantisation step size
        if n_bits is not None:
            self._n_levels = 2 ** n_bits - 1  # number of intervals
        else:
            self._n_levels = None

    # ── delegation helpers ────────────────────────────────────────────────────

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, closure=None):
        """Run base optimizer step then apply memristor device model."""
        # Snapshot weights before update
        snapshots: dict[int, torch.Tensor] = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self.apply_to_bias and p.dim() == 1:
                    continue  # skip bias vectors
                snapshots[id(p)] = p.data.clone()

        # Base optimizer update (Adam / SGD / …)
        loss = self.optimizer.step(closure)

        # Apply device model to each updated parameter
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) not in snapshots:
                    continue
                old = snapshots[id(p)]
                delta = p.data - old          # ideal weight update ΔW

                # 1. Write noise: ΔG = ΔW · (1 + N(0, σ))
                if self.sigma > 0.0:
                    noise = torch.randn_like(delta).mul_(self.sigma)
                    delta = delta * (1.0 + noise)

                # 2. Apply noisy update
                p.data = old + delta

                # 3. Conductance limits
                p.data.clamp_(self.g_min, self.g_max)

                # 4. Quantisation
                if self._n_levels is not None:
                    self._quantize_inplace(p.data)

        return loss

    def _quantize_inplace(self, w: torch.Tensor) -> None:
        """Round weights to nearest conductance level in [g_min, g_max]."""
        span = self.g_max - self.g_min
        # Normalise to [0, 1], round to n_levels, de-normalise
        w.sub_(self.g_min).div_(span)           # → [0, 1]
        w.mul_(self._n_levels).round_()         # → {0, …, n_levels}
        w.div_(self._n_levels).mul_(span).add_(self.g_min)  # → [g_min, g_max]


def make_optimizer(cfg, params) -> MemristorWrapper | torch.optim.Optimizer:
    """Return MemristorWrapper(Adam) when memristor is configured, else plain Adam."""
    base = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if not getattr(cfg, "memristor_sigma", 0.0) and not getattr(cfg, "memristor_bits", None):
        return base  # no memristor — plain Adam

    return MemristorWrapper(
        base,
        sigma=getattr(cfg, "memristor_sigma", 0.0),
        g_min=getattr(cfg, "memristor_g_min", -1.0),
        g_max=getattr(cfg, "memristor_g_max", 1.0),
        n_bits=getattr(cfg, "memristor_bits", None),
    )
