"""Loss functions for temporal SNN training.

TET Loss (Temporal Efficient Training)
---------------------------------------
Apply cross-entropy on the output membrane potential at EVERY timestep,
then average.  This ensures gradients reach early timesteps directly
(avoids vanishing gradient through the decay chain).

  L_TET = (1/T) * sum_t CE(v_out[t], y)

where v_out[t] is the membrane of output neurons at timestep t.

First-Spike Loss (optional)
----------------------------
Ranking loss: correct class should fire BEFORE all incorrect classes.
For each pair (correct class k*, wrong class k), penalise if k fires
earlier than k*.

  L_fsl = mean_{b} mean_{k≠k*} max(0, t_correct[b] - t_wrong[b,k] + margin)

We mix TET + FSL:  L = w_tet * L_TET + w_fsl * L_fsl
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def tet_loss(membranes: Tensor, targets: Tensor) -> Tensor:
    """Temporal Efficient Training loss.

    membranes : [T, B, K]  output membrane potentials
    targets   : [B]        integer class labels
    Returns scalar loss.
    """
    T, B, K = membranes.shape
    # Reshape to [T*B, K] for F.cross_entropy, repeat targets T times
    mem_flat = membranes.reshape(T * B, K)
    tgt_flat = targets.repeat(T)                     # [T*B]
    return F.cross_entropy(mem_flat, tgt_flat)


def last_step_loss(membranes: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy on only the LAST timestep membrane."""
    return F.cross_entropy(membranes[-1], targets)


def first_spike_loss(
    spikes_out: Tensor,
    membranes: Tensor,
    targets: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """Ranking loss: correct class should spike before wrong classes.

    spikes_out : [T, B, K]  output spikes (0/1 via surrogate)
    membranes  : [T, B, K]  output membranes (for TET term)
    targets    : [B]

    Returns first-spike ranking loss (scalar).
    """
    T, B, K = spikes_out.shape
    dev = spikes_out.device

    # Soft first-spike time approximation (differentiable through surrogate):
    # t_soft[b,k] = sum_t t * spike[t,b,k] / (sum_t spike[t,b,k] + eps)
    # This is a weighted average of spike times.
    t_idx = torch.arange(T, device=dev, dtype=torch.float32)  # [T]
    t_mat = t_idx[:, None, None].expand(T, B, K)               # [T, B, K]

    spike_sum = spikes_out.sum(0).clamp(min=1e-6)             # [B, K]
    # Soft first spike: weighted mean time of spikes
    t_mean = (t_mat * spikes_out).sum(0) / spike_sum          # [B, K]
    # Neurons that never fired get rank T
    never_fired = (spike_sum < 0.5)
    t_mean = torch.where(never_fired, torch.full_like(t_mean, float(T)), t_mean)

    # Extract correct class times and wrong class times
    one_hot = F.one_hot(targets, K).bool()                    # [B, K]
    t_correct = t_mean[one_hot].unsqueeze(1).expand(B, K)     # [B, K]
    t_wrong   = t_mean                                         # [B, K]

    # Mask out correct class
    wrong_mask = ~one_hot                                      # [B, K]

    # Ranking loss: we want t_correct < t_wrong, so penalise t_correct - t_wrong + margin
    raw = t_correct - t_wrong + margin                         # [B, K]
    rank_loss = raw.clamp(min=0.0)[wrong_mask].mean()

    return rank_loss


def combined_loss(
    membranes: Tensor,
    spikes_out: Tensor,
    targets: Tensor,
    loss_mode: str = "tet",
    tet_weight: float = 1.0,
    fsl_weight: float = 0.5,
    margin: float = 1.0,
) -> Tensor:
    """Compute training loss according to loss_mode.

    loss_mode:
      "tet"         – TET loss only
      "last"        – CE at last timestep only
      "first_spike" – TET + first-spike ranking loss
    """
    if loss_mode == "tet":
        return tet_loss(membranes, targets)
    elif loss_mode == "last":
        return last_step_loss(membranes, targets)
    elif loss_mode == "first_spike":
        l_tet = tet_loss(membranes, targets)
        l_fsl = first_spike_loss(spikes_out, membranes, targets, margin=margin)
        return tet_weight * l_tet + fsl_weight * l_fsl
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode!r}")
