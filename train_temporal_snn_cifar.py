#!/usr/bin/env python3
"""Train a temporal SNN (first-spike coding) on CIFAR100:20 from scratch.

This is a completely NEW architecture separate from the rate-coded CSNN.

Key differences from old CSNN:
  - Input encoding : latency (bright=early spike, dark=late) vs Poisson rate
  - Training       : surrogate-gradient BPTT end-to-end vs greedy STDP + frozen readout
  - Decoding       : first-spike (argmin firing time) vs spike counts
  - T              : 16 timesteps vs 200-300 (much shorter, info is in timing not count)

Usage:
  # Default config (TET loss, T=16, 120 epochs, BPTT)
  python train_temporal_snn_cifar.py

  # E-prop training
  python train_temporal_snn_cifar.py --use-eprop --eprop-kappa 0.8

  # Quick smoke test (BPTT)
  python train_temporal_snn_cifar.py --epochs 5 --batch-size 32 --run-id smoke_test

  # Quick smoke test (e-prop)
  python train_temporal_snn_cifar.py --use-eprop --epochs 3 --batch-size 32 --run-id eprop_smoke_test
"""
import argparse
import json
import sys
import os
from dataclasses import fields

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from temporal_snn.config import TemporalSNNCfg


def parse_args():
    p = argparse.ArgumentParser(
        description="Train temporal SNN with first-spike coding on CIFAR100:20"
    )
    p.add_argument("--config", default=None,
                   help="Path to a JSON config. Explicit CLI options override JSON values.")
    # Dataset
    p.add_argument("--dataset", default="cifar100:0,1,2,3,5,8,13,14,17,19",
                   help="Dataset spec, e.g. 'cifar100:0,1,2,3,5,8,13,14,17,19'")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-classes", type=int, default=10)

    # Temporal encoding
    p.add_argument("--T", type=int, default=16, help="Timesteps for latency encoding")
    p.add_argument("--x-min", type=float, default=0.1,
                   help="Pixels below x_min stay silent (background suppression)")

    # Architecture
    p.add_argument("--c1-out", type=int, default=32)
    p.add_argument("--c2-out", type=int, default=64)
    p.add_argument("--fc1-out", type=int, default=256)

    # LIF
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--v-threshold", type=float, default=1.0)
    p.add_argument("--surrogate-alpha", type=float, default=2.0)

    # Training
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--dropout-fc", type=float, default=0.3)

    # Loss
    p.add_argument("--loss-mode", default="tet", choices=["tet", "last", "first_spike"],
                   help=(
                       "tet: TET (CE at every timestep) | "
                       "last: CE at last timestep | "
                       "first_spike: TET + first-spike ranking loss"
                   ))
    p.add_argument("--fsl-weight", type=float, default=0.5,
                   help="Weight of first-spike ranking loss (when loss_mode=first_spike)")
    p.add_argument("--tet-weight", type=float, default=1.0)

    # E-prop
    p.add_argument("--use-eprop", action="store_true",
                   help="Use e-prop (eligibility propagation) instead of BPTT")
    p.add_argument("--eprop-kappa", type=float, default=0.8,
                   help="E-prop eligibility trace decay (0 = no filtering)")
    p.add_argument("--eprop-rate-loss", action="store_true",
                   help="L_out from spike counts (rate loss) instead of membrane potential")
    p.add_argument("--eprop-feedback", default="symmetric",
                   choices=["symmetric", "random"],
                   help=(
                       "symmetric: use W^T for learning signal (exact e-prop) | "
                       "random: use fixed random B (DFA-style, no weight transport)"
                   ))

    # Recurrent connection
    # ALIF
    p.add_argument("--use-alif", action="store_true",
                   help="Use ALIF neurons in FC1->LIF3 (adaptive threshold, Bellec 2020)")
    p.add_argument("--alif-rho", type=float, default=0.96,
                   help="ALIF adaptation decay (default 0.96)")
    p.add_argument("--alif-beta", type=float, default=0.07,
                   help="ALIF threshold coupling strength (default 0.07)")

    p.add_argument("--use-recurrent", action="store_true",
                   help="Add lateral W_rec connection to FC1→LIF3 (requires --use-eprop)")
    p.add_argument("--rec-init-scale", type=float, default=0.1,
                   help="W_rec init: kaiming_normal * rec_init_scale (default 0.1)")

    p.add_argument("--n-conv-layers", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Number of convolutional layers: 1, 2 (default), 3, or 4")
    p.add_argument("--c3-out", type=int, default=128,
                   help="Conv3 output channels (only used when --n-conv-layers>=3, default 128)")
    p.add_argument("--c4-out", type=int, default=256,
                   help="Conv4 output channels (only used when --n-conv-layers=4, default 256)")

    # Memristor device model
    p.add_argument("--memristor-sigma", type=float, default=0.0,
                   help="Write-noise std (multiplicative). 0=disabled (default).")
    p.add_argument("--memristor-g-min", type=float, default=-1.0,
                   help="Min conductance / weight clamp (default -1.0)")
    p.add_argument("--memristor-g-max", type=float, default=1.0,
                   help="Max conductance / weight clamp (default 1.0)")
    p.add_argument("--memristor-bits", type=int, default=None,
                   help="Precision bits for quantisation (None=float32)")

    # TWD decoder
    p.add_argument("--use-twd", action="store_true", default=False,
                   help="Enable Temporal Weighting Decoder (learned α[t] weights)")
    p.add_argument("--twd-weight", type=float, default=1.0,
                   help="Weight of TWD CE loss (combined with TET, default 1.0)")

    # ETTFS-init
    p.add_argument("--ettfs-init", action="store_true", default=False,
                   help="ETTFS-init: scale weights by sqrt(T) to prevent signal diminishing")

    # BNTT
    p.add_argument("--use-bntt", action="store_true", default=False,
                   help="BatchNorm Through Time: separate BN per timestep (recommended with TWD)")

    # GAP
    p.add_argument("--use-gap", action="store_true", default=False,
                   help="Global Average Pooling instead of Flatten before FC1 (-98%% crossbar cells)")

    # Weight Perturbation / SPSA
    p.add_argument("--use-wp", action="store_true", default=False,
                   help="Weight Perturbation / SPSA (hardware-native, no backprop)")
    p.add_argument("--wp-sigma", type=float, default=0.05,
                   help="Perturbation std (c_k).  Default 0.05.")
    p.add_argument("--wp-sigma-decay", type=float, default=1.0,
                   help="Per-epoch sigma multiplier (1.0=no decay, 0.99=slow decay)")
    p.add_argument("--wp-sigma-min", type=float, default=0.001,
                   help="Floor for sigma after decay")

    # Greedy layer-wise
    p.add_argument("--use-greedy", action="store_true", default=False,
                   help="Greedy layer-wise training (Predictive Coding style)")
    p.add_argument("--greedy-epochs", type=int, default=60,
                   help="Epochs per conv stage in greedy training")
    p.add_argument("--finetune-epochs", type=int, default=100,
                   help="FC fine-tune epochs after greedy conv stages")
    p.add_argument("--greedy-full-finetune", action="store_true", default=False,
                   help="Unfreeze all layers for finetune (end-to-end from greedy init)")
    p.add_argument("--finetune-lr", type=float, default=-1.0,
                   help="LR for finetune phase. -1=same as --lr. For full-finetune use ~0.1x of lr.")
    p.add_argument("--fc-warmup-epochs", type=int, default=0,
                   help="Phase-1: FC-only warm-up epochs before full finetune (0=skip)")

    # Output neuron threshold
    p.add_argument("--v-threshold-out", type=float, default=-1.0,
                   help="Output LIF threshold (-1 = same as v_threshold). Lower = more spikes.")

    # WTA lateral inhibition
    p.add_argument("--wta-enable", action="store_true", default=False,
                   help="Enable hard WTA in conv layers (per-position channel competition)")
    p.add_argument("--wta-layers", type=str, default="conv1",
                   help="Which layers get WTA: conv1, conv2, all, or comma-list (default: conv1)")

    # Output
    p.add_argument("--outdir", default="runs_csnn",
                   help="Output root dir. Default runs_csnn so cron_exp_status.py sees it.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--resume-checkpoint", default=None,
                   help="Path to a .pt checkpoint (state_dict) to warm-start weights from.")
    p.add_argument("--start-epoch", type=int, default=1,
                   help="First epoch to run (1=fresh). Set >1 when resuming mid-training.")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)

    config_probe, _ = p.parse_known_args()
    if config_probe.config:
        with open(config_probe.config, "r", encoding="utf-8") as f:
            config_defaults = json.load(f)
        valid_fields = {field.name for field in fields(TemporalSNNCfg)}
        unknown = sorted(set(config_defaults) - valid_fields)
        if unknown:
            p.error(f"unknown config keys in {config_probe.config}: {', '.join(unknown)}")
        p.set_defaults(**config_defaults)

    return p.parse_args()


def main():
    args = parse_args()
    config_fields = {field.name for field in fields(TemporalSNNCfg)}
    cfg = TemporalSNNCfg(**{
        key: value for key, value in vars(args).items() if key in config_fields
    })

    if cfg.use_greedy:
        from temporal_snn.greedy_trainer import train as greedy_train
        greedy_train(cfg)
        return

    if getattr(cfg, 'use_wp', False):
        from temporal_snn.wp_trainer import train as wp_train
        wp_train(cfg)
        return

    if cfg.use_eprop:
        from temporal_snn.eprop_trainer import train as eprop_train
        eprop_train(cfg)
    else:
        from temporal_snn.trainer import train
        train(cfg)


if __name__ == "__main__":
    main()
