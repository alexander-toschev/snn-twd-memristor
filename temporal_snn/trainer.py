"""Training and evaluation loop for TemporalCSNN."""
from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TemporalSNNCfg
from .model import TemporalCSNN
from .dataset import LatencyEncoder, make_dataloaders
from .loss import combined_loss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_scheduler(optimizer, cfg: TemporalSNNCfg, steps_per_epoch: int):
    if cfg.lr_scheduler == "cosine":
        # Warmup + cosine decay via LambdaLR
        T_max = cfg.epochs * steps_per_epoch
        warmup = cfg.warmup_epochs * steps_per_epoch
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, T_max - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size * steps_per_epoch, gamma=cfg.lr_gamma
        )
    else:
        return None  # no scheduler


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model: TemporalCSNN,
    loader: DataLoader,
    encoder: LatencyEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: TemporalSNNCfg,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct_mem = 0
    correct_fs  = 0
    correct_twd = 0
    total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)    # [B, C, H, W]
        labels = labels.to(device, non_blocking=True)  # [B]

        # Latency encode on GPU
        spikes_in = encoder(imgs)                      # [T, B, C, H, W]

        optimizer.zero_grad()
        fwd = model(spikes_in)
        if len(fwd) == 3:
            membranes, spikes_out, twd_score = fwd
        else:
            membranes, spikes_out = fwd
            twd_score = None

        loss = combined_loss(
            membranes, spikes_out, labels,
            loss_mode=cfg.loss_mode,
            tet_weight=cfg.tet_weight,
            fsl_weight=cfg.fsl_weight,
        )
        # Add TWD CE loss when decoder is active
        if twd_score is not None:
            import torch.nn.functional as _F
            twd_weight = getattr(cfg, 'twd_weight', 1.0)
            loss = loss + twd_weight * _F.cross_entropy(twd_score, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        B = labels.size(0)
        total_loss += loss.item() * B
        total      += B

        with torch.no_grad():
            pred_mem = membranes.sum(0).argmax(1)
            correct_mem += (pred_mem == labels).sum().item()
            pred_fs = TemporalCSNN.first_spike_pred(spikes_out.detach())
            correct_fs  += (pred_fs == labels).sum().item()
            if twd_score is not None:
                correct_twd += (twd_score.detach().argmax(1) == labels).sum().item()

    return {
        "loss":      total_loss / total,
        "acc_mem":   correct_mem / total,
        "acc_fs":    correct_fs  / total,
        "acc_twd":   correct_twd / total if correct_twd > 0 else None,
    }


@torch.no_grad()
def _eval_epoch(
    model: TemporalCSNN,
    loader: DataLoader,
    encoder: LatencyEncoder,
    cfg: TemporalSNNCfg,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    correct_mem = 0
    correct_fs  = 0
    correct_twd = 0
    total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        spikes_in = encoder(imgs)
        fwd = model(spikes_in)
        if len(fwd) == 3:
            membranes, spikes_out, twd_score = fwd
        else:
            membranes, spikes_out = fwd
            twd_score = None

        pred_mem = membranes.sum(0).argmax(1)
        pred_fs  = TemporalCSNN.first_spike_pred(spikes_out)

        correct_mem += (pred_mem == labels).sum().item()
        correct_fs  += (pred_fs  == labels).sum().item()
        if twd_score is not None:
            correct_twd += (twd_score.argmax(1) == labels).sum().item()
        total       += labels.size(0)

    return {
        "acc_mem": correct_mem / total,
        "acc_fs":  correct_fs  / total,
        "acc_twd": correct_twd / total if correct_twd > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training runner
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TemporalSNNCfg) -> Dict:
    _set_seed(cfg.seed)

    # ── Resolve run_id and outdir ─────────────────────────────────────────
    import datetime
    if cfg.run_id is None:
        # Must match cron_exp_status.py regex: ^20\d{6}T\d{6}Z_[0-9a-f]{6,}$
        import hashlib
        ts  = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        key = f"temporal_T{cfg.T}_{cfg.loss_mode}_seed{cfg.seed}"
        h   = hashlib.md5(key.encode()).hexdigest()[:12]
        cfg.run_id = f"{ts}_{h}"

    run_dir = Path(cfg.outdir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path  = run_dir / "train.log"
    ckpt_dir  = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── Save config ───────────────────────────────────────────────────────
    import dataclasses
    with open(run_dir / "cfg.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # ── Device & encoder ─────────────────────────────────────────────────
    device  = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    encoder = LatencyEncoder(T=cfg.T, x_min=cfg.x_min)

    print(f"[temporal_snn] Run: {cfg.run_id}")
    print(f"[temporal_snn] Device: {device}")
    print(f"[temporal_snn] Dataset: {cfg.dataset}")
    print(f"[temporal_snn] T={cfg.T}  loss_mode={cfg.loss_mode}  epochs={cfg.epochs}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    tr_dl, te_dl = make_dataloaders(
        cfg.dataset, cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and (device.type == "cuda"),
    )
    print(f"[temporal_snn] Train: {len(tr_dl.dataset)} samples  Test: {len(te_dl.dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TemporalCSNN(
        in_channels=3, n_classes=cfg.n_classes,
        c1_out=cfg.c1_out, c2_out=cfg.c2_out, fc1_out=cfg.fc1_out,
        tau=cfg.tau, v_threshold=cfg.v_threshold,
        surrogate_alpha=cfg.surrogate_alpha,
        dropout_fc=cfg.dropout_fc,
        n_conv_layers=getattr(cfg, 'n_conv_layers', 2),
        c3_out=getattr(cfg, 'c3_out', 128),
        wta_enable=getattr(cfg, 'wta_enable', False),
        wta_layers=getattr(cfg, 'wta_layers', 'conv1'),
        v_threshold_out=getattr(cfg, 'v_threshold_out', -1.0),
        ettfs_init=getattr(cfg, 'ettfs_init', False),
        T_for_init=cfg.T,
        use_twd=getattr(cfg, 'use_twd', False),
        use_bntt=getattr(cfg, 'use_bntt', False),
        use_gap=getattr(cfg, 'use_gap', False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[temporal_snn] Parameters: {n_params:,}")

    # ── Resume: load checkpoint weights ─────────────────────────────────────
    start_epoch = getattr(cfg, 'start_epoch', 1)
    resume_ckpt = getattr(cfg, 'resume_checkpoint', None)
    if resume_ckpt:
        print(f"[temporal_snn] Resuming from checkpoint: {resume_ckpt}")
        state = torch.load(resume_ckpt, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[temporal_snn]   Missing keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"[temporal_snn]   Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
        print(f"[temporal_snn] Checkpoint loaded. Starting from epoch {start_epoch}.")

        # ── Optimiser & scheduler ─────────────────────────────────────────────
    from .memristor import make_optimizer as _make_opt
    optimizer = _make_opt(cfg, model.parameters())
    _sched_opt = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer
    scheduler = _make_scheduler(_sched_opt, cfg, steps_per_epoch=len(tr_dl))

    # Fast-forward scheduler to correct position when resuming mid-training.
    if start_epoch > 1 and scheduler is not None:
        steps_done = (start_epoch - 1) * len(tr_dl)
        print(f"[temporal_snn] Fast-forwarding scheduler by {steps_done} steps (epoch {start_epoch-1} done).")
        for _ in range(steps_done):
            scheduler.step()

    # ── status.json (cron-compatible) ──────────────────────────────────────
    status_path = run_dir / "status.json"
    created_at  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pid         = os.getpid()

    def _write_status(status: str, epoch: int = 0):
        obj = {
            "status":     status,
            "stage":      f"epoch {epoch}/{cfg.epochs}" if epoch else "init",
            "pct":        round(100.0 * epoch / cfg.epochs, 1) if epoch else 0.0,
            "i":          epoch,
            "n":          cfg.epochs,
            "pid":        pid,
            "created_at": created_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cfg": {
                "seed":      cfg.seed,
                "time":      cfg.T,
                "loss_mode": cfg.loss_mode,
                "dataset":   cfg.dataset,
            },
        }
        with open(status_path, "w") as f:
            json.dump(obj, f, indent=2)

    _write_status("running", epoch=0)

    # ── Training loop ─────────────────────────────────────────────────────
    history = []
    best_acc_fs  = 0.0
    best_acc_mem = 0.0
    best_acc_twd = 0.0
    best_epoch   = 0

    # Recover best-so-far from existing metrics.jsonl when resuming.
    if start_epoch > 1:
        _metrics_path = run_dir / "metrics.jsonl"
        if _metrics_path.exists():
            try:
                with open(_metrics_path) as _mf:
                    for _line in _mf:
                        _line = _line.strip()
                        if not _line:
                            continue
                        _rec = json.loads(_line)
                        _ep  = _rec.get("epoch", 0)
                        _fs  = _rec.get("test_acc_fs", 0.0) or 0.0
                        _mem = _rec.get("test_acc_mem", 0.0) or 0.0
                        _twd = _rec.get("test_acc_twd", 0.0) or 0.0
                        if _fs > best_acc_fs:
                            best_acc_fs = _fs
                            best_epoch  = _ep
                        if _mem > best_acc_mem:
                            best_acc_mem = _mem
                        if _twd > best_acc_twd:
                            best_acc_twd = _twd
                print(f"[temporal_snn] Prior best: fs={best_acc_fs:.4f}@ep{best_epoch}  mem={best_acc_mem:.4f}")
            except Exception as _e:
                print(f"[temporal_snn] Warning: could not load prior metrics: {_e}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        tr_stats = _train_epoch(model, tr_dl, encoder, optimizer, scheduler, cfg, device)
        te_stats = _eval_epoch(model, te_dl, encoder, cfg, device)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        record = {
            "epoch":       epoch,
            "lr":          round(lr_now, 8),
            "train_loss":  round(tr_stats["loss"], 6),
            "train_acc_mem": round(tr_stats["acc_mem"], 4),
            "train_acc_fs":  round(tr_stats["acc_fs"],  4),
            "test_acc_mem":  round(te_stats["acc_mem"],  4),
            "test_acc_fs":   round(te_stats["acc_fs"],   4),
            "elapsed_s":   round(elapsed, 1),
        }
        history.append(record)

        # Track best
        if te_stats["acc_fs"] > best_acc_fs:
            best_acc_fs  = te_stats["acc_fs"]
            best_epoch   = epoch
            torch.save(model.state_dict(), run_dir / "best_fs.pt")
        if te_stats["acc_mem"] > best_acc_mem:
            best_acc_mem = te_stats["acc_mem"]
            torch.save(model.state_dict(), run_dir / "best_mem.pt")
        if te_stats.get("acc_twd") and te_stats["acc_twd"] > best_acc_twd:
            best_acc_twd = te_stats["acc_twd"]
            torch.save(model.state_dict(), run_dir / "best_twd.pt")

        twd_str = f" te_twd={te_stats['acc_twd']:.3f}" if te_stats.get('acc_twd') else ""
        line = (
            f"epoch {epoch:3d}/{cfg.epochs} | loss={tr_stats['loss']:.4f} | "
            f"tr_mem={tr_stats['acc_mem']:.3f} tr_fs={tr_stats['acc_fs']:.3f} | "
            f"te_mem={te_stats['acc_mem']:.3f} te_fs={te_stats['acc_fs']:.3f}{twd_str} | "
            f"best_fs={best_acc_fs:.3f}@{best_epoch} | {elapsed:.1f}s"
        )
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        # Append to metrics.jsonl (compatible with cron status checker)
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        # Periodic checkpoint
        if epoch % cfg.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")

        # Update status.json after every epoch (cron reads this)
        _write_status("running", epoch=epoch)

    # ── Save final summary ────────────────────────────────────────────────
    summary = {
        "run_id":       cfg.run_id,
        "dataset":      cfg.dataset,
        "T":            cfg.T,
        "loss_mode":    cfg.loss_mode,
        "best_acc_fs":  round(best_acc_fs,  4),
        "best_acc_mem": round(best_acc_mem, 4),
        "best_acc_twd": round(best_acc_twd, 4) if best_acc_twd > 0 else None,
        "best_epoch":   best_epoch,
        "epochs":       cfg.epochs,
        "finished_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Mark terminal status
    _write_status("ok", epoch=cfg.epochs)

    print(f"\n[temporal_snn] Done.  Best first-spike acc: {best_acc_fs:.4f}  (epoch {best_epoch})")
    print(f"[temporal_snn] Best membrane acc:          {best_acc_mem:.4f}")
    print(f"[temporal_snn] Results: {run_dir}")

    return summary
