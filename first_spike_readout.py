"""
First-spike R-STDP readout layer experiment.

Trains a K-neuron output layer on top of frozen CSNN spike-count features
using an online reward-modulated Hebbian rule that approximates first-spike R-STDP:

  - pred = argmax(W @ x)            # highest score = "fires first"
  - correct  -> W[y]    += lr * x_norm   (STDP: reinforce correct class)
  - wrong    -> W[pred] -= lr_neg * x_norm  (anti-STDP: weaken wrong class)
               W[y]    += lr * x_norm    (STDP: strengthen correct class)
  - W rows periodically L2-normalised  (weight norm, like in Mozafari 2018)

Usage:
  python first_spike_readout.py --memmap-dir runs_csnn/_memmap_rs02_hybrid9 \
      --epochs 20 --lr 0.01 --lr-neg 0.01 --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_memmap(memmap_dir: str):
    d = Path(memmap_dir)
    Xtr_mm = np.memmap(str(d / "Xtr.memmap"), dtype="float32", mode="r")
    ytr    = np.memmap(str(d / "Xtr.memmap.y.memmap"), dtype="int64", mode="r")
    Xte_mm = np.memmap(str(d / "Xte.memmap"), dtype="float32", mode="r")
    yte    = np.memmap(str(d / "Xte.memmap.y.memmap"), dtype="int64", mode="r")

    n_tr, n_te = len(ytr), len(yte)
    feat = Xtr_mm.size // n_tr
    Xtr = np.array(Xtr_mm.reshape(n_tr, feat), dtype=np.float32)
    Xte = np.array(Xte_mm.reshape(n_te, feat), dtype=np.float32)
    ytr = np.array(ytr, dtype=np.int64)
    yte = np.array(yte, dtype=np.int64)
    return Xtr, ytr, Xte, yte


def row_normalize(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True).clip(min=eps)
    return W / norms


def evaluate(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    scores = X @ W.T          # [N, K]
    preds  = scores.argmax(1) # [N]
    return float((preds == y).mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_first_spike(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    *,
    n_classes: int = 10,
    feat_dim: int | None = None,
    epochs: int = 20,
    lr: float = 0.01,
    lr_neg: float = 0.01,
    norm_every: int = 100,   # normalise weights every N samples
    seed: int = 42,
    verbose: bool = True,
) -> dict:

    rng = np.random.default_rng(seed)

    if feat_dim is None:
        feat_dim = Xtr.shape[1]

    # Use only spike-count part (first 32768 dims) if feature vector is wider
    # The last 10 dims are WTA histogram — skip them for a clean first-spike sim.
    count_dim = feat_dim
    if Xtr.shape[1] > feat_dim:
        Xtr = Xtr[:, :feat_dim]
        Xte = Xte[:, :feat_dim]

    # Normalise feature vectors (L2) once — stable for all epochs
    eps = 1e-8
    Xtr_norm = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + eps)
    Xte_norm = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + eps)

    # Initialise weight matrix small and normalised
    W = rng.standard_normal((n_classes, count_dim)).astype(np.float32) * 0.01
    W = row_normalize(W)

    results = {
        "epochs": [],
        "train_acc": [],
        "test_acc":  [],
        "correct_frac": [],  # rstdp reward quality per epoch
    }

    n_tr = len(ytr)

    for epoch in range(epochs):
        t0 = time.time()
        order = rng.permutation(n_tr)
        correct_count = 0

        for step, idx in enumerate(order):
            x  = Xtr_norm[idx]   # [D]
            y  = int(ytr[idx])

            scores = W @ x        # [K]
            pred   = int(scores.argmax())
            correct = (pred == y)
            correct_count += int(correct)

            if correct:
                # STDP: reinforce correct class neuron
                W[y] += lr * x
            else:
                # anti-STDP: weaken wrong class; strengthen correct
                W[pred] -= lr_neg * x
                W[y]    += lr * x

            # Periodic weight normalisation (keeps weights bounded, like in Mozafari)
            if (step + 1) % norm_every == 0:
                W = row_normalize(W)

        W = row_normalize(W)

        rew_acc  = correct_count / n_tr
        tr_acc   = evaluate(W, Xtr_norm, ytr)
        te_acc   = evaluate(W, Xte_norm, yte)
        elapsed  = time.time() - t0

        results["epochs"].append(epoch + 1)
        results["train_acc"].append(round(tr_acc, 4))
        results["test_acc"].append(round(te_acc, 4))
        results["correct_frac"].append(round(rew_acc, 4))

        if verbose:
            print(
                f"epoch {epoch+1:3d}/{epochs} | "
                f"reward_acc={rew_acc:.3f} | "
                f"train={tr_acc:.3f} | test={te_acc:.3f} | "
                f"{elapsed:.1f}s"
            )

    results["best_test_acc"] = max(results["test_acc"])
    results["best_epoch"]    = int(np.argmax(results["test_acc"])) + 1
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir",  default="runs_csnn/_memmap_rs02_hybrid9")
    ap.add_argument("--epochs",      type=int,   default=20)
    ap.add_argument("--lr",          type=float, default=0.01)
    ap.add_argument("--lr-neg",      type=float, default=0.01)
    ap.add_argument("--norm-every",  type=int,   default=100)
    ap.add_argument("--feat-dim",    type=int,   default=32768,
                    help="Use first N dims of feature vector (spike counts only)")
    ap.add_argument("--n-classes",   type=int,   default=10)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--out-json",    default=None)
    args = ap.parse_args()

    print(f"Loading memmap from {args.memmap_dir} ...")
    Xtr, ytr, Xte, yte = load_memmap(args.memmap_dir)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")
    print(f"  Feature dim used: {args.feat_dim}")
    print(f"  lr={args.lr}  lr_neg={args.lr_neg}  epochs={args.epochs}  seed={args.seed}")
    print()

    results = train_first_spike(
        Xtr, ytr, Xte, yte,
        n_classes=args.n_classes,
        feat_dim=args.feat_dim,
        epochs=args.epochs,
        lr=args.lr,
        lr_neg=args.lr_neg,
        norm_every=args.norm_every,
        seed=args.seed,
    )

    print()
    print(f"Best test acc: {results['best_test_acc']:.4f}  (epoch {results['best_epoch']})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
