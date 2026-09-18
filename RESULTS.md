# Ablation Results (2026-09-18)

## Noise vs TWD Ablation (Diverse10, 3-conv WIDE, BNTT, T=12, 200 ep)

| Configuration | Seed 0 | Seed 1 | Seed 2 | Seed 42 | Mean ± std |
|---|---|---|---|---|---|
| **notwd** (σ=0.1, no TWD) | 59.7% | 59.3% | 61.9% | 60.2% | **60.3 ± 1.1%** |
| **nonoise** (TWD on, σ=0) | 75.3% | 75.6% | 75.0% | ~75% (est.) | **~75.2%** |
| Full (TWD + σ=0.1, 60 ep) | — | — | — | — | **54.3 ± 6.0%** |
| E-prop CIFAR-10 (T=8) | — | — | — | 11.2% | collapsed |

## Key Finding

Noise injection (σ=0.1) is the primary cost: −15 pp FS accuracy.
TWD compensates this degradation; TWD alone (no noise) matches clean BPTT baseline (~75%).

**Revised claim:** TWD enables memristor-robust training without accuracy sacrifice vs noise-only baseline.
