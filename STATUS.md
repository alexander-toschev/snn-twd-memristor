# Experiment Status

Updated: 2026-09-17

## Phase 1 — Priority 1

Concrete configs are prepared for all requested runs. The training entry point accepts
`--config PATH`; explicit command-line options override values loaded from JSON.

### Run all experiments sequentially

```bash
cd /mnt/d/projects/snn-mnist
bash scripts/run_phase1_ablations.sh
```

The launcher activates the `bindsnet` conda environment, stops on the first failed run,
and writes a combined log under `runs_csnn/phase1_logs/`.

### Experiment 1.4 — TWD ablation

- Configs: `configs/ablation/seed{0,1,2,42}_notwd.json`
- Change from champion seed configs: `use_twd=false`
- Noise remains enabled: `memristor_sigma=0.1`
- Expected first-spike accuracy: approximately 29–35%, below the 54.3% champion mean

Single-run example:

```bash
python train_temporal_snn_cifar.py --config configs/ablation/seed42_notwd.json
```

### Experiment 1.5 — Noise ablation

- Configs: `configs/ablation/seed{0,1,2,42}_nonoise.json`
- Change from champion seed configs: `memristor_sigma=0.0`
- TWD remains enabled: `use_twd=true`, `twd_weight=0.1`
- Expected first-spike accuracy: approximately 40–50%

Single-run example:

```bash
python train_temporal_snn_cifar.py --config configs/ablation/seed42_nonoise.json
```

### Experiment 1.1 — E-prop CIFAR-10 pilot

- Config: `configs/cifar10/ep_k08_T8_cifar10.json`
- Dataset: CIFAR-10, 10 classes
- Seed: 42; T=8; κ=0.8; epochs=120
- Expected first-spike accuracy: approximately 15–19%; collapse is the expected result

```bash
python train_temporal_snn_cifar.py --config configs/cifar10/ep_k08_T8_cifar10.json
```

## Runtime status

- CUDA check: available in `bindsnet` (`torch 2.5.1`, NVIDIA GeForce GTX 1660 SUPER)
- Full ablation matrix: prepared, not started by default
- E-prop CIFAR-10 pilot: running on CUDA
  - Run ID: `20260917T153713Z_5939491e1d02`
  - Run directory: `runs_csnn/20260917T153713Z_5939491e1d02/`
  - Live log: `runs_csnn/phase1_logs/exp1_1_ep_k08_T8_cifar10_seed42.log`
  - Initial validation: CIFAR-10 50,000/10,000 samples, 826,922 parameters
  - Final FS result: pending completion of 120 epochs
