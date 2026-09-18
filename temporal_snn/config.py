"""Configuration dataclass for temporal SNN experiments."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TemporalSNNCfg:
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset: str = "cifar100:0,1,2,3,5,8,13,14,17,19"
    # ^ same 10-class CIFAR100 subset as old CSNN experiments
    data_dir: str = "data"
    n_classes: int = 10

    # ── Temporal encoding ─────────────────────────────────────────────────
    T: int = 16              # timesteps (latency encoding: bright=early, dark=late)
    x_min: float = 0.1       # pixels below this → no spike (background suppression)

    # ── Architecture ──────────────────────────────────────────────────────
    # Input is 3-channel RGB 32×32 CIFAR
    c1_out: int = 32         # Conv1 output channels (5×5 kernel, no pad → 28×28)
    c2_out: int = 64         # Conv2 output channels (3×3 pad=1 → 14×14 after pool)
    fc1_out: int = 256       # FC hidden dim

    # Number of convolutional layers (1, 2, or 3).
    # n_conv_layers=1: Conv1 → pool → FC
    # n_conv_layers=2: Conv1 → pool → Conv2 → pool → FC  (default, matches prior runs)
    # n_conv_layers=3: Conv1 → pool → Conv2 → pool → Conv3 (no pool) → FC
    n_conv_layers: int = 2

    # Conv3 output channels (only used when n_conv_layers>=3)
    c3_out: int = 128

    # Conv4 output channels (only used when n_conv_layers=4)
    c4_out: int = 256

    # ── LIF neuron dynamics ───────────────────────────────────────────────
    tau: float = 0.5         # membrane decay factor (V[t+1] = tau*V[t] + I[t])
    v_threshold: float = 1.0
    surrogate_alpha: float = 2.0  # ATan surrogate steepness

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int = 120
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"   # "cosine" | "step" | "none"
    lr_step_size: int = 40         # for step scheduler
    lr_gamma: float = 0.1          # for step scheduler
    warmup_epochs: int = 5

    # ── Loss ──────────────────────────────────────────────────────────────
    # "tet"        : TET loss — CE on membrane potential at every timestep, averaged
    # "last"       : CE only at last timestep membrane
    # "first_spike": ranking loss on output spike times + TET warmup
    loss_mode: str = "tet"
    # Weight of first-spike ranking loss when loss_mode="first_spike"
    fsl_weight: float = 0.5
    # TET weight when combined with first_spike
    tet_weight: float = 1.0

    # ── Dropout ───────────────────────────────────────────────────────────
    dropout_fc: float = 0.3

    # ── Output / Checkpointing ────────────────────────────────────────────
    outdir: str = "runs_csnn"   # same root as CSNN — cron scans runs_csnn/**
    run_id: Optional[str] = None   # auto-generated if None
    save_every: int = 20           # save checkpoint every N epochs

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 2
    pin_memory: bool = True

    # ── E-prop ────────────────────────────────────────────────────────────
    use_eprop: bool = False         # use EPropAccumulator instead of BPTT
    eprop_kappa: float = 0.8        # eligibility trace decay (0 = no LPF)
    eprop_feedback: str = "symmetric"  # "symmetric" (W^T) or "random" (fixed B)
    eprop_rate_loss: bool = False   # L_out from spike counts instead of membrane

    # ── Recurrent connection (FC→LIF_fc) ─────────────────────────────────
    use_recurrent: bool = False     # add W_rec lateral connection to FC→LIF_fc
    rec_init_scale: float = 0.1    # W_rec = kaiming_normal * rec_init_scale

    # ── ALIF (Adaptive LIF, Bellec 2020) ─────────────────────────────────
    use_alif: bool = False          # use ALIF neurons in FC→LIF_fc
    alif_rho: float = 0.96          # adaptation decay
    alif_beta: float = 0.07         # threshold coupling strength

    # ── Memristor device model ────────────────────────────────────────────
    # Linear write-noise model: ΔG = ΔW·(1+N(0,σ)), G ∈ [g_min, g_max]
    memristor_sigma: float = 0.0          # noise std (0 = ideal/disabled)
    memristor_g_min: float = -1.0         # min conductance (weight clamp)
    memristor_g_max: float = 1.0          # max conductance (weight clamp)
    memristor_bits: Optional[int] = None  # precision bits (None = float32)

    # -- Winner-Takes-All lateral inhibition --
    # Hard WTA in conv layers: at each spatial position, only the channel
    # with the highest membrane potential keeps its spike; others masked to 0.
    # Applied during BPTT forward pass; gradients flow via membrane (unchanged).
    wta_enable: bool = False         # enable WTA competition
    wta_layers: str = "conv1"        # which layers: "conv1", "conv2", "all", or comma-list

    # -- Output neuron threshold --
    # Separate threshold for output LIF layer only. -1 = same as v_threshold.
    # Lower value → output fires more readily → fewer silent samples.
    v_threshold_out: float = -1.0

    # -- ETTFS-init (Che et al., ICML 2026) --
    # Scales all conv/linear weights by sqrt(T) after Kaiming init.
    # Compensates for signal diminishing in TTFS: pre-synaptic spikes are
    # T times sparser than dense activations, so weights need to be larger
    # to maintain signal variance across layers.
    ettfs_init: bool = False

    # -- TWD — Temporal Weighting Decoder (Che et al., ICML 2026) --
    # Learned α[t] weights replace argmin(first_spike_time).
    # Loss: CE(score, target) where score[c] = Σ_t softplus(α[t]) · spike[t,c]
    use_twd: bool = False
    twd_weight: float = 1.0   # weight of TWD CE loss (combined with TET)

    # -- BNTT — BatchNorm Through Time --
    # Separate BN parameters for each timestep t ∈ [0, T).
    # Standard BN shares statistics across all t; BNTT lets each t learn
    # its own scale/shift, improving accuracy at the cost of T× more BN params.
    use_bntt: bool = False

    # -- GAP — Global Average Pooling instead of Flatten before FC1 --
    # Replaces Flatten(C×H×W) → FC1 with AdaptiveAvgPool2d(1) → FC1.
    # Reduces fc1 input from C*H*W to C, eliminating the spatial bottleneck.
    # For 3-conv model: 12544→256 inputs (-98% crossbar cells).
    use_gap: bool = False

    # ── Resume ────────────────────────────────────────────────────────────
    # Set resume_checkpoint to a model state_dict .pt path to warm-start.
    # Set start_epoch > 1 to skip already-completed epochs in the loop.
    # Adam will be re-initialised (no saved optimizer state), but the LR
    # scheduler is fast-forwarded to the correct cosine/step position.
    resume_checkpoint: Optional[str] = None  # path to epoch_NNNN.pt or best_fs.pt
    start_epoch: int = 1                     # first epoch to run (1 = fresh start)

    # ── Weight Perturbation / SPSA ─────────────────────────────────────────
    # Hardware-native training: estimate gradients via two scalar loss measurements.
    # No backpropagation through the network; maps to physical memristor write.
    use_wp: bool = False             # use WP/SPSA instead of BPTT
    wp_sigma: float = 0.05           # perturbation std (c_k in SPSA notation)
    wp_sigma_decay: float = 1.0      # per-epoch multiplier (1.0 = no decay)
    wp_sigma_min: float = 0.001      # floor for sigma after decay

    # -- Greedy layer-wise training --
    use_greedy: bool = False
    greedy_epochs: int = 60       # epochs per conv stage
    finetune_epochs: int = 100    # epochs for FC fine-tune
    greedy_full_finetune: bool = False  # unfreeze all layers for finetune (end-to-end from greedy init)
    finetune_lr: float = -1.0             # LR for finetune phase; -1 = use same as cfg.lr
    fc_warmup_epochs: int = 0              # phase-1: FC-only warm-up before full finetune (0=skip)
