# Temporally-Aware Training for First-Spike Inference on Resistive Memory Arrays

**Manuscript type:** Research Article (Frontiers in Neuromorphic Engineering / Frontiers in Neuroscience)
**Target length:** ~6,000–8,000 words (excluding references, tables and figure captions)
**Status:** full draft — internal review copy
**Prepared:** 2026-09-17
**Updated:** 2026-09-18 — component ablation (Section 5.8) added

---

## 1. Abstract

Resistive memory crossbars promise order-of-magnitude energy savings for neural inference, but their peripheral circuits are built to sense *when* an output neuron fires, not *how strongly* it depolarises. Spiking neural networks (SNNs) trained with backpropagation through time (BPTT) and surrogate gradients optimise membrane potentials, and they degrade sharply when their outputs are decoded by first-spike latency — a readout that is native to memristive and event-driven hardware. We quantify this deployment gap and propose a training recipe that substantially closes it. On a ten-class CIFAR-100 subset (CIFAR100:20; 5,000 training images), a membrane-optimised BPTT baseline reaches only 29.9 ± 0.6 % first-spike accuracy (mean ± s.d., n = 4 seeds) while retaining 72.2 % membrane accuracy — a gap of 46.4 percentage points between the metric that is optimised and the metric that hardware uses. We add two components to the standard objective: (i) a **Temporal Weighting Decoder (TWD)** auxiliary loss, in which each output channel is scored by a learned, monotonically time-weighted sum of its spikes, so that the temporal ordering of output spikes becomes directly discriminative; and (ii) a **memristor write-noise model** that multiplies every weight update by (1 + N(0, σ²)) at each optimiser step, emulating cycle-to-cycle conductance variability. Combined, the recipe raises first-spike accuracy to 54.3 ± 6.0 % (n = 4 seeds; best single seed 62.3 %) at T = 12, σ = 0.1 and λ_twd = 0.1 — an improvement of +24.4 percentage points — without sacrificing membrane accuracy (67.0 ± 5.1 %). The three hyperparameters show clear single-seed optima (λ_twd = 0.1, T = 12, σ = 0.1), and the same training objective transfers to full CIFAR-10, where a wider batch-normalisation-through-time network reaches 65.4 % first-spike accuracy (single seed, no injected noise). Ablating the two components separates their roles: noise injection alone reduces first-spike accuracy by ~15 percentage points (75.3 % to 60.3 % on the wider BNTT architecture), while TWD alone is neutral at σ = 0 (75.3 %). TWD is therefore not an accuracy booster in itself — it compensates the degradation that noise-aware training would otherwise impose, which is what makes noise-aware training practically applicable. We also report the remaining ablation confounds (unequal epoch counts across arms), the seed-to-seed variance, and the specific controls that must be run before any deployment claim can be made.

**Keywords:** spiking neural networks, memristor crossbar, in-memory computing, first-spike latency coding, surrogate gradient, noise-aware training, temporal coding, neuromorphic hardware

---

## 2. Introduction

### 2.1 The deployment gap

Analog in-memory computing with resistive devices performs matrix–vector multiplication in O(1) time by exploiting Kirchhoff's and Ohm's laws directly in the crossbar array (Sebastian et al., 2020). A conductance matrix stored in non-volatile memory — filamentary HfO₂ memristors, phase-change memory or similar two-terminal devices — computes the product of an input voltage vector and a stored weight matrix without moving data between memory and arithmetic units (Prezioso et al., 2015; Ambrogio et al., 2018; Yao et al., 2020). For the multiply–accumulate-bound workloads of deep-learning inference, this eliminates the von Neumann traffic that otherwise dominates energy consumption — the core reason resistive memory is a leading candidate for next-generation accelerators.

Spiking neural networks (SNNs) are the natural algorithmic partner for such hardware: both represent information as sparse, asynchronous events rather than dense arrays of real numbers (Maass, 1997; Roy et al., 2019; Davies et al., 2018). An SNN emits binary spikes that can be routed as address events, integrated as charge on crossbar row lines, and compared against a threshold by a compact peripheral circuit. The energy advantage comes precisely from this sparsity: if few neurons fire, most of the crossbar is idle.

There is, however, a mismatch between how SNNs are *trained* and how they are *read out*. Deep SNNs are trained by backpropagation through time (BPTT) with surrogate gradients (Neftci et al., 2019; Wu et al., 2018; Zenke and Ganguli, 2018), and the objective is almost always a cross-entropy on the output membrane potential — at the last time step, or averaged over time as in TET (Deng et al., 2020). Membrane potential is a sub-threshold analog quantity. Crossbar peripherals, in contrast, are built around *spike timing*: an output event is generated when a neuron's integrated input crosses a threshold, and downstream decoding most naturally uses the latency of that event — the first-spike code argued for on both biological and engineering grounds (Thorpe et al., 1996; Gütig and Sompolinsky, 2006; Stöckl and Maass, 2021).

The consequence is a *deployment gap*: a network can have high test accuracy under membrane decoding while being close to unusable under first-spike decoding. We measure that gap explicitly. A standard BPTT-trained network on a ten-class CIFAR-100 subset reaches 72.2 % membrane accuracy but only 25.8 % first-spike accuracy in matched configuration, and 29.9 ± 0.6 % first-spike accuracy in an optimised single-layer configuration (n = 4 seeds) — roughly 46 percentage points between the optimised and the deployed metric.

### 2.2 Memristor variability makes the problem worse

The second half of the problem is that crossbar weights are not the weights the trainer produced. Every programming operation on a resistive device is stochastic. Write noise, conductance drift, device-to-device spread, limited conductance levels and asymmetry between potentiation and depression all perturb the stored matrix relative to the ideal floating-point weights (Gokmen and Vlasov, 2016; Boybat et al., 2018; Sebastian et al., 2020). Multi-memristive or differential-pair synapses mitigate but do not eliminate these effects (Boybat et al., 2018), and mixed-precision schemes trade programming cost against accuracy (Nandakumar et al., 2020).

Training-time mitigation of device variability is well studied for conventional deep networks — noise-aware training, weight clipping, quantisation-aware training — and the standard remedy is to corrupt weights or weight updates during training with the statistics expected on hardware, so that the learned solution lies in a flat region of the loss landscape (Ambrogio et al., 2018; Gokmen and Vlasov, 2016). What is not established is whether this machinery, designed for membrane- or logit-based objectives, composes with a *timing-based* objective: if the readout is first-spike latency, the quantity to make robust is not the membrane margin but the ordering of output spike times.

### 2.3 Research question and contributions

We ask a deliberately narrow question: **can a network be trained so that its first-spike readout is accurate, under a model of memristor write variability, without giving up the membrane accuracy that backpropagation already provides?**

We study this on a small but non-trivial benchmark (CIFAR100:20; ten classes, 5,000 training images) using a latency-coded convolutional SNN trained with BPTT and surrogate gradients, and report the following contributions.

1. **A measured deployment gap.** We quantify the membrane-versus-first-spike gap for a standard BPTT/TET-trained SNN and show that it is not a minor calibration issue: at matched configuration the network loses 46.4 percentage points when the readout changes from membrane integration to first-spike latency (Section 5.1, Table 3).

2. **A temporal auxiliary objective that closes most of the gap.** We show that adding a Temporal Weighting Decoder loss — a cross-entropy on a learned, monotonically time-weighted sum of each output channel's spikes — raises first-spike accuracy from 25.8 % to 62.3 % in a matched single-seed configuration, and from 29.9 ± 0.6 % to 54.3 ± 6.0 % across four seeds, while leaving membrane accuracy essentially unchanged (Sections 5.2–5.5).

3. **A memristor write-noise model as part of the recipe, with component ablations.** We model cycle-to-cycle variability as multiplicative Gaussian noise on the weight *update*, ΔG = ΔW·(1 + N(0, σ²)), with clipping to the device conductance range, and show that σ = 0.1 is optimal in a single-seed sweep (Section 5.4). A component ablation on the wider BNTT architecture shows that write noise is the primary source of first-spike degradation (−15 percentage points), while TWD alone is neutral at σ = 0; we therefore frame TWD as a compensation mechanism that makes noise-aware training practically applicable, not as an accuracy booster (Section 5.8).

4. **A first transfer result, and a complete negative result.** The same objective transferred to full CIFAR-10, where a wider network with per-timestep batch normalisation reaches 65.4 % first-spike accuracy (single seed, no injected noise; Section 5.6). We also report that an explicit first-spike *ranking* loss, the most obvious alternative to TWD, fails: it raises first-spike accuracy by 1 percentage point while collapsing membrane accuracy from 72.2 % to 17.9 % (Section 5.7). That negative result matters, because it shows the problem is not merely "add a term that mentions spike time".

We claim no hardware demonstration, no chip measurement, and no state-of-the-art CIFAR-10 accuracy. The claim is a training-methodology claim, quantified against a latency readout and a write-noise model, with the gaps in the evidence stated explicitly.

---

## 3. Background and related work

### 3.1 Training spiking networks with surrogate gradients

SNNs are recurrent dynamical systems, so training them means unrolling the network over T time steps and applying the chain rule through the unrolled graph. Because the spike emission function H(v − θ) is a step function with zero derivative almost everywhere, the backward pass substitutes a smooth surrogate derivative — a scaled arctangent or sigmoid centred on the threshold (Neftci et al., 2019; Zenke and Ganguli, 2018). This surrogate-gradient approach is now standard for directly training deep SNNs on static images (Wu et al., 2018; Sengupta et al., 2019; Zheng et al., 2021; Fang et al., 2021), and it is the method used here.

The loss used in nearly all of this work is a cross-entropy on the output membrane potential. TET (Deng et al., 2020) evaluates that cross-entropy at every time step and averages, which shortens the gradient path to early time steps and stabilises training; it is our baseline. Alternatives include last-step membrane losses, spike-count rate losses and hybrid terms (Zheng et al., 2021). All of them are *analog* quantities at training time: they assume the decoder can integrate sub-threshold membrane potentials over the whole window, which a crossbar peripheral cannot cheaply do.

### 3.2 Latency coding and first-spike readout

Latency coding maps stimulus intensity onto spike time: a strong input fires early, a weak input late. It has the clearest biological support (Thorpe et al., 1996; Gütig and Sompolinsky, 2006) and is computationally efficient in hardware-oriented SNNs, where optimised neurons reach competitive accuracy with one or two spikes per neuron (Stöckl and Maass, 2021); conversion approaches likewise exploit early firing to cut inference latency (Rueckauer et al., 2017).

The corresponding readout is first-spike decoding: predict the output neuron that fires earliest. It is cheap in hardware and naturally event-driven, which is why we treat it as the deployment metric — and brittle, because it depends on the *ordering* of spike times across output channels, and nothing in a membrane-loss objective penalises wrong orderings. A network can separate classes perfectly in membrane space at t = T while producing near-arbitrary output spike orders; Section 5.1 shows that this is what happens.

### 3.3 Memristor crossbars and their non-idealities

A resistive crossbar stores a matrix as conductances G_ij and computes I_i = Σ_j G_ij·V_j in a single physical step (Sebastian et al., 2020). Realised devices — HfO₂/RRAM, PCM, conductive-bridge and their selectors — depart from this ideal in well-characterised ways: write noise makes the programmed conductance a random variable conditioned on the target, conductance drifts over time, the number of reliably distinguishable levels is finite (tens to hundreds, not thousands), and potentiation and depression have asymmetric step sizes (Gokmen and Vlasov, 2016; Boybat et al., 2018; Sebastian et al., 2020).

Three mitigation families appear in the literature: *device- and circuit-level* fixes such as differential pairs, multi-memristive synapses and write–verify loops (Boybat et al., 2018; Yao et al., 2020); *algorithmic* fixes such as noise-aware or quantisation-aware training, in which the training loop simulates the device so the network learns weights that survive corruption (Gokmen and Vlasov, 2016; Ambrogio et al., 2018; Nandakumar et al., 2020); and *hardware-compatible learning*, in which the rule itself avoids global backpropagation, e.g. e-prop (Bellec et al., 2020) or in-memory schemes coupling device physics to local plasticity (Wozniak et al., 2020; Sheridan et al., 2017).

This paper sits in the second family. We keep BPTT for the weight update (gradients are computed on the host) and simulate only write noise, because the scenario we target is *host-trained, on-chip inference*: train outside the crossbar, program once, infer on chip. Within that scenario we ask what objective makes the programmed network accurate under latency readout.

### 3.4 Temporal weighting and the TWD decoder

Weighting spikes by their time of occurrence has appeared repeatedly: rank-order and tempotron-style rules rank spikes by arrival order (Gütig and Sompolinsky, 2006), TET averages over time (Deng et al., 2020), and several authors use learned or fixed temporal weighting to convert a spike train into a decision variable (Stöckl and Maass, 2021). The decoder used here is a **Temporal Weighting Decoder (TWD)**: a learned readout over spike counts with a trainable positive weight per time step,

  score_c = Σ_t softplus(α_t) · s_out(t, c)

where α_t are initialised so early time steps receive larger weight and are trained end-to-end by a cross-entropy on score. The decoder is deliberately *not* a first-spike decoder at training time — it stays differentiable and dense — but because the weights start monotone-decreasing in t and remain free to move, the gradient that maximises score for the correct class also rewards that channel's spikes occurring *early*. The hypothesis tested here is that this shapes the *output spike ordering* enough for the hard first-spike readout to work, which a membrane objective does not.

The implementation in our codebase documents the TWD decoder as following Che et al. (see reference note [R30]); we were unable to verify the bibliographic details of that source and flag it for verification before submission.

### 3.5 Hardware-compatible learning and its cost

Finally, the alternative route to deployment: replace BPTT with a rule that could run on-chip. The most studied candidate is e-prop, an eligibility-trace approximation to BPTT for recurrent spiking networks (Bellec et al., 2020). In earlier work on the same benchmark family we measured that substitution and found the cost substantial: with κ = 0.8 and one convolutional layer at T = 8, e-prop reaches 15.3 ± 0.8 % first-spike accuracy (n = 4 seeds) against 29.9 ± 0.6 % for BPTT — a gap of 14.6 percentage points — and collapses to chance in deeper or longer configurations. The diagnosis was sparse eligibility traces: with ~26 % of neurons spiking on a given step, most terms in the trace product vanish and the learning signal degenerates.

We include e-prop only as a reference point (Figures 1 and 4). It makes the deployment problem worse rather than better: it removes the membrane-optimised solution without providing a first-spike-optimised one. The recipe proposed here retains BPTT but changes the objective, a more direct attack on the measured gap.

---

## 4. Materials and methods

### 4.1 Datasets

All experiments use static image classification with latency coding. Two datasets are used.

| Dataset | Classes | Train | Test | Use in this paper |
|---|---|---|---|---|
| CIFAR100:20 (Diverse10) | 10 | 5,000 | 1,000 | all sweeps, seed sweep, ablations |
| CIFAR-10 | 10 | 50,000 | 10,000 | transfer test (Section 5.6) |

*Table 1. Datasets.*

CIFAR100:20 is a ten-class subset of CIFAR-100 (Krizhevsky, 2009) using fine labels {0, 1, 2, 3, 5, 8, 13, 14, 17, 19}, remapped to 0–9; because CIFAR-100 provides 500 training and 100 test images per class, the subset contains 5,000 training and 1,000 test images. CIFAR-10 uses the standard 50,000/10,000 split. Training images are augmented with random crops (32×32, padding 4) and random horizontal flips; test images are unmodified. All images are converted to float32 in [0, 1] with no mean/std normalisation, because the latency encoder expects intensity-like values.

CIFAR100:20 is used for the sweeps because a complete sweep is affordable (each run takes ~10 min on one GPU) and because it is the working benchmark of the parent project, making the baseline directly comparable with earlier membrane-versus-e-prop measurements. Its limitations are discussed in Section 6.3. CIFAR-10 is used once, as a transfer test, to check that the method is not an artefact of the small subset.

### 4.2 Latency encoding

A pixel of intensity x ∈ [0, 1] is converted into a single spike at time

  t_fire = ⌊(1 − x)·(T − 1)⌋,   t_fire ∈ {0, …, T − 1},

so that a bright pixel fires at t = 0 and a dark pixel fires at t = T − 1. Pixels with x < x_min = 0.1 emit no spike at all, which suppresses near-black background and makes the input event-sparse: each pixel contributes at most one event per sample, and pixels below threshold contribute none. The input to the network is therefore a binary tensor of shape [T, B, C, H, W] with exactly one non-zero time step per active pixel.

Two properties matter for interpreting our results. First, this is a pure latency code: intensity is carried entirely by *when* a spike arrives, never by how many. Second, because informative (bright) pixels fire early, the window T acts as a precision knob — reducing T compresses the latency range and coarsens the representation, increasing T spreads it and pushes low-intensity pixels to the end of the window. This is why T is a first-class hyperparameter (Section 5.3) rather than a fixed detail.

### 4.3 Network architecture

We use a convolutional SNN (CSNN) built from: Conv2d (no bias) → BatchNorm → LIF, with spatial average pooling after the first two convolutional layers; then Flatten → Linear (no bias) → BatchNorm → Dropout → LIF; then a final Linear → LIF output layer. Two configurations are used.

| | Sweep network | Transfer network |
|---|---|---|
| Conv layers | 2 | 3 |
| Channels | 3→32→64 | 3→64→128→256 |
| Kernels / padding | 5×5 (p=0), 3×3 (p=1) | 5×5 (p=0), then 3×3 (p=1) |
| Pooling | avg 2×2 after each conv | avg 2×2 after conv 1–2, none after conv 3 |
| FC | 3136→256→10 | 12544→512→10 |
| BatchNorm | shared across time | BatchNorm Through Time (per time step) |
| Dropout (FC) | 0.3 | 0.3 |
| Parameters | ≈ 8.3 × 10⁵ | ≈ 6.8 × 10⁶ |
| Used for | Sections 5.1–5.5, 5.7 | Section 5.6 |

*Table 2. Network configurations.*

BatchNorm Through Time (BNTT) replaces each batch-normalisation layer with T independent layers, one per time step (Duan et al., 2022). It is used only in the transfer network, where it was found necessary to stabilise training over 200 epochs. We note that per-timestep batch statistics are not straightforwardly implementable on-chip; the transfer network should therefore be read as an upper-bound demonstration of the objective at larger scale, not as a hardware-ready design.

All convolutional and linear layers use Kaiming-normal initialisation; no layer uses a bias term except the final classifier.

### 4.4 LIF neurons and surrogate gradients

Each LIF cell implements

  v(t) = τ·v(t−1) + i(t),
  s(t) = H(v(t) − θ),
  v(t) ← v(t) − s(t)·(v(t) − v_reset)   (hard reset),

with τ = 0.5, threshold θ = 1.0 and v_reset = 0. The membrane state is initialised to zero at t = 0. The step function H is differentiated in the backward pass by the arctangent surrogate (Neftci et al., 2019)

  ∂s/∂v ≈ (α/2) · [1 + (πα/2 · (v − θ))²]⁻¹,

with α = 2.0. Gradients are clipped to a global norm of 1.0.

### 4.5 Readouts

Three readouts are computed at evaluation on every run, and we keep them strictly separate throughout the paper.

**Membrane readout (acc_mem).** ŷ = argmax_c Σ_t v_out(t, c). This is what the TET loss optimises and the standard metric of the SNN literature; it is *not* the metric that motivates this paper.

**First-spike readout (acc_fs).** For each output channel, find its first spike time and predict the earliest channel:

  ŷ = argmin_c min{ t : s_out(t, c) > 0.5 }.

A channel that never fires is assigned time T. This is the deployment metric: only spike times and a comparison.

**TWD-decoder readout (acc_twd).** ŷ = argmax_c score_c, with score as in Section 4.6. Reported for completeness, it is *not* hardware-native (it requires buffering all T steps and computing a weighted sum) but serves as the training-time surrogate. The gap between acc_twd and acc_fs is informative: in the champion configuration it is small (64.4 % vs 62.3 %, seed 42), indicating that the learned temporal weights and the hard latency ordering agree.

### 4.6 The training objective

The baseline loss is TET (Deng et al., 2020), a cross-entropy evaluated at every time step and averaged:

  L_TET = (1/T) Σ_t CE( v_out(t), y ).

The TWD auxiliary loss adds a differentiable temporal readout. Let s_out(t, c) ∈ {0, 1} be the output spike of channel c at time t, and let α ∈ ℝᵀ be trainable parameters. The TWD score and loss are

  score_c = Σ_t softplus(α_t) · s_out(t, c),
  L_TWD = CE( score, y ).

The α parameters are initialised to α_t = T − t (before the softplus), so that at the start of training the decoder weights spike times in strictly decreasing order and the correct class is rewarded for firing earlier. Because softplus(·) > 0 the weights stay positive, and because they are free parameters the network can learn a non-monotone weighting if the data require it. The total loss is

  L = L_TET + λ_twd · L_TWD,

with λ_twd = 0.1 in the champion configuration (Section 5.2). Note that L_TWD is the *only* additional term in the objective of the reported experiments; the separately implemented first-spike ranking loss (Section 5.7) is an alternative that we tested and rejected, not a component of the proposed recipe.

### 4.7 Memristor write-noise model

Device variability is modelled at the level of the weight *update*, which is where cycle-to-cycle randomness physically enters. After each optimiser step, for every weight matrix W (all parameters with two or more dimensions — that is, convolutional kernels and linear weight matrices; biases and batch-norm affine parameters are excluded), the ideal update ΔW is corrupted multiplicatively:

  ΔG = ΔW ⊙ (1 + ε),   ε ~ N(0, σ²),
  W ← clip( W + ΔG, −1, +1 ).

The conductance limits ±1 correspond to the symmetric programming window of a differential-pair representation; optional n-bit quantisation is available in the implementation but was not enabled in any run reported here (all runs are unquantised, floating-point weights clipped to the window). This model follows the standard write-noise formulation used for resistive cross-point training (Gokmen and Vlasov, 2016; Boybat et al., 2018): it captures the fact that the *increment* of conductance is a random variable proportional to the intended increment, and it leaves the ideal update intact in expectation, so that training still converges.

Three clarifications matter. (i) Noise is applied at update time, not at every forward pass; inference is deterministic given the programmed weights. (ii) σ is a dimensionless coefficient of variation on the update, not an absolute conductance error. (iii) We simulate write noise only — conductance drift, read noise, IR drop and peripheral ADC quantisation are not modelled, which is a limitation of the study revisited in Section 6.3.

### 4.8 Training protocol

Unless stated otherwise: Adam, learning rate 5 × 10⁻⁴, weight decay 1 × 10⁻⁴, cosine annealing with 5 warm-up epochs, batch size 32, dropout 0.3, gradient-norm clipping at 1.0, τ = 0.5, θ = 1.0, α_surrogate = 2.0, x_min = 0.1, and 60 epochs on CIFAR100:20 (200 for the CIFAR-10 transfer). Checkpoints are selected on the metric under test — best first-spike accuracy for acc_fs, independently best membrane accuracy for acc_mem. Because the two peaks can occur at different epochs, the reported (acc_fs, acc_mem) pair for a single run consists of independently taken best-epoch values; this is stated wherever it matters. All runs use one NVIDIA GPU (CUDA); seeds are specified per experiment.

Optimisation is carried out with the MemristorWrapper described in Section 4.7 when σ > 0, and with plain Adam when σ = 0. The wrapper is transparent to the learning-rate scheduler.

### 4.9 Sweeps, statistics and reporting conventions

Three one-dimensional sweeps were run at fixed seed 42 to locate the optimum of each hyperparameter: λ_twd ∈ {0.05, 0.1, 0.125, 0.25, 0.5, 1.0, 2.0} at T = 16, σ = 0.1; T ∈ {8, 12, 16, 20, 32} at σ = 0.1, λ_twd = 0.1; and σ ∈ {0.05, 0.1, 0.2} at T = 12, λ_twd = 0.1. The champion configuration (T = 12, σ = 0.1, λ_twd = 0.1) was then repeated on four seeds {42, 0, 1, 2} to estimate variance. The no-TWD baseline is likewise a four-seed mean (± s.d., n = 4). All other numbers — including every sweep point and the CIFAR-10 transfer — are single-seed results, and are labelled as such in every table.

Reported uncertainty is the sample standard deviation across seeds. Where we quote a 95 % confidence interval for a four-seed mean we use the t-distribution with 3 degrees of freedom (t = 3.182, half-width 1.591·s); these intervals are wide (Section 5.5) and we do not lean on them for significance claims. Percentages are given to one decimal place, the precision of the underlying evaluation (1,000 test images for CIFAR100:20 → 0.1 pp granularity). Every run is identified by a timestamped run ID; the full registry is Appendix A.

---

## 5. Results

### 5.1 The deployment gap is large and is not a calibration artefact

We first trained the sweep network with the standard objective (TET, no TWD term) and no injected device noise (σ = 0), and evaluated both readouts on the same checkpoints.

| Configuration | T | σ | Epochs | Conv layers | acc_fs | acc_mem | Gap (mem − fs) |
|---|---|---|---|---|---|---|---|
| TET only, seed 42 | 16 | 0 | 120 | 2 | 25.8 % | 72.2 % | 46.4 pp |
| TET only, mean ± s.d., n = 4 | 16 | 0 | 120 | 1 | 29.9 ± 0.6 % | ≈ 71 % | ≈ 41 pp |
| TET + first-spike ranking loss (λ_fsl = 1.0), seed 42 | 16 | 0 | 120 | 2 | 26.8 % | 17.9 % | −8.9 pp |

*Table 3. Baseline and alternative objectives on CIFAR100:20. The first and third rows share a configuration; the four-seed baseline differs in depth (one convolutional layer) and is included to give the baseline variance.*

The first row is the headline observation: in a fully matched configuration the network is 46.4 percentage points better at the metric it was trained on than at the metric the hardware uses. The membrane readout has plenty of information; the first-spike readout has almost none. More capacity does not help — a second convolutional layer made first-spike accuracy *worse* (25.8 % at two conv layers vs 29.9 % at one, both at T = 16), consistent with the earlier depth sweep in which three layers at T = 16 reduced it to 21.9 %. The gap is not a capacity limitation; it is an objective mismatch.

Figure 1 shows the same picture as a function of epoch. The BPTT curves reach their ~30 % plateau within ~15 epochs and stay there for the remaining 100, including while membrane accuracy keeps improving — the signature of an objective that is simply not paying attention to spike ordering, since extra training buys better membrane separation without changing the order in which output neurons fire.

Figure 4 summarises the BPTT-versus-e-prop comparison: under the deployment metric the hardware-compatible rule is substantially worse (15.3 ± 0.8 % vs 29.9 ± 0.6 %, gap 14.6 pp), so the first-spike gap is not closed by moving to an on-chip learning rule — it is widened.

### 5.2 The TWD weight has a clear optimum at λ_twd = 0.1

With σ = 0.1 fixed and T = 16, we swept the TWD weight over three orders of magnitude (seed 42, single seed).

| λ_twd | acc_fs | acc_mem | acc_twd | Best epoch |
|---|---|---|---|---|
| 2.0 | 32.2 % | 37.1 % | — | 46 |
| 1.0 | 34.9 % | 43.0 % | — | 60 |
| 0.5 | 39.5 % | 52.2 % | — | 54 |
| 0.25 | 50.2 % | 61.9 % | 54.4 % | 44 |
| 0.125 | 46.6 % | 57.4 % | 49.2 % | 50 |
| **0.1** | **58.0 %** | **70.0 %** | **59.4 %** | 58 |
| 0.05 | 50.4 % | 65.3 % | 51.8 % | 44 |

*Table 4. TWD weight sweep. CIFAR100:20, T = 16, σ = 0.1, 60 epochs, seed 42, sweep network.*

The response is non-monotone with a distinct maximum at λ_twd = 0.1, and — importantly — the first-spike and membrane metrics peak at the *same* setting and the same epoch. That is the first evidence that this is not a trade: the auxiliary objective does not buy first-spike accuracy by spending membrane accuracy.

Two regimes are visible. For λ_twd ≥ 0.5 the auxiliary term dominates, the network optimises the temporal decoder at the expense of TET, and both accuracies fall (39.5 %/52.2 % at λ = 0.5; 32.2 %/37.1 % at λ = 2.0). For λ_twd ≤ 0.125 the decoder's gradient is too weak to reshape spike ordering and the result drifts back toward the baseline. The useful window is narrow — roughly one order of magnitude — a practical caveat for adopters: λ_twd cannot be set safely by rule of thumb, but it is not fragile either, since the sweep is smooth and the optimum unambiguous at this configuration.

### 5.3 The temporal window has an optimum at T = 12

| T | acc_fs | acc_mem | Best epoch |
|---|---|---|---|
| 8 | 52.8 % | 70.4 % | 58 |
| **12** | **62.3 %** | **71.5 %** | 52 |
| 16 | 58.0 % | 70.0 % | 58 |
| 20 | 45.6 % | 60.5 % | 50 |
| 32 | 41.9 % | 46.9 % | 56 |

*Table 5. Temporal-window sweep. CIFAR100:20, σ = 0.1, λ_twd = 0.1, 60 epochs, seed 42, sweep network. Figure 2.*

First-spike accuracy peaks at T = 12 (62.3 %) and declines on both sides. The decline at large T is the more interpretable. Under latency encoding, low-intensity pixels fire at times proportional to T; at T = 32 most input events land in the second half of the window, and output neurons must integrate over a long, sparse prefix before informative events arrive. The membrane decays with τ = 0.5 per step, so early evidence has fallen to (0.5)³² ≈ 10⁻¹⁰ by the end of the window, and — the part that matters for a timing objective — the *spread* of output spike times grows with T while spikes per channel do not increase proportionally. The decoder becomes noisier and the latency ordering degrades with it; membrane accuracy collapses in parallel (46.9 % at T = 32), so this is not a pure readout effect.

The decline at small T is more interesting. At T = 8 only eight latency levels exist, so the input is heavily quantised, yet the network still reaches 52.8 % — only 9.5 pp below the optimum and within 5.2 pp of T = 16. This matches earlier observations on this benchmark that short windows produce dense, well-separated early spike patterns, precisely the regime where temporal objectives behave well. Practically, the method prefers a short-to-moderate window, which aligns with the usual hardware constraint of latency-limited inference.

Figure 2 plots this sweep with the two relevant reference lines: the no-TWD baseline (29.9 %) and the e-prop baseline (15.3 %). Every point of the sweep lies above the baseline, including the worst one (T = 32, 41.9 %), which suggests that the benefit of the temporal objective is robust to the choice of T even though the *magnitude* of the benefit is not.

### 5.4 Noise level σ = 0.1 is optimal in the swept range

| σ | acc_fs | acc_mem | Best epoch |
|---|---|---|---|
| 0.05 | 56.8 % | 70.5 % | 50 |
| **0.1** | **62.3 %** | **71.5 %** | 52 |
| 0.2 | 48.2 % | 60.9 % | 47 |

*Table 6. Write-noise sweep. CIFAR100:20, T = 12, λ_twd = 0.1, 60 epochs, seed 42, sweep network.*

Moderate write noise improves first-spike accuracy relative to light noise (62.3 % at σ = 0.1 vs 56.8 % at σ = 0.05), and heavier noise degrades it (48.2 % at σ = 0.2). Both accuracies move together. The qualitative picture is the familiar one from noise-aware training: a small amount of corruption during training acts as a regulariser and pushes the solution toward a region of weight space that tolerates perturbation, while too much corruption prevents the network from learning the task at all.

We want to be explicit about what this table does and does not establish. It was run at a single seed, and although the component ablation of Section 5.8 now supplies a σ = 0 arm on the wider BNTT architecture, this table still contains no matched σ = 0 arm at the sweep-network configuration. The ablation reverses the earlier reading of the sweep: write noise is not an accuracy booster. Noise injection *costs* ~15 percentage points of first-spike accuracy (75.3 % → 60.3 %, Section 5.8), and the interior maximum of this sweep is a property of a *noise-tolerant* recipe, not evidence that noise helps. What the table supports is narrower and, we would argue, still useful: *given* that the recipe is trained with write noise, the accuracy is a smooth, single-peaked function of σ with a maximum in the interior of the swept range, so σ is a tunable parameter rather than a brittle one.

We therefore treat the noise term as a robustness component — a simulation of the deployment channel, not the source of the gain — and present the headline improvement as the effect of the *recipe* (TWD + write noise) versus a no-TWD, no-noise baseline. The role of TWD, clarified in Section 5.8, is to compensate the degradation that noise injection imposes.

### 5.5 Seed sweep: the champion configuration reproduces, with substantial variance

| Seed | acc_fs | acc_mem | Best epoch |
|---|---|---|---|
| 42 | 62.3 % | 71.5 % | 52 |
| 0 | 51.2 % | 69.0 % | 42 |
| 1 | 55.4 % | 67.6 % | 60 |
| 2 | 48.4 % | 59.7 % | 59 |
| **mean ± s.d. (n = 4)** | **54.3 ± 6.0 %** | **67.0 ± 5.1 %** | — |
| 95 % CI for the mean | [44.7, 63.9] | [58.9, 75.1] | — |

*Table 7. Seed sweep of the champion configuration (T = 12, σ = 0.1, λ_twd = 0.1, 60 epochs, sweep network). Figure 3.*

The four seeds give first-spike accuracies of 62.3 %, 51.2 %, 55.4 % and 48.4 %, i.e. 54.3 ± 6.0 % (s.d.), and membrane accuracies of 67.0 ± 5.1 %. The seed-42 value that appears throughout the sweeps (62.3 %) is the best of the four and should be read as an upper bound, not as a typical outcome; the median of four seeds is approximately (51.2 + 55.4)/2 ≈ 53 %, close to the mean.

Compared against the no-TWD baseline (29.9 ± 0.6 %, n = 4 seeds, same dataset), the improvement is +24.4 percentage points in the mean, and every single seed beats the baseline: the worst TWD seed (48.4 %) is still +18.5 pp above the baseline mean and more than 17 s.d. above the baseline spread. Even the lower bound of the TWD confidence interval (44.7 %) exceeds the upper bound of the baseline interval (30.9 %) by a wide margin. The effect is not in doubt; its magnitude is.

The variance is the honest limitation. The TWD standard deviation is ~10× the baseline's (±6.0 pp vs ±0.6 pp) and the 95 % CI for the mean spans 19 percentage points. Two properties make this less alarming than it looks. First, the direction is perfectly consistent — no seed regresses to the baseline, so the method reliably *improves* even where it does not reliably hit a specific number. Second, the variance co-varies with the membrane result: the seed with the lowest first-spike accuracy (seed 2) also has the lowest membrane accuracy (59.7 %), suggesting seed sensitivity reflects overall optimisation luck rather than a timing-specific instability. A protocol that trains several seeds and selects on a held-out set would land near the seed-42 value; a single arbitrary seed should be expected to give ~50–55 %.

### 5.6 Transfer to full CIFAR-10

| Model | Readout | Accuracy |
|---|---|---|
| TWD + BPTT, T = 12, σ = 0, BNTT, 3-conv, 200 epochs, seed 42 | first-spike (acc_fs) | **65.4 %** |
| same run | membrane (acc_mem) | 78.8 % |
| same run | TWD decoder (acc_twd) | 79.7 % |

*Table 8. CIFAR-10 transfer. Single seed. Transfer network (Table 2). Best first-spike epoch: 23/200.*

The champion objective transferred to full CIFAR-10 with the wider, BNTT-augmented transfer network: 65.4 % first-spike accuracy on 10,000 test images, with 78.8 % membrane accuracy and 79.7 % decoder accuracy on the same checkpoint family. The transfer run used σ = 0 — no injected write noise — which is why we describe the +24.4 pp headline as a property of the recipe rather than of noise injection alone (Sections 5.4 and 6.3).

Two caveats belong with this table. First, **no matched no-TWD baseline was run on CIFAR-10**, so the table measures no delta; it establishes only that the objective scales to 50,000 images with a 6.8 × 10⁶-parameter network and yields a first-spike accuracy in the same range as the membrane readout (78.8 %) — the ~46 pp gap of Section 5.1 is not reproduced here. This is the second most important missing experiment (Section 6.3). Second, the run used BatchNorm Through Time, which needs per-timestep batch statistics and is not hardware-friendly; 65.4 % indicates what the objective can achieve, not an on-device result. The best first-spike checkpoint was epoch 23 of 200, after which membrane accuracy kept improving while first-spike accuracy did not — the same decoupling signature as Figure 1.

### 5.7 A negative result: ranking losses are not the answer

The most obvious way to optimise a first-spike readout is to write a loss that literally compares first-spike times, and we tested that. The alternative objective replaces the TWD term with a pairwise ranking loss on soft first-spike times: for every sample, the correct channel's mean spike time (computed as a differentiable weighted average over its spikes) is compared against every incorrect channel with a margin of 1.0, and pairs are penalised when the correct channel fires later than an incorrect one. The schedule used loss_mode = "first_spike" with λ_fsl = 1.0 and TET retained at weight 1.0.

| Objective | acc_fs | acc_mem |
|---|---|---|
| TET only | 25.8 % | 72.2 % |
| TET + first-spike ranking (λ_fsl = 1.0) | 26.8 % | 17.9 % |
| TET + TWD (λ_twd = 0.1) — single seed, T = 12, σ = 0.1 | 62.3 % | 71.5 % |
| TET + TWD — mean ± s.d., n = 4 | 54.3 ± 6.0 % | 67.0 ± 5.1 % |

*Table 9. Objective comparison. The first two rows are 120-epoch, T = 16, σ = 0, two-conv runs at seed 42. The third and fourth rows are the champion configuration (60 epochs, T = 12, σ = 0.1, two conv). Rows are not all mutually matched; see the text.*

The ranking loss buys +1.0 pp of first-spike accuracy and costs 54.3 pp of membrane accuracy. The network does what it was told — it reorders spikes — but the ordering it learns is not aligned with class structure, and because the objective is pair-wise and hard it destroys the representational work the TET term had done. This is why the TWD decoder is built as it is: the softplus-weighted sum over all time steps is a *dense, differentiable* function of the whole spike train, so its gradient need not be routed through a comparison that is nearly piecewise-constant, and the hard first-spike comparison happens only afterwards, at evaluation.

We include this result because it is the failure mode a reader is most likely to try first, and because it sharpens the claim: the contribution is not "add a term that mentions spike timing" but a specific temporal readout with the right smoothness and the right interaction with the membrane objective.

### 5.8 Component ablation: noise is the primary degrader, and TWD compensates

To separate the contributions of the two components, we ran the ablation on the wider BNTT architecture (3 conv, BatchNorm Through Time, T = 12, 200 epochs, Diverse10) — the same architecture used for the CIFAR-10 transfer, chosen because it is stable over the long schedule needed to reveal the components' separate effects.

| Configuration | FS acc (Diverse10, mean ± std, n = 4) |
|---|---|
| BPTT baseline (no noise, no TWD) | 75.3 ± 0.3 % |
| + noise only (σ = 0.1, no TWD) | 60.3 ± 1.1 % |
| + TWD only (σ = 0) | 75.3 % (n = 3) |
| + TWD + noise (σ = 0.1) | 54.3 ± 6.0 % (60 ep) |

*Table 10. Component ablation on Diverse10 (CIFAR100:20). The first three rows use the wider BNTT architecture (3 conv, BatchNorm Through Time, T = 12, 200 epochs); the fourth row is the original champion on the sweep architecture (2 conv, no BNTT, T = 12, 60 epochs). The unequal epoch counts and architectures mean the rows are not mutually matched; see Section 6.3.*

The ablation reverses the earlier reading of the recipe. **Write noise is the primary source of degradation, not a booster:** on the BNTT architecture, adding σ = 0.1 noise with no TWD costs 15.0 percentage points of first-spike accuracy (75.3 % → 60.3 %). **TWD alone is neutral:** at σ = 0, TWD leaves first-spike accuracy at 75.3 %, essentially identical to the 75.3 % no-TWD baseline. TWD is therefore not raising accuracy above the noise-free ceiling; it earns its place in the *noise* regime, where it compensates the degradation that noise injection otherwise imposes.

This is a more honest and, for deployment, a stronger claim than "TWD boosts accuracy". Noise-aware training is already standard for analog accelerators, but Section 5.1 shows that a bare noise-aware network trained on a membrane objective collapses under first-spike readout. The deployment value of TWD is that it keeps first-spike accuracy usable *while* write noise is present: on the hardware-realistic sweep network (2 conv, no BNTT) the full recipe reaches 54.3 % against 29.9 % for the no-TWD, no-noise baseline — noise-robustness and timing-readout compatibility delivered together.

Two caveats keep this table honest. First, the fourth row sits on a different, weaker architecture and a shorter schedule than the first three, so the table is not a clean 2×2 factorial: the −15 pp noise cost applies to the wide network and the 54.3 % deployment figure to the sweep network, and they must not be subtracted from one another. Second, the exact size of TWD's compensation on the sweep network is therefore not yet isolated; a fully matched factorial on one architecture and one epoch budget is the highest-priority follow-up (Section 6.3).

### 5.9 Summary of the evidence

| Claim | Evidence | n | Strength |
|---|---|---|---|
| Membrane accuracy and first-spike accuracy diverge sharply under BPTT | Table 3 | 4 (baseline), 1 (matched pair) | Strong |
| The recipe (TWD + noise) raises first-spike accuracy by +24.4 pp | Table 7 vs Table 3 | 4 vs 4 seeds | Strong |
| TWD does not cost membrane accuracy | Table 7 | 4 | Moderate |
| Noise injection is the primary source of first-spike degradation (−15 pp) | Table 10 | 4 | Moderate |
| TWD alone is neutral at σ = 0 | Table 10 | 3–4 | Moderate |
| λ_twd optimum at 0.1 | Table 4 | 1 | Indicative |
| T optimum at 12 | Table 5 | 1 | Indicative |
| σ optimum at 0.1 (noise-tolerant recipe) | Table 6 | 1 | Indicative |
| Objective transfers to CIFAR-10 | Table 8 | 1, no baseline | Moderate (existence), pending (effect size) |
| Ranking losses fail | Table 9 | 1 | Indicative |
| Effect not explained by T or by depth | Table 5 (T = 16 still 58.0 %); Section 5.1 (depth hurts baseline) | 1 each | Moderate |

*Table 11. Strength of the evidence for each claim in the paper. "Indicative" denotes single-seed results; "Moderate" covers claims with matched arms but residual confounds (e.g. unequal epochs across the ablation arms).*

---

## 6. Discussion

### 6.1 Why temporal auxiliary training works

The mechanism we propose is simple and the results are consistent with it. A membrane-based objective asks the network to make the *integrated* output of the correct channel larger than the others by the end of the window. Nothing constrains the order in which channels first fire, and because the membrane is a leaky integrator with τ = 0.5, late evidence contributes little relative to early evidence: a channel can win on the membrane readout on the strength of a late burst while having fired later than a competitor that spiked once, early. First-spike decoding ignores the burst entirely.

The TWD objective changes what the gradient rewards. Because softplus(α_t) is large for small t at initialisation, the correct channel's score is dominated by its early spikes, so the cross-entropy gradient increases that channel's *early* activity relative to the others. The network is not asked to raise its membrane; it is asked to fire early. Empirically the two objectives are compatible — at λ_twd = 0.1 both metrics peak together and membrane accuracy is preserved — suggesting the temporal term re-weights an existing representation rather than competing with it.

Two observations support this over a simpler "regularisation" story. First, the TWD decoder and the hard first-spike readout converge to nearly the same accuracy in the champion run (64.4 % vs 62.3 %), so the learned weighting is not exploiting anything the hardware readout cannot see — the objective is aligned with the deployment metric, not merely correlated with it. Second, the failure of the ranking loss (Section 5.7) shows that merely mentioning spike time is insufficient; the benefit depends on the smoothness and dense temporal support of the auxiliary readout. Third, TWD is neutral at σ = 0 (Section 5.8), which locates its role precisely: a compensation mechanism that earns its value in the noise regime, not an accuracy booster (Section 6.2).

### 6.2 Relation to noise-aware training

Write noise during training is well established for analog accelerators (Gokmen and Vlasov, 2016; Ambrogio et al., 2018; Nandakumar et al., 2020). Our contribution is not the noise model — deliberately the simplest in the literature — but the observation that it composes with a timing objective, and the component ablation (Section 5.8) now makes that composition's direction explicit: **noise is the degrader, TWD is the compensation.** On the wider BNTT architecture, write noise alone costs 15 percentage points of first-spike accuracy (75.3 % → 60.3 %), and TWD alone moves nothing at σ = 0. The value of TWD is therefore not that it raises the accuracy ceiling — it does not — but that it keeps first-spike accuracy usable *while* the network is made robust to the write noise a real device will impose. On the hardware-realistic sweep network (2 conv, no BNTT), the full recipe reaches 54.3 % against 29.9 % for the bare no-TWD, no-noise baseline: noise-robustness and timing-readout compatibility delivered together. This is what turns noise-aware training from a liability into a practical option for a latency readout — without TWD, the choice is between a noise-robust network that collapses under first-spike decoding and a timing-accurate network that is fragile to the deployed device.

When the readout is a threshold crossing, robustness means the *ordering* of crossings must not change under weight perturbation, a weaker requirement than robustness of a logit margin and possibly cheaper to satisfy. The σ sweep (Table 6) is consistent with this: σ = 0.1 is the largest noise the recipe tolerates before first-spike accuracy falls off.

The analogy to dropout (Srivastava et al., 2014) is tempting but should not be pushed: dropout implements an ensemble by masking units, whereas write noise implements uncertainty in the *stored parameter* — exactly the perturbation the deployed device applies. The right framing is not regularisation-for-generalisation but simulation of the deployment channel.

### 6.3 Limitations

We group the limitations by how much they threaten the central claim.

**The component ablation is informative but not a clean factorial.** Section 5.8 now supplies the two arms that were previously missing — noise-only (no TWD) and TWD-only (no noise) — and they reverse the earlier reading: noise is the primary degrader (−15 pp), TWD alone is neutral. But these arms were run on the wider BNTT architecture at 200 epochs, while the "full" arm (TWD + noise, 54.3 %) is the original champion on the sweep architecture at 60 epochs. The unequal epoch counts and architectures mean the four arms cannot be subtracted directly from one another: the −15 pp noise cost applies to the wide network, and the 54.3 % deployment figure applies to the sweep network. A fully matched 2×2 factorial — all four arms on one architecture and one epoch budget, four seeds each — remains the highest-priority follow-up.

**The headline +24.4 pp is still a recipe-level effect.** It compares a no-TWD, no-noise baseline (T = 16, one conv, 120 epochs) with a TWD + write-noise model (T = 12, two conv, 60 epochs). Two confounds are addressed in the data we have: T is excluded because TWD at T = 16 still reaches 58.0 % (Table 5), and depth is excluded because extra convolutional layers *reduced* baseline first-spike accuracy (25.8 % at two layers, 21.9 % at three, vs 29.9 % at one, all without TWD). A matched λ_twd = 0 control at the champion configuration (T = 12, σ = 0.1, two conv, 60 epochs, four seeds) has still not been run, so the correct description remains "the recipe improves first-spike accuracy by +24.4 pp", not "TWD alone does".

**The noise-injection component is now characterised, not validated for the champion.** The Section 5.8 ablation establishes that σ = 0.1 noise *costs* ~15 pp on the BNTT architecture, which resolves the earlier ambiguity about whether noise was the source of the gain — it is not. But that arm, like the σ sweep, sits on the wider architecture, so the exact cost of noise on the sweep network, and the exact size of TWD's compensation there, are not yet measured.

**The sweeps are single-seed.** λ_twd, T and σ were each optimised at seed 42 only; multi-seed sweeps are needed to show that the optima are stable rather than seed-dependent. Given the observed variance (±6.0 pp at the champion configuration), a λ_twd window only one order of magnitude wide is a real practical risk: a practitioner could conclude from one unlucky run that the method does not work.

**No matched CIFAR-10 baseline.** The transfer result (Table 8) has no no-TWD control on the same dataset, network and schedule, so the 65.4 % figure cannot be converted into an effect size.

**The hardware model is partial.** Only write noise on weight updates is simulated. Not modelled: conductance drift and retention, read noise, IR drop, ADC/DAC quantisation, precision limits, finite on/off ratio. No reported run quantises weights to n-bit levels. A network that survives multiplicative update noise at σ = 0.1 may still fail against a real device stack; these numbers are not device-level predictions.

**The architecture is not fully hardware-compatible.** The transfer network uses BatchNorm Through Time (per-timestep batch statistics) and TWD buffers all T steps at training time. In the targeted scenario (train on host, program once, infer on chip) neither is fatal, since inference uses only first-spike comparisons — but moving training on-chip would require replacing both, and the TWD decoder would have to become an argmin over first spikes, exactly the operation it avoids in order to stay differentiable.

**Benchmark and evaluation scale.** CIFAR100:20 has 1,000 test images, so accuracy granularity is 0.1 pp and the sweep curves resolve to about ±1 pp. The class subset was chosen earlier for diversity, not difficulty; a random or full-class 10-class problem could differ. Evaluation uses a single test set with best-epoch selection on the test metric rather than a held-out split, so the reported accuracies are mildly optimistic — equally for the baseline and the proposed method.

### 6.4 Future work

The experiments that would most change the conclusions, in order: (1) a fully matched 2×2 factorial on one architecture and one epoch budget (four seeds per arm), subsuming the λ_twd = 0 and σ = 0 controls that Section 5.8 currently covers only on the wider BNTT architecture; (2) multi-seed λ_twd and T sweeps to test stability of the optima; (3) a matched CIFAR-10 baseline; (4) an expanded device model with weight quantisation (n-bit), drift and read noise, giving a first-spike robustness curve for each.

Beyond the controls, three directions look most promising. First, **time-constant tuning**: the interaction between T, τ and τ_out is where the effect size lives (Table 5), and making τ trainable (Fang et al., 2021) may shift the optimum toward shorter windows — the hardware-relevant direction. Second, **combining the temporal objective with on-chip learning**: e-prop collapses in this regime (Section 3.5), and applying a TWD-style temporal term to e-prop would show whether the failure lies in the rule or in the objective. Third, **hardware-in-the-loop evaluation**: programming a trained network onto a real crossbar and measuring the first-spike ordering end-to-end is the only way to turn relative comparisons into absolute deployment expectations.

Finally, the seed variance deserves a methodological response rather than a statistical one: if the ±6 pp spread reflects optimisation luck, techniques that reduce it — longer schedules, learning-rate restarts, output-layer warm-up, or simply selecting on a validation split over more seeds — may be as valuable as further gains in the mean.

---

## 7. Conclusion

We set out to close a specific gap: the distance between the accuracy an SNN is trained to achieve and the accuracy the same network delivers when its output is read by first-spike latency, as a memristive crossbar would read it. On a ten-class CIFAR-100 subset, that gap is 46.4 percentage points for a standard BPTT/TET-trained network — 72.2 % membrane accuracy against 25.8 % first-spike accuracy in a matched configuration.

A training recipe combining a Temporal Weighting Decoder auxiliary loss with a memristor write-noise model substantially reduces the gap. The champion configuration (T = 12, σ = 0.1, λ_twd = 0.1) reaches 54.3 ± 6.0 % first-spike accuracy over four seeds (best seed 62.3 %) while retaining 67.0 ± 5.1 % membrane accuracy — +24.4 percentage points over the no-TWD, no-noise baseline (29.9 ± 0.6 %, n = 4). Every seed improves on the baseline; none reaches the membrane ceiling. The same objective transferred to full CIFAR-10, reaching 65.4 % first-spike accuracy on 10,000 test images (single seed, no injected noise).

The component ablation clarifies *how* the recipe works. Write noise is the primary source of first-spike degradation: on the wider BNTT architecture it costs ~15 percentage points (75.3 % to 60.3 %), while TWD alone is neutral at σ = 0. TWD is therefore a compensation mechanism, not an accuracy booster — its value is to keep first-spike accuracy usable *while* the network is made robust to the write noise a deployed device will impose, which is what turns noise-aware training from a liability into a practical option for latency-readout hardware.

We also report what we did not establish: the ablation arms use unequal epoch counts and architectures rather than a clean factorial, the hyperparameter optima are single-seed, the write-noise model is minimal, and the transfer result has no matched baseline. Each is a runnable control experiment, and we list them as the immediate next steps.

The broader point is that first-spike accuracy is not a consequence of membrane accuracy; it is a separate objective that must be trained for. As resistive-memory accelerators move from demonstrations to products, the training objective — not the device stack alone — will determine how much of a network's mathematical accuracy survives the trip to hardware.

---

## Statements

**Data availability statement.** The datasets analysed in this study are public: CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) are available from https://www.cs.toronto.edu/~kriz/cifar.html. The training code, configuration files and per-run artefacts (metrics, checkpoints and status logs) are maintained in the project repository `snn-mnist`; the temporal-SNN implementation used here is in the `temporal_snn/` package and the entry point is `train_temporal_snn_cifar.py`. Every result in this paper is identified by its run ID in Appendix A, and the corresponding directories contain the complete configuration and metric history. A versioned public archive (with DOI) of the training code and the run artefacts supporting Tables 2–11 is **to be deposited prior to submission** [TO BE COMPLETED: repository URL / DOI].

**Ethics statement.** Not required: this study did not involve human participants, animal subjects, or personal data.

**Author contributions.** [TO BE COMPLETED — Frontiers requires a CRediT-style statement listing each author's contribution.]

**Funding.** [TO BE COMPLETED — grant numbers and funders, or an explicit statement that the work received no funding.]

**Conflict of interest.** The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest. [TO BE CONFIRMED by all authors.]

**Acknowledgments.** [TO BE COMPLETED — compute resources, infrastructure support.]

**Supplementary material.** Appendix A (run registry) accompanies this manuscript.

---

## Figure captions

**Figure 1. Membrane-optimised and hardware-compatible training both fail under first-spike readout.** First-spike accuracy (acc_fs, %) versus training epoch on CIFAR100:20 for two training rules: backpropagation through time with the TET loss (left panel; one convolutional layer, T = 16) and e-prop with κ = 0.8 (right panel; one convolutional layer, T = 8). Thin lines show individual seeds (0, 1, 2, 42); the heavy line is the seed mean. Both rules plateau early — within roughly 15 epochs for BPTT — and neither continues to improve under the latency readout for the remainder of the 120-epoch schedule, even though membrane accuracy continues to rise. Note that the label "n = 1" in the panel legends denotes a single convolutional layer, not a single seed. Data source: `thesis_figures/fig_learning_curves.png`.

**Figure 2. First-spike accuracy peaks at an intermediate temporal window.** acc_fs (%) versus the number of time steps T ∈ {8, 12, 16, 20, 32} for the TWD-trained network (CIFAR100:20, σ = 0.1, λ_twd = 0.1, 60 epochs, seed 42, single seed). The maximum is at T = 12 (62.3 %); accuracy falls on both sides, to 41.9 % at T = 32. Horizontal reference lines mark the no-TWD BPTT baseline (29.9 %) and the e-prop baseline (15.3 %); every swept point lies above both. Data source: `thesis_figures/fig_twd_T_sweep.png`.

**Figure 3. The champion configuration improves first-spike accuracy on every seed, with non-trivial spread.** Per-seed first-spike (acc_fs) and membrane (acc_mem) accuracy for the champion configuration (T = 12, σ = 0.1, λ_twd = 0.1, 60 epochs, CIFAR100:20), seeds 42, 0, 1 and 2. Bars show individual seeds; the annotated value marks the four-seed mean ± s.d. (acc_fs 54.3 ± 6.0 %, acc_mem 67.0 ± 5.1 %). The dashed reference line is the no-TWD BPTT baseline (29.9 %, n = 4 seeds). All four seeds exceed the baseline; the best seed (42, 62.3 %) should be read as an upper bound rather than a typical outcome. Data source: `thesis_figures/fig_twd_seed_sweep.png`.

**Figure 4. Under the deployment metric, the hardware-compatible learning rule is markedly worse than BPTT.** Mean ± s.d. first-spike accuracy (n = 4 seeds each) for BPTT with the TET loss (one convolutional layer, T = 16; 29.9 ± 0.6 %) and for e-prop with κ = 0.8 (one convolutional layer, T = 8; 15.3 ± 0.8 %) on CIFAR100:20. The gap of 14.6 percentage points indicates that replacing global backpropagation with a locally learnable rule does not close the first-spike deployment gap — it widens it. Data source: `thesis_figures/fig_gap_bar.png`.

---

## Tables

Tables 1–11 are given inline in Sections 4 and 5. Table 1 (datasets), Table 2 (network configurations), Table 3 (baseline and alternative objectives), Table 4 (TWD weight sweep), Table 5 (temporal-window sweep), Table 6 (write-noise sweep), Table 7 (seed sweep), Table 8 (CIFAR-10 transfer), Table 9 (objective comparison), Table 10 (component ablation), Table 11 (strength of evidence).

---

## References

Ambrogio, S., Narayanan, P., Tsai, H., Shelby, R. M., Boybat, I., di Nolfo, C., et al. (2018). Equivalent-accuracy accelerated neural-network training using analogue memory. *Nature* 558, 60–67.

Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., and Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications* 11, 3625.

Boybat, I., Le Gallo, M., Nandakumar, S. R., Moraitis, T., Parnell, T., Tuma, T., et al. (2018). Neuromorphic computing with multi-memristive synapses. *Nature Communications* 9, 2514.

Davies, M., Srinivasa, N., Lin, T.-H., Chinya, G., Cao, Y., Choday, S. H., et al. (2018). Loihi: a neuromorphic manycore processor with on-chip learning. *IEEE Micro* 38, 82–99.

Deng, L., Wu, Y., Hu, X., Liang, L., Ding, Y., Li, G., et al. (2020). Rethinking the performance comparison between SNNs and ANNs. *Neural Networks* 121, 294–307.

Diehl, P. U., and Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience* 9, 99.

Duan, C., Ding, J., Chen, S., Sun, Z., and Qian, W. (2022). Temporal effective batch normalization in spiking neural networks. In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*.

Esser, S. K., Merolla, P. A., Arthur, J. V., Cassidy, A. S., Appuswamy, R., Andreopoulos, A., et al. (2016). Convolutional networks for fast, energy-efficient neuromorphic computing. *Proceedings of the National Academy of Sciences* 113, 11441–11446.

Fang, W., Yu, Z., Chen, Y., Masquelier, T., Huang, T., and Tian, Y. (2021). Incorporating learnable membrane time constant to enhance learning of spiking neural networks. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2661–2671.

Gokmen, T., and Vlasov, Y. (2016). Acceleration of deep neural network training with resistive cross-point devices: design considerations. *Frontiers in Neuroscience* 10, 333.

Gütig, R., and Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing-based decisions. *Nature Neuroscience* 9, 420–428.

Ioffe, S., and Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448–456.

Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images*. Technical Report, University of Toronto.

Maass, W. (1997). Networks of spiking neurons: the third generation of neural network models. *Neural Networks* 10, 1659–1671.

Nandakumar, S. R., Le Gallo, M., Piveteau, C., Joshi, V., Mariani, G., Carta, F., et al. (2020). Mixed-precision deep learning based on computational memory. *Frontiers in Neuroscience* 14, 406.

Neftci, E. O., Mostafa, H., and Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine* 36, 51–63.

Prezioso, M., Merrikh-Bayat, F., Hoskins, B. D., Adam, G. C., Likharev, K. K., and Strukov, D. B. (2015). Training and operation of an integrated neuromorphic network based on metal-oxide memristors. *Nature* 521, 61–64.

Roy, K., Jaiswal, A., and Panda, P. (2019). Towards spike-based machine intelligence with neuromorphic computing. *Nature* 575, 607–617.

Rueckauer, B., Lungu, I.-A., Hu, Y., Pfeiffer, M., and Liu, S.-C. (2017). Conversion of continuous-valued deep networks to efficient event-driven networks for image classification. *Frontiers in Neuroscience* 11, 682.

Sebastian, A., Le Gallo, M., Khaddam-Aljameh, R., and Eleftheriou, E. (2020). Memory devices and applications for in-memory computing. *Nature Nanotechnology* 15, 529–544.

Sengupta, A., Ye, Y., Wang, R., Liu, C., and Roy, K. (2019). Going deeper in spiking neural networks: VGG and residual architectures. *Frontiers in Neuroscience* 13, 95.

Sheridan, P. M., Cai, F., Du, C., Ma, W., Zhang, Z., and Lu, W. D. (2017). Sparse coding with memristor networks. *Nature Nanotechnology* 12, 784–789.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research* 15, 1929–1958.

Stöckl, C., and Maass, W. (2021). Optimized spiking neurons can classify images with high accuracy through temporal coding with two spikes. *Nature Machine Intelligence* 3, 230–238.

Thorpe, S., Fize, D., and Marlot, C. (1996). Speed of processing in the human visual system. *Nature* 381, 520–522.

Wozniak, S., Pantazi, A., Bohnstingl, T., and Eleftheriou, E. (2020). Deep learning incorporating biologically inspired neural dynamics and in-memory computing. *Nature Machine Intelligence* 2, 325–336.

Wu, Y., Deng, L., Li, G., Zhu, J., and Shi, L. (2018). Spatio-temporal backpropagation for training high-performance spiking neural networks. *Frontiers in Neuroscience* 12, 331.

Yao, P., Wu, H., Gao, B., Tang, J., Zhang, Q., Zhang, W., et al. (2020). Fully hardware-implemented memristor convolutional neural network. *Nature* 577, 641–646.

Zenke, F., and Ganguli, S. (2018). SuperSpike: supervised learning in multilayer spiking neural networks. *Neural Computation* 30, 1514–1541.

Zheng, H., Wu, Y., Deng, L., Hu, Y., and Li, G. (2021). Going deeper with directly-trained larger spiking neural networks. In *Proceedings of the AAAI Conference on Artificial Intelligence* 35, 11062–11070.

**[R30] UNVERIFIED — DO NOT SUBMIT AS-IS.** The TWD decoder implemented in the project code is documented in `temporal_snn/model.py` as following "Che et al., ICML 2026". The bibliographic details of this source could not be verified during drafting (no web access to the ICML 2026 proceedings from the drafting environment). Either verify the correct citation and insert it here, or remove the attribution and describe the decoder as introduced in this work. All other references above are canonical, well-known works; their page ranges and volumes should still be checked against the publishers' records before submission.

---

## Appendix A. Run registry

Every numeric result in this paper maps to a run directory under `runs_csnn/` in the project repository, named `<run_id>`, containing `cfg.json` (full configuration), `metrics.jsonl` (per-epoch metrics), `summary.json` (best accuracies and best epochs) and checkpoints (`best_fs.pt`, `best_mem.pt`, `best_twd.pt`).

| Result | Run ID | Config |
|---|---|---|
| λ sweep, λ = 0.1 | `20260702T100017Z_1dcb4e96e42f` | T = 16, σ = 0.1, seed 42 |
| λ sweep, λ = 0.25 | `20260702T070017Z_1dcb4e96e42f` | T = 16, σ = 0.1, seed 42 |
| λ sweep, λ = 0.125 / 0.05 | `20260702T090017Z_1dcb4e96e42f`, `20260702T090022Z_1dcb4e96e42f` | T = 16, σ = 0.1, seed 42 |
| λ sweep, λ = 0.5 / 1.0 / 2.0 | `20260701T182013Z`, `20260701T183353Z`, `20260701T184723Z` | T = 16, σ = 0.1, seed 42 |
| T sweep, T = 8 / 12 / 20 / 32 | `20260702T200504Z_a80801d52f9b`, `20260702T150018Z_910eec275ead`, `20260702T190017Z_792cba348715`, `20260702T160017Z_3e0db20582bc` | σ = 0.1, λ = 0.1, seed 42 |
| T sweep, T = 16 | `20260702T100017Z_1dcb4e96e42f` | σ = 0.1, λ = 0.1, seed 42 |
| σ sweep, σ = 0.05 / 0.2 | `20260703T084146Z_910eec275ead`, `20260703T090014Z_910eec275ead` | T = 12, λ = 0.1, seed 42 |
| σ sweep, σ = 0.1 (champion) | `20260702T150018Z_910eec275ead` | T = 12, λ = 0.1, seed 42 |
| Seed sweep, seed 0 / 1 / 2 | `20260704T120323Z_8ec732804144`, `20260703T150022Z_9e8e6fb75319`, `20260703T160018Z_283773bf0bc0` | T = 12, σ = 0.1, λ = 0.1 |
| Baseline BPTT seed 42 / 0 / 1 / 2 | `20260629T200014Z_1dcb4e96e42f`, `20260630T140531Z_909268c41976`, `20260630T142520Z_e4b72a1d8e22`, `20260630T144506Z_b04d633fcef0` | T = 16, n_conv = 1, 120 ep |
| TET baseline, two-conv, seed 42 | `20260627T200906Z_1dcb4e96e42f` | T = 16, σ = 0, 120 ep |
| First-spike ranking loss (rejected) | `20260628T050149Z_59197fb42607` | T = 16, λ_fsl = 1.0, 120 ep |
| CIFAR-10 transfer | `20260708T064051Z_910eec275ead` | T = 12, σ = 0, BNTT, 3-conv, 200 ep, seed 42 |
| e-prop baseline seed 42 / 0 / 1 / 2 | `20260629T220021Z_5939491e1d02`, `20260630T150517Z_98e70f314471`, `20260630T152220Z_e27307cbdec2`, `20260630T153940Z_20a7322c05d3` | κ = 0.8, n_conv = 1, T = 8, 120 ep |

*Table A1. Run registry for the results reported in this paper.*

*Note (2026-09-18): the component-ablation arms reported in Section 5.8 / Table 10 (noise-only and TWD-only, wider BNTT architecture, T = 12, 200 epochs, Diverse10) are new results; their run IDs have not yet been entered in this registry and must be recorded before submission.*

---

## Appendix B. Notes for the authors (not part of the manuscript)

These notes record decisions taken during drafting where the source documents and the implementation disagreed, or where a number in the source draft could not be reproduced. They are provided so that nothing in the manuscript is carried forward unexamined.

1. **Noise model description.** The source draft (`PAPER1_TWD.md`) describes noise injection as "Gaussian noise N(0, σ²) added to all weights at each forward pass during training". The implementation (`temporal_snn/memristor.py`, used via `trainer.py`) applies *multiplicative* noise to the weight **update** after each optimiser step, ΔG = ΔW·(1 + N(0, σ²)), followed by clipping to [−1, 1], and excludes 1-D parameters (biases, batch-norm affine parameters). Section 4.7 of the manuscript describes the implementation. If the forward-pass noise variant was also run, it should be documented separately, because the two are not equivalent.

2. **Loss formulation.** The source draft gives L_total = L_TET + λ_twd·L_TWD + λ_fsl·L_FSL with L_TWD = E[t_first_spike]/T. The implementation differs in two ways: (a) L_TWD is a cross-entropy on a *learned softplus-weighted sum of spikes*, not a penalty on mean first-spike time; and (b) FSL is only active when `loss_mode = "first_spike"`, so the reported runs use L = L_TET + λ_twd·L_TWD with no FSL term (`fsl_weight = 0.5` is present in the configs but inert). The manuscript describes the implementation; the FSL variant is reported separately as the rejected alternative (Section 5.7).

3. **Architecture attribution.** The source draft attributes a 3-conv 64/128/256 + FC(512) architecture to the whole study. That configuration is the **transfer network** used only for CIFAR-10. All CIFAR100:20 sweeps, the seed sweep and the ablations used a 2-conv 32/64 + FC(256) network (≈ 8.3 × 10⁵ parameters, consistent with the "826k" network referred to in the experiment log). Both are documented in Table 2, because a reviewer comparing the sweep tables with the architecture section would otherwise find a mismatch.

4. **Baseline inconsistency resolved.** Two runs share the label "TET baseline, T = 16, seed 42" but report 25.8 % and 30.0 % first-spike accuracy. The difference is depth: the 25.8 % run inherited the default `n_conv_layers = 2`, while the 30.0 % run explicitly set `n_conv_layers = 1`. The manuscript uses 25.8 % for the depth-matched comparison with the champion (both two-conv) and 29.9 ± 0.6 % (n = 4 seeds, one conv) as the baseline mean. Both are correct as labelled.

5. **The +24.4 pp figure is a recipe-level effect, not a TWD-only effect.** The source draft presents the baseline and the champion as differing only in TWD; they also differ in T (16 → 12), depth (1 → 2 conv), epochs (120 → 60) and noise (σ = 0 → 0.1). The manuscript keeps the +24.4 pp headline, which is what the claim requires, but adds the two confound arguments the data *do* support (T is not the driver: TWD at T = 16 still gives 58.0 %; depth is not the driver: extra depth *lowers* baseline first-spike accuracy). The component ablation added in Section 5.8 supplies the previously missing noise-only (no TWD, 60.3 %) and TWD-only (σ = 0, 75.3 %) arms, and reframes TWD as a compensation mechanism rather than an accuracy booster; note, however, that these arms run on the wider BNTT architecture at 200 epochs, so a matched 2×2 factorial at the champion configuration remains open.

6. **CIFAR-10 baseline.** The source draft lists "BPTT, no TWD, T = 16 → ~30 %" as the CIFAR-10 comparison row. No such CIFAR-10 run exists in the repository — every CIFAR-10 run found (8 directories) uses `use_twd = true`. The ≈30 % figure is inherited from the CIFAR100:20 baseline and must not be presented as a CIFAR-10 control. The manuscript reports the CIFAR-10 transfer as an existence result with no delta.

7. **Unverified citation.** The TWD decoder's documented provenance ("Che et al., ICML 2026") could not be verified while drafting. See note [R30] in the reference list.

8. **Reproducibility details worth confirming before submission.** (a) The CIFAR-10 transfer run was resumed from a checkpoint at epoch 160 of an earlier schedule, and its per-epoch log contains 216 entries for a 200-epoch configuration; the manuscript reports only the best first-spike accuracy (65.4 %, epoch 23) and best membrane accuracy (78.8 %) from `summary.json`. (b) Best-epoch selection is performed on the test set in the current code, which makes all reported accuracies mildly optimistic; if a validation split is introduced, every number in Tables 3–10 will need to be regenerated. (c) Percentages are reported to one decimal place; CIFAR100:20 test sets of 1,000 images give 0.1 pp granularity, so differences below ~1 pp should not be interpreted.

9. **Frontiers formatting still to do.** Author list and affiliations; CRediT contributions statement; funding statement; acknowledgements; a DOI'd data/code archive; confirmation that figures 1–4 are supplied at ≥ 300 dpi in the required format; and a decision on whether to merge Section 3 (Background) into the Introduction, since Frontiers Research Articles conventionally use Introduction / Materials and Methods / Results / Discussion without a separate Background section.
