# TPU-Accelerated Neuromorphic Scaling via Composite Learning Rules

**Author:** Chidubem Ntiwunka-Ifeanyi  
**Faculty Supervisor:** Dr. Onyema Osuagwu  
**License:** Apache 2.0  

---

## Overview

This repository contains the codebase for evaluating energy-information scaling laws on synchronous systolic hardware. The primary objective is to bypass the `O(L × N × T)` memory bottleneck inherent in Backpropagation Through Time (BPTT) by mapping a composite learning rule (STDP + Direct Feedback Alignment + e-prop) onto standard hardware accelerators, specifically targeting Google Cloud TPU Matrix Multiply Units (MXUs).

While mapping these asynchronous, event-driven networks to custom logic on FPGAs represents a long-term trajectory for edge deployments, the current project scope focuses strictly on optimization for standard systolic arrays.

---

## Timeline

- **Phase 1 (Complete):** Implementation of a C++ crossbar simulator baseline utilizing the Spiking Heidelberg Digits (SHD) dataset, demonstrating convergence without temporally unrolled computation graphs. Adaptive LIF (ALIF) neuron dynamics are implemented in fixed-point Q8.8 arithmetic to mirror hardware register/SRAM constraints. Initial RTL logic exploration for hardware mapping is also present.
  
  - *Validation:* The simulator successfully converges on the SHD dataset, proving the algorithmic validity of the DFA/E-prop composite rule without needing to store `T` timesteps in memory.

- **Phase 2 (Actively In Progress):** Translating the simulation engine into a JAX/XLA-compliant framework.
  
  - *Complete:* A single-timestep LIF/e-prop update kernel (`@jax.jit`-compiled) is implemented (`jax_translation/snn_core.py`) and numerically mirrors the C++ baseline's neuron dynamics and eligibility trace update (exact parity achieved).
  
  - *Next Steps (Current Focus):* Folding this kernel into a `jax.lax.scan` loop for full temporal compilation (`T = 500`), and batching via `jax.vmap`. This is the final step before benchmarking against the LSTM baseline.

- **Phase 3 (Scheduled):** Scaling to multi-class (2048 hidden units), profiling MXU utilization vs. BPTT.

- **Phase 4 (Scheduled – Profiling & Publication):** Profile MXU utilization vs. BPTT baseline; draft findings for submission.

---

## Repository Structure

- `src/` – Core C++ headers (ALIF neuron, e-prop traces, Q8.8 math).
- `jax_translation/` – **Early-stage JAX port.** Contains `snn_core.py` (single-step compiled kernel) and placeholder for the `lax.scan` temporal loop.
- `scripts/` – Python utilities for SHD preprocessing and plotting.
- `tests/` – C++ test suite (mathematical parity, memory profiling benchmarks).
- `plots/` – Generated visualizations of network performance.

---

## Benchmarking Baseline

The mathematical and hardware efficiency of this framework will be evaluated against a standard 2-layer Long Short-Term Memory (LSTM) network containing 256 hidden units per layer, trained via BPTT over 500 timesteps. The primary comparison metric is the memory footprint (`O(1)` vs. `O(L × N × T)`).

---

## Hardware and Software Prerequisites

- Google Cloud TPU environment (v4 or later recommended)
- Python 3.10+ (JAX & JAXlib TPU-compatible build)
- C++17 Compiler (GCC/Clang) & CMake
