# TPU-Accelerated Neuromorphic Scaling via Composite Learning Rules
**Author:** Chidubem Ntiwunka-Ifeanyi  
**Faculty Supervisor:** Dr. Onyema Osuagwu  
**License:** Apache 2.0  
## Overview
This repository contains the codebase for evaluating energy-information scaling laws on synchronous systolic hardware. The primary objective is to bypass the O(L x N x T) memory bottleneck inherent in Backpropagation Through Time (BPTT) by mapping a composite learning rule (STDP + Direct Feedback Alignment + e-prop) onto standard hardware accelerators, specifically targeting Google Cloud TPU Matrix Multiply Units (MXUs).
While mapping these asynchronous, event-driven networks to custom logic on FPGAs represents a long-term trajectory for edge deployments, the current project scope focuses strictly on optimization for standard systolic arrays.
## Progress Update
* **Phase 1 (Complete):** Implementation of a C++ crossbar simulator baseline utilizing the Spiking Heidelberg Digits (SHD) dataset, demonstrating convergence without temporally unrolled computation graphs. Adaptive LIF (ALIF) neuron dynamics are implemented in fixed-point Q8.8 arithmetic to mirror hardware register/SRAM constraints. Initial RTL logic exploration for hardware mapping is also present.
* **Phase 2 (In Progress):** Translating the simulation engine into a JAX/XLA-compliant framework. A single-timestep LIF/e-prop update kernel (`@jax.jit`-compiled) is implemented and numerically mirrors the C++ baseline's neuron dynamics and eligibility trace update. Folding this kernel into a `jax.lax.scan` loop for full temporal compilation, and batching via `jax.vmap`, are the immediate next steps and are not yet implemented.
## Repository Structure
The repository is organized to cleanly separate the core neural dynamics definitions, hardware simulation logic, data ingestion, and testing suites.
* `.github/workflows/`
  * Contains `build.yml` for automated CI/CD C++ compilation and test execution.
* `src/`
  * Contains the core C++ headers defining the localized update math: `alif_neuron.h` (adaptive LIF), `eprop_trace.h` (eligibility trace), `snn_network.h` (network assembly), `q8_8_math.h` (fixed-point arithmetic), `data_pipeline.h` (input handling), and `telemetry_logger.h`.
  * `jax_translation/snn_core.py` — Early-stage JAX port of the LIF + e-prop update rule. Currently a single-step compiled function.
* `scripts/`
  * Python utilities for SHD dataset preprocessing (`preprocess_shd.py`) and result visualization (`plot_results.py`).
* `data/`
  * Not tracked in version control. Target directory for raw SHD HDF5 files and the generated bit-packed binary input tensors.
* `tests/`
  * C++ test suite validating mathematical parity and performance: `run_shd.cpp`, `test_alif.cpp`, `test_eprop.cpp`, `test_math.cpp`, `test_encoder.cpp`, `test_network.cpp`, `test_integration.cpp`.
  * Includes `test_benchmarks.cpp` and `test_memory_profiling.cpp` for performance characterization, directly supporting the project's memory-bottleneck thesis.
* `plots/`
  * Generated visualizations of network performance, including membrane voltage telemetry and weight convergence data.
## Benchmarking Baseline
The mathematical and hardware efficiency of this framework are evaluated against a standard 2-layer Long Short-Term Memory (LSTM) network containing 256 hidden units per layer, trained via BPTT over 500 timesteps.
## Hardware and Software Prerequisites
* Google Cloud TPU environment (v4 or later recommended)
* Python 3.10+ (JAX & JAXlib TPU-compatible build)
* C++17 Compiler (GCC/Clang) & CMake
* Verilator or standard HDL simulator (for `.sv` files)
