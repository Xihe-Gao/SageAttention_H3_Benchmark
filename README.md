# SageAttention H3 Benchmark

MiniMax H3 QKV benchmark and numerical cross-validation project for SageAttention.

## Scope

This project evaluates six captured MiniMax H3 QKV samples (`step=[5, 19]`,
`layer=[3, 25, 47]`) with:

- BF16 PyTorch SDPA reference
- `int8_qk_fp8_pv` CUDA kernel
- `fp8_qk_fp8_pv` CUDA kernel
- FP8 CUDA kernel with `smooth_q`, `smooth_v`, and `smooth_qv`
- matching pure-PyTorch SageAttention simulations

The CUDA paths use per-thread Q/K quantization and FP32 PV accumulation. The
simulation is intentionally independent of the CUDA implementation so that the
two outputs can be cross-validated.

## Files

- `docs/SageAttention_QKV_Benchmark报告.md` — Chinese benchmark report
- `docs/Official_INT8_FP8_CUDA_Kernel_Walkthrough.md` — Python API → C++ binding → CUDA launcher → kernel source walkthrough
- `scripts/benchmark_sageattention.py` — CUDA kernel benchmark
- `scripts/benchmark_sageattention_sim.py` — full-length simulation benchmark
- `scripts/cross_validate_sim_kernel.py` — direct simulation-vs-kernel comparison
- `scripts/generate_benchmark_report.py` — report generator
- `config/capture_config.json`, `config/workflow_api.json`, `config/prompt.txt` — capture metadata
- `third_party/SageAttention` — SageAttention CUDA/Triton implementation (pinned to `feature/qk-fp8-pv-fp8`)
- `third_party/SageAttention_Simulation` — independent PyTorch simulation implementation
- `results/sageattention_benchmark_results.json` — measured results

The large `data/captures/` directory contains the raw Q/K/V tensors and is excluded
from Git by design. Store it separately when reproducing the benchmark.

## Dependencies

This repository pins the tested implementation and simulation as Git submodules.
Clone with submodules enabled:

```bash
git clone --recurse-submodules https://github.com/Xihe-Gao/SageAttention_H3_Benchmark.git
```

For an existing checkout, initialize them with:

```bash
git submodule update --init --recursive
```

The submodules are pinned to specific commits for reproducibility. To deliberately
refresh them to their configured remote branches, run `git submodule update --remote`
and commit the resulting submodule pointer changes.

## Step-by-step reproduction

The recommended order follows the report: acquire QKV data first, then run the
independent FP8 simulation, then benchmark the compiled CUDA operators, and
finally compare simulation outputs with matching operator outputs.

### 1. Acquire QKV data from ComfyUI

Use the official ComfyUI `MiniMax H3: Text to Video` workflow with the model
files listed above. Set the capture configuration in `config/capture_config.json`:

```json
{
  "output_dir": "/path/to/SageAttention_H3_Benchmark/data/captures",
  "steps": [5, 19],
  "layers": [3, 25, 47]
}
```

Point ComfyUI at this file and run the workflow:

```bash
export COMFYUI_QKV_CAPTURE_CONFIG=/path/to/SageAttention_H3_Benchmark/config/capture_config.json
```

The capture hook saves `q.pt`, `k.pt`, `v.pt`, and `metadata.json` under one
directory per `(step, layer)`. The tensors are the post-RMSNorm/RoPE Q/K and V
attention inputs, saved as BF16 in `B,H,S,D` layout. Copy or mount those
directories into this project's `data/captures/` directory before benchmarking.

The raw `data/captures/` directory is intentionally excluded from Git because it is
large. The workflow and prompt used for the reference capture are retained as
`config/workflow_api.json` and `config/prompt.txt`.

### 2. Run the FP8 algorithm simulation

The simulation is a pure-PyTorch implementation in
`third_party/SageAttention_Simulation/h3_fp8_smoothq_sim`. It independently performs
per-thread Q/K quantization, per-channel V quantization, tiled online softmax,
FP8 probability conversion, and FP32 PV accumulation.

From this project directory:

```bash
PYTHONPATH=/path/to/SageAttention_H3_Benchmark/third_party/SageAttention \
  python scripts/benchmark_sageattention_sim.py
```

This runs each of the five simulation configurations once for every captured
sample and appends their timing and relative-L2-to-BF16 values to
`results/sageattention_benchmark_results.json`.

### 3. Run the FP8 CUDA operator benchmark

Build/install the SageAttention checkout first, including the CUDA extension
for the current GPU. The tested operator configuration is per-thread Q/K
quantization with FP32 PV accumulation:

```bash
cd /path/to/SageAttention_H3_Benchmark/third_party/SageAttention
python setup.py build_ext --inplace
cd /path/to/SageAttention_H3_Benchmark
PYTHONPATH=/path/to/SageAttention_H3_Benchmark/third_party/SageAttention \
  python scripts/benchmark_sageattention.py
```

The benchmark uses CUDA Events, excludes disk I/O and CPU-to-GPU loading, and
records BF16 SDPA plus the INT8/FP8 CUDA paths (including smooth-q/v variants).
The resulting per-sample timing and relative L2 values are written to
`results/sageattention_benchmark_results.json`.

### 4. Compare simulation outputs with CUDA outputs

Run the direct cross-validation after both sets of results are available:

```bash
PYTHONPATH=/path/to/SageAttention_H3_Benchmark/third_party/SageAttention \
  python scripts/cross_validate_sim_kernel.py
```

This computes `||simulation - kernel||₂ / ||kernel||₂` for every sample and
matching configuration, and stores the values in the same JSON file. It does
not infer this metric from the two BF16 errors; it compares the output tensors
directly.

### 5. Generate the Markdown report

```bash
python scripts/generate_benchmark_report.py
```

The generated `docs/SageAttention_QKV_Benchmark报告.md` contains the data-acquisition
description, simulation and operator implementation notes, test methodology,
the per-method results, and the simulation-vs-operator cross-validation table.

The captured tensors are BF16, layout `B,H,S,D`, shape `[1, 56, 73923, 128]`.
