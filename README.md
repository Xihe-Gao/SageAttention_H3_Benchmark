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

- `SageAttention_QKV_Benchmark报告.md` — Chinese benchmark report
- `benchmark_sageattention.py` — CUDA kernel benchmark
- `benchmark_sageattention_sim.py` — full-length simulation benchmark
- `cross_validate_sim_kernel.py` — direct simulation-vs-kernel comparison
- `generate_benchmark_report.py` — report generator
- `capture_config.json`, `workflow_api.json`, `prompt.txt` — capture metadata
- `sageattention_benchmark_results.json` — measured results

The large `captures/` directory contains the raw Q/K/V tensors and is excluded
from Git by design. Store it separately when reproducing the benchmark.

## Reproduction

Run inside the SageAttention checkout with the project on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/SageAttention python benchmark_sageattention.py
PYTHONPATH=/path/to/SageAttention python benchmark_sageattention_sim.py
PYTHONPATH=/path/to/SageAttention python cross_validate_sim_kernel.py
python generate_benchmark_report.py
```

The captured tensors are BF16, layout `B,H,S,D`, shape `[1, 56, 73923, 128]`.
See the report for hardware, timing semantics, quantization details, and error
definitions.
