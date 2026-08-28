#!/venv/main/bin/python
"""Directly compare SageAttention_Sim outputs with matching CUDA kernels."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/qkv_minimax_h3")
CAPTURE_ROOT = ROOT / "captures"
DEFAULT_RESULTS = ROOT / "sageattention_benchmark_results.json"
SIM_ROOT = Path(__file__).resolve().parent / "third_party/SageAttention_Simulation/h3_fp8_smoothq_sim"
sys.path.insert(0, str(SIM_ROOT))

from sim import sim_attention  # noqa: E402
from sageattention import (  # noqa: E402
    sageattn_qk_fp8_pv_fp8_cuda,
    sageattn_qk_int8_pv_fp8_cuda,
)


PAIRS = [
    ("int8_qk_fp8_pv", "sim_int8_qk_fp8_pv", "int8", False, False),
    ("fp8_qk_fp8_pv", "sim_fp8_qk_fp8_pv", "fp8", False, False),
    ("fp8_qk_fp8_pv_smooth_q", "sim_fp8_qk_fp8_pv_smooth_q", "fp8", True, False),
    ("fp8_qk_fp8_pv_smooth_v", "sim_fp8_qk_fp8_pv_smooth_v", "fp8", False, True),
    ("fp8_qk_fp8_pv_smooth_qv", "sim_fp8_qk_fp8_pv_smooth_qv", "fp8", True, True),
]


def load_tensor(directory: Path, name: str) -> torch.Tensor:
    return torch.load(
        directory / f"{name}.pt", map_location="cpu", weights_only=True, mmap=True
    ).to("cuda").contiguous()


def relative_l2(output: torch.Tensor, reference: torch.Tensor, chunk: int = 2048) -> float:
    numerator = 0.0
    denominator = 0.0
    for begin in range(0, output.shape[2], chunk):
        end = min(begin + chunk, output.shape[2])
        out_part = output[:, :, begin:end, :].float()
        ref_part = reference[:, :, begin:end, :].float()
        diff = out_part - ref_part
        numerator += float(torch.sum(diff * diff, dtype=torch.float64).item())
        denominator += float(torch.sum(ref_part * ref_part, dtype=torch.float64).item())
    return math.sqrt(numerator / denominator)


def kernel_output(q, k, v, mode, smooth_q, smooth_v):
    if mode == "int8":
        return sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, tensor_layout="HND", is_causal=False,
            qk_quant_gran="per_thread", pv_accum_dtype="fp32",
            smooth_k=True, smooth_v=False, return_lse=False,
        )
    return sageattn_qk_fp8_pv_fp8_cuda(
        q, k, v, tensor_layout="HND", is_causal=False,
        qk_quant_gran="per_thread", pv_accum_dtype="fp32",
        smooth_k=True, smooth_q=smooth_q, smooth_v=smooth_v,
        return_lse=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cta-k", type=int, default=64)
    parser.add_argument("--block-q", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    document = json.loads(args.results.read_text())
    cross = document.setdefault("simulation_kernel_cross_validation", {})
    cross.update({
        "metric": "||simulation.float-kernel.float||_2 / ||kernel.float||_2",
        "reference": "matching compiled CUDA kernel output",
        "samples": cross.get("samples", {}),
    })

    for sample in document["samples"]:
        name = sample["sample"]
        sample_results = cross["samples"].setdefault(name, {})
        q, k, v = (load_tensor(CAPTURE_ROOT / name, item) for item in ("q", "k", "v"))
        for kernel_name, sim_name, mode, smooth_q, smooth_v in PAIRS:
            if sim_name in sample_results and not args.force:
                print(f"[{name}] {sim_name}: already present, skip", flush=True)
                continue
            print(f"[{name}] {sim_name} vs {kernel_name}", flush=True)
            kernel = kernel_output(q, k, v, mode, smooth_q, smooth_v)
            simulation = sim_attention(
                q, k, v, q.shape[-1] ** -0.5, mode=mode,
                smooth_q=smooth_q, smooth_v=smooth_v,
                cta_k=args.cta_k, block_q=args.block_q,
            )
            torch.cuda.synchronize()
            error = relative_l2(simulation, kernel)
            sample_results[sim_name] = {
                "simulation_method": sim_name,
                "kernel_method": kernel_name,
                "relative_l2": error,
            }
            args.results.write_text(json.dumps(document, indent=2), encoding="utf-8")
            print(f"  sim_vs_kernel_relative_l2={error:.8f}", flush=True)
            del simulation, kernel
            gc.collect()
            torch.cuda.empty_cache()
        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()

    args.results.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"Wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
