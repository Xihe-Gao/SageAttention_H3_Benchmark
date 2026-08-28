#!/venv/main/bin/python
"""Append full-length SageAttention_Sim results to the existing kernel JSON."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/qkv_minimax_h3")
CAPTURE_ROOT = ROOT / "captures"
DEFAULT_RESULTS = ROOT / "sageattention_benchmark_results.json"
SIM_ROOT = Path("/workspace/SageAttention_Sim/sageattention/h3_fp8_smoothq_sim")
sys.path.insert(0, str(SIM_ROOT))

from sim import sim_attention  # noqa: E402


SIM_METHODS = [
    ("sim_int8_qk_fp8_pv", dict(mode="int8", smooth_q=False, smooth_v=False)),
    ("sim_fp8_qk_fp8_pv", dict(mode="fp8", smooth_q=False, smooth_v=False)),
    ("sim_fp8_qk_fp8_pv_smooth_q", dict(mode="fp8", smooth_q=True, smooth_v=False)),
    ("sim_fp8_qk_fp8_pv_smooth_v", dict(mode="fp8", smooth_q=False, smooth_v=True)),
    ("sim_fp8_qk_fp8_pv_smooth_qv", dict(mode="fp8", smooth_q=True, smooth_v=True)),
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


def one_cuda_timing(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    return output, float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cta-k", type=int, default=64)
    parser.add_argument("--block-q", type=int, default=2048)
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    document = json.loads(args.results.read_text())
    wanted = set(args.samples) if args.samples else None
    document["simulation"] = {
        "implementation": str(SIM_ROOT),
        "description": "faithful tiled pure-PyTorch SageAttention simulation",
        "cta_k": args.cta_k,
        "block_q": args.block_q,
        "warmup": 0,
        "repeats": 1,
        "timing": "CUDA Events; one full end-to-end simulation run",
    }

    for sample in document["samples"]:
        sample_name = sample["sample"]
        if wanted is not None and sample_name not in wanted:
            continue
        directory = CAPTURE_ROOT / sample_name
        q, k, v = (load_tensor(directory, name) for name in ("q", "k", "v"))
        reference = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        torch.cuda.synchronize()

        for method_name, kwargs in SIM_METHODS:
            if method_name in sample["methods"] and not args.force:
                print(f"[{sample_name}] {method_name}: already present, skip", flush=True)
                continue
            print(f"[{sample_name}] {method_name}: full simulation", flush=True)
            output, elapsed = one_cuda_timing(
                lambda kwargs=kwargs: sim_attention(
                    q, k, v, q.shape[-1] ** -0.5,
                    cta_k=args.cta_k, block_q=args.block_q, **kwargs,
                )
            )
            assert output.dtype == torch.bfloat16
            assert torch.isfinite(output).all().item()
            error = relative_l2(output, reference)
            sample["methods"][method_name] = {
                "mean_ms": elapsed,
                "median_ms": elapsed,
                "min_ms": elapsed,
                "max_ms": elapsed,
                "stdev_ms": 0.0,
                "times_ms": [elapsed],
                "relative_l2": error,
                "measurement_kind": "single_full_simulation_run",
            }
            args.results.write_text(json.dumps(document, indent=2), encoding="utf-8")
            print(f"  time={elapsed:.3f} ms, rel_l2={error:.8f}", flush=True)
            del output
            gc.collect()
            torch.cuda.empty_cache()

        del reference, q, k, v
        gc.collect()
        torch.cuda.empty_cache()

    args.results.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"Wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
