#!/venv/main/bin/python
import argparse
import gc
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from sageattention import (
    sageattn_qk_int8_pv_fp8_cuda,
    sageattn_qk_fp8_pv_fp8_cuda,
)

CAPTURE_ROOT = Path(__file__).resolve().parents[1] / "data/captures"
RESULT_PATH = Path(__file__).resolve().parents[1] / "results/sageattention_benchmark_results.json"
EXPECTED = [
    'step_05_layer_03', 'step_05_layer_25', 'step_05_layer_47',
    'step_19_layer_03', 'step_19_layer_25', 'step_19_layer_47',
]


def cuda_time_ms(fn, warmup: int, repeats: int):
    out = None
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()

    times = []
    last = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        if last is not None:
            del last
        last = out
    return last, times


def relative_l2(output: torch.Tensor, reference: torch.Tensor, chunk: int = 2048):
    # Chunk over sequence to keep peak FP32 scratch memory bounded.
    numerator = 0.0
    denominator = 0.0
    seq = output.shape[2]
    for begin in range(0, seq, chunk):
        end = min(begin + chunk, seq)
        out_part = output[:, :, begin:end, :].float()
        ref_part = reference[:, :, begin:end, :].float()
        diff = out_part - ref_part
        numerator += float(torch.sum(diff * diff, dtype=torch.float64).item())
        denominator += float(torch.sum(ref_part * ref_part, dtype=torch.float64).item())
        del out_part, ref_part, diff
    return math.sqrt(numerator / denominator)


def summarize_times(times):
    ordered = sorted(times)
    return {
        'mean_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
        'times_ms': times,
    }


def load_capture(name, seq_limit=None):
    directory = CAPTURE_ROOT / name
    tensors = []
    for item in ('q', 'k', 'v'):
        tensor = torch.load(
            directory / f'{item}.pt', map_location='cpu',
            weights_only=True, mmap=True,
        )
        if seq_limit is not None:
            tensor = tensor[:, :, :seq_limit, :]
        tensors.append(tensor.to('cuda').contiguous())
    metadata = json.loads((directory / 'metadata.json').read_text())
    return (*tensors, metadata)


def benchmark_sample(name, warmup, repeats, seq_limit=None):
    q, k, v, metadata = load_capture(name, seq_limit)
    torch.cuda.synchronize()

    methods = [
        (
            'bf16_sdpa',
            lambda: F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            ),
        ),
        (
            'int8_qk_fp8_pv',
            lambda: sageattn_qk_int8_pv_fp8_cuda(
                q, k, v, tensor_layout='HND', is_causal=False,
                qk_quant_gran='per_thread', pv_accum_dtype='fp32',
                smooth_k=True, smooth_v=False, return_lse=False,
            ),
        ),
        (
            'fp8_qk_fp8_pv',
            lambda: sageattn_qk_fp8_pv_fp8_cuda(
                q, k, v, tensor_layout='HND', is_causal=False,
                qk_quant_gran='per_thread', pv_accum_dtype='fp32',
                smooth_q=False, smooth_k=True, smooth_v=False,
                return_lse=False,
            ),
        ),
        (
            'fp8_qk_fp8_pv_smooth_q',
            lambda: sageattn_qk_fp8_pv_fp8_cuda(
                q, k, v, tensor_layout='HND', is_causal=False,
                qk_quant_gran='per_thread', pv_accum_dtype='fp32',
                smooth_q=True, smooth_k=True, smooth_v=False,
                return_lse=False,
            ),
        ),
        (
            'fp8_qk_fp8_pv_smooth_v',
            lambda: sageattn_qk_fp8_pv_fp8_cuda(
                q, k, v, tensor_layout='HND', is_causal=False,
                qk_quant_gran='per_thread', pv_accum_dtype='fp32',
                smooth_q=False, smooth_k=True, smooth_v=True,
                return_lse=False,
            ),
        ),
        (
            'fp8_qk_fp8_pv_smooth_qv',
            lambda: sageattn_qk_fp8_pv_fp8_cuda(
                q, k, v, tensor_layout='HND', is_causal=False,
                qk_quant_gran='per_thread', pv_accum_dtype='fp32',
                smooth_q=True, smooth_k=True, smooth_v=True,
                return_lse=False,
            ),
        ),
    ]

    results = {}
    reference = None
    for method_name, fn in methods:
        print(f'[{name}] {method_name}: warmup={warmup}, repeats={repeats}', flush=True)
        output, times = cuda_time_ms(fn, warmup, repeats)
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all().item()
        timing = summarize_times(times)
        if method_name == 'bf16_sdpa':
            reference = output
            error = 0.0
        else:
            error = relative_l2(output, reference)
            del output
        results[method_name] = {
            **timing,
            'relative_l2': error,
        }
        print(
            f'  mean={timing["mean_ms"]:.3f} ms, '
            f'median={timing["median_ms"]:.3f} ms, rel_l2={error:.8f}',
            flush=True,
        )
        torch.cuda.empty_cache()

    del reference, q, k, v
    gc.collect()
    torch.cuda.empty_cache()
    return {
        'sample': name,
        'step': metadata['step'],
        'layer': metadata['layer'],
        'sigma': metadata['sigma'],
        'shape': [1, 56, seq_limit or metadata['shape'][2], 128],
        'dtype': 'torch.bfloat16',
        'methods': results,
    }


def base_document(warmup, repeats, seq_limit):
    return {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'benchmark_scope': 'end-to-end GPU kernels; excludes disk I/O and CPU-to-GPU load',
        'timing': 'CUDA Events; warmup excluded; arithmetic mean over repeats',
        'reference': 'torch.nn.functional.scaled_dot_product_attention BF16 output',
        'relative_l2': '||candidate.float-reference.float||_2 / ||reference.float||_2',
        'warmup': warmup,
        'repeats': repeats,
        'seq_limit': seq_limit,
        'environment': {
            'gpu': torch.cuda.get_device_name(),
            'compute_capability': list(torch.cuda.get_device_capability()),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'python': platform.python_version(),
            'sageattention_repo': '/workspace/SageAttention',
            'sageattention_commit': 'd1a57a546c3d395b1ffcbeecc66d81db76f3b4b5',
            'sageattention_branch': 'feature/qk-fp8-pv-fp8',
        },
        'samples': [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seq-limit', type=int)
    parser.add_argument('--samples', nargs='*', default=EXPECTED)
    parser.add_argument('--output', type=Path, default=RESULT_PATH)
    args = parser.parse_args()

    document = base_document(args.warmup, args.repeats, args.seq_limit)
    for name in args.samples:
        started = time.time()
        sample = benchmark_sample(name, args.warmup, args.repeats, args.seq_limit)
        sample['wall_seconds'] = time.time() - started
        document['samples'].append(sample)
        args.output.write_text(json.dumps(document, indent=2), encoding='utf-8')
        print(f'Completed {name} in {sample["wall_seconds"]:.1f}s', flush=True)

    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
