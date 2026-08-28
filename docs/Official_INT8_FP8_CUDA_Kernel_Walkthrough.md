# 官方 INT8 QK + FP8 PV CUDA Kernel 实现说明

本文是官方 SageAttention INT8 QK + FP8 PV CUDA kernel 的源码阅读指南 for the CUDA implementation of INT8 QK and FP8 PV attention in SageAttention2/SageAttention2++.

It follows the path used by the high-level `sageattn` API on an SM120 GPU. 实现源码位于 SM89 文件路径下；在 SM120 GPU 上，`setup.py` 会将相同源码重新编译为 SM120 机器码。

源码均来自 benchmark 仓库的 `third_party/SageAttention` submodule；文中的相对路径均相对于该目录。

## 1. Scope and fixed configuration

The implementation contains many compile-time branches. Start with one fixed configuration:

| Option | Value |
| --- | --- |
| GPU | SM120 |
| Input | BF16 |
| Layout | NHD |
| Head dimension | 128 |
| Attention | Non-causal |
| QK quantization | Per-thread |
| QK computation | INT8 × INT8 → INT32 |
| PV computation | E4M3 × E4M3 |
| PV accumulation | FP32 long-term + FP32 short-term |
| Output | BF16 |
| Return LSE | False |

这对应于 SM120 自动选择 in `sageattention/core.py`:

```python
sageattn_qk_int8_pv_fp8_cuda(
    q,
    k,
    v,
    tensor_layout="NHD",
    is_causal=False,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32+fp32",
)
```

Ignore other template branches during the first reading pass.

## 2. End-to-end computation

The full computation can be summarized as:

```text
BF16 Q ── per-thread quantization ──> INT8 Q + FP32 q_scale
BF16 K ── smoothing/quantization ─> INT8 K + FP32 k_scale
BF16 V ── transpose/quantization ─> E4M3 V + FP32 v_scale

INT8 Q × INT8 K^T
       │
       └─> INT32 score fragments
              │
              ├─ dequantize with q_scale × k_scale
              ├─ apply softmax scale and masks
              └─ online softmax
                       │
                       └─ probabilities converted to E4M3
                                  │
                                  └─ E4M3 P × E4M3 V
                                             │
                                             ├─ FP16 short-term accumulation
                                             ├─ FP32 long-term buffer
                                             ├─ normalize by softmax denominator
                                             ├─ apply v_scale
                                             └─ BF16 output
```

Mathematically, the target operation is:

```text
S = Q K^T / sqrt(head_dim)
P = softmax(S)
O = P V
```

The kernel approximates both matrix multiplications using quantized operands while retaining FP32 state for online-softmax normalization and long-term PV accumulation.

## 3. Source map

Read the files in this order:

| Step | File | Important symbol |
| ---: | --- | --- |
| 1 | `sageattention/core.py` | `sageattn` |
| 2 | `sageattention/core.py` | `sageattn_qk_int8_pv_fp8_cuda` |
| 3 | `sageattention/triton/quant_per_thread.py` | `per_thread_int8` |
| 4 | `sageattention/triton/quant_per_thread.py` | `per_channel_fp8` |
| 5 | `csrc/fused/fused.cu` | `QuantInt8Kernel` |
| 6 | `csrc/fused/fused.cu` | `TransposePadPermuteKernel` |
| 7 | `csrc/fused/fused.cu` | `MeanScaleKernel` |
| 8 | `sageattention/sm89_compile.py` | custom-op wrappers |
| 9 | `csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu` | CUDA launcher |
| 10 | `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh` | `qk_int_sv_f8_attn_kernel` |
| 11 | `csrc/qattn/attn_utils.cuh` | QK, softmax, and PV helpers |
| 12 | `csrc/mma.cuh` | inline PTX MMA wrappers |
| 13 | `csrc/qattn/pybind_sm89.cpp` | extension bindings |
| 14 | `setup.py` | architecture and source selection |

The project uses `sv` in low-level symbol names for softmax-value multiplication. It corresponds to `PV` in the public API.

## 4. High-level dispatch

Open `sageattention/core.py` and find:

```python
def sageattn(...)
```

It reads the CUDA compute capability and selects an implementation. For SM120 it selects:

```python
sageattn_qk_int8_pv_fp8_cuda(
    ...,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32+fp32",
)
```

Next, read:

```python
def sageattn_qk_int8_pv_fp8_cuda(...)
```

Divide this function into the following stages:

1. Validate CUDA device, dtype, layout, and quantization granularity.
2. Pad head dimension to 64 or 128.
3. Optionally subtract the sequence mean from K.
4. Quantize Q and K to INT8.
5. Transpose, pad, and quantize V to E4M3.
6. Select the CUDA kernel according to `pv_accum_dtype`.
7. Slice a padded output back to the original head dimension.
8. Correct and return LSE when requested.

### K smoothing

When `smooth_k=True`, the wrapper computes:

```text
K_mean = mean(K, sequence_dimension)
K_smooth = K - K_mean
```

The attention score becomes:

```text
Q K^T = Q K_smooth^T + Q K_mean^T
```

The first term is computed by the quantized kernel. The second term is relevant to LSE correction but cancels from softmax probabilities because it is constant across a row.

## 5. INT8 Q/K quantization

The Python entry is `per_thread_int8` in `sageattention/triton/quant_per_thread.py`.

For the fixed configuration:

```text
BLKQ  = 128
WARPQ = 32
BLKK  = 64
```

Q gets one scale for each 32-token row group inside a 128-token CTA block. K gets one scale for each 64-token block.

For NHD tensors:

```text
Q input:   [B, Nq, Hq, D]
K input:   [B, Nk, Hk, D]
Q INT8:    [B, Nq, Hq, D]
K INT8:    [B, Nk, Hk, D]
q_scale:   [B, Hq, ceil(Nq / 128) * 4]
k_scale:   [B, Hk, ceil(Nk / 64)]
```

The CUDA implementation starts at `QuantInt8Kernel` in `csrc/fused/fused.cu`.

Conceptually, each quantization group performs:

```text
absmax = max(abs(x))
scale = absmax / 127
x_int8 = round(x / scale)
x ≈ x_int8 × scale
```

While reading `QuantInt8Kernel`, identify:

- how threads load packed BF16/FP16 values;
- warp/block absmax reduction;
- scale storage;
- conversion and saturation to INT8;
- optional fused subtraction of K mean;
- NHD versus HND stride calculation.

Then read the launcher `quant_per_thread_int8_cuda` near the lower half of the same file.

## 6. FP8 V preprocessing

The Python entry is `per_channel_fp8` in `sageattention/triton/quant_per_thread.py`.

V is not consumed in its original attention layout. It is transposed so the PV MMA reads contiguous values along the sequence dimension.

For NHD:

```text
V input:  [B, Nk, Hk, D]
V kernel: [B, D, Hk, padded_Nk]
```

For HND:

```text
V input:  [B, Hk, Nk, D]
V kernel: [B, Hk, D, padded_Nk]
```

`padded_Nk` is rounded up to a multiple of 64.

Read these CUDA components in `csrc/fused/fused.cu`:

1. `TransposePadPermuteKernel`
2. `transpose_pad_permute_cuda`
3. `MeanScaleKernel`
4. `scale_fuse_quant_cuda`

The per-channel quantization convention is:

```text
scale[d] = absmax(V[:, d]) / scale_max
V_fp8[:, d] = E4M3(V[:, d] / scale[d])
V[:, d] ≈ V_fp8[:, d] × scale[d]
```

For normal FP32 PV accumulation, `scale_max` is 448, the maximum finite E4M3 magnitude. SageAttention2++ may use a smaller value for its FP16 short-term accumulator to control overflow.

The output scale shape is:

```text
v_scale: [B, Hk, D]
```

## 7. Python custom-op layer

Open `sageattention/sm89_compile.py`.

This file registers PyTorch custom operators and forwards calls to `sageattention._qattn_sm89`.

For the fixed SM120 configuration, trace:

```python
qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(...)
```

This layer provides:

- a stable Python callable;
- mutation declaration for the output tensor;
- fake/meta implementation for `torch.compile`;
- forwarding to the pybind extension.

It does not contain the attention algorithm.

## 8. CUDA launcher

Open:

`csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu`

The launcher is responsible for runtime validation and compile-time specialization.

Read it in this order:

1. CUDA and contiguous checks.
2. Q/K/V/output dtype checks.
3. NHD/HND shape and stride extraction.
4. GQA validation: `num_qo_heads % num_kv_heads == 0`.
5. Optional LSE allocation.
6. Head-dimension dispatch.
7. Causal-mask dispatch.
8. QK quantization-granularity dispatch.
9. Output dtype dispatch.
10. Kernel template construction.
11. Dynamic shared-memory size.
12. Grid and block dimensions.
13. Kernel launch arguments.

The important tiling constants are:

```cpp
constexpr int CTA_Q = 128;
constexpr int CTA_K = 64;
constexpr int WARP_Q = 32;
constexpr int WARP_K = 64;
```

The launch geometry is approximately:

```text
grid.x = ceil(qo_len / CTA_Q)
grid.y = num_qo_heads
grid.z = batch_size

warps per CTA = (CTA_Q / WARP_Q) × (CTA_K / WARP_K)
              = 4 × 1
              = 4
```

## 9. Main CUDA kernel

Open `qk_int_sv_f8_cuda_sm89.cuh` and find:

```cpp
__global__ void qk_int_sv_f8_attn_kernel(...)
```

### 9.1 Compile-time tile counts

For `head_dim=128`:

```text
num_warps_q       = CTA_Q / WARP_Q = 4
num_warps_k       = CTA_K / WARP_K = 1
num_tiles_q       = WARP_Q / 16    = 2
num_tiles_k       = WARP_K / 16    = 4
num_tiles_qk_inner= head_dim / 32  = 4
num_tiles_v       = head_dim / 16  = 8
```

Use these concrete values while reading array declarations and loops.

### 9.2 Persistent register state

The important fragments are:

```cpp
int32_t RS[num_tiles_q][num_tiles_k][8];
float RO[num_tiles_q][num_tiles_v][8];
float m[num_tiles_q][2];
float d[num_tiles_q][2];
```

They mean:

| Fragment | Purpose |
| --- | --- |
| `RS` | INT32 QK score fragment |
| `RO` | Long-term FP32 PV output accumulator |
| `m` | Online-softmax row maximum |
| `d` | Online-softmax row denominator |

In the SageAttention2++ path, helper functions also create short-lived FP16 PV accumulators before periodically adding them into `RO`.

### 9.3 Shared-memory regions

Find the declarations of:

```text
smem_Q
smem_K
smem_V
smem_O
```

Q and K use a permuted shared-memory layout suitable for `ldmatrix`. V uses a different orientation because it is the right operand of the PV MMA. The same shared-memory allocation may be reused for output during the epilogue.

Study `permuted_smem.cuh` only after understanding which logical matrix each region stores.

### 9.4 Global-to-shared movement

The kernel uses asynchronous copies and double-buffer-like scheduling:

```text
load K/V tile
commit async copy
wait for required copy group
synchronize CTA
compute current tile
start loading next tile
```

Important helpers in `attn_utils.cuh` include:

- `load_global_to_share`
- `load_fp8_V_global_to_share`

Also inspect `cp_async.cuh` for the PTX-level wrappers.

## 10. INT8 QK MMA

Find `compute_int_qk` in `attn_utils.cuh`.

The helper:

1. loads Q and K fragments with `ldmatrix`;
2. advances through the head dimension in chunks of 32;
3. initializes the accumulator on the first chunk;
4. updates it in place for later chunks.

The core instruction wrapper is:

```cpp
mma_sync_m16n16k32_row_col_s8s8s32(...)
```

It is defined in `csrc/mma.cuh` and uses inline PTX based on:

```text
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
```

Two `m16n8k32` instructions form one logical `m16n16k32` result.

After QK MMA:

```text
RS_int32 = Q_int8 × K_int8^T
RS_float = float(RS_int32)
dequant_scale = q_scale × k_scale
S = RS_float × dequant_scale × sm_scale
```

## 11. Masks and boundary handling

Before softmax, the kernel may apply:

- `apply_causal_mask`
- `apply_out_of_bound_mask`

The causal condition masks positions where:

```text
key_index > query_index
```

The boundary mask is always needed on the last K tile when `kv_len` is not divisible by `CTA_K`.

Masked scores are replaced by a large negative value so their exponentials become zero.

## 12. Online softmax

Find `update_mdo` in `attn_utils.cuh`.

For each score tile, the kernel maintains a running maximum and denominator:

```text
tile_max = row_max(S_tile)
m_new = max(m_old, tile_max)
alpha = exp2((m_old - m_new) × scale)
P_tile = exp2((S_tile - m_new) × scale)
d_new = alpha × d_old + row_sum(P_tile)
RO = alpha × RO
```

The implementation uses base-2 exponentials, so the wrapper multiplies the softmax scale by `log2(e)`.

The key idea is that the complete attention matrix is never materialized in global memory. Each K/V tile updates the softmax state and output accumulator immediately.

Also inspect:

- `accumulate_d`
- `accumulate_d_f8`
- row-max and row-sum helpers in `mma.cuh`

## 13. Probability conversion to FP8

After online-softmax updates, the probability fragment is converted to E4M3 by:

```cpp
RS_32_to_8(...)
```

in `attn_utils.cuh`.

This step:

1. receives FP32 exponentials;
2. applies the probability scaling convention;
3. converts values to E4M3;
4. packs several FP8 values into `uint32_t` registers;
5. produces the left operand for PV MMA.

The packed fragment is commonly named `RS_f8`.

The softmax denominator and output accumulator account for the scaling used during this conversion.

## 14. FP8 PV MMA

The standard FP32-accumulator helper is:

```cpp
compute_fp8_sv(...)
```

The underlying instruction wrapper is:

```cpp
mma_sync_m16n16k32_row_col_f8f8f32(...)
```

which uses E4M3 operands and FP32 accumulation.

SageAttention2++ uses:

```cpp
compute_fp8_sv_inst_buf_fp16_accu(...)
```

The two-level accumulation strategy is:

```text
E4M3 P × E4M3 V
        │
        └─> FP32 short-term instruction buffer
                  │ periodically converted/added
                  └─> FP32 long-term RO
```

This is the meaning of:

```text
pv_accum_dtype="fp32+fp32"
```

Compare it with:

| Public option | Helper strategy |
| --- | --- |
| `fp32` | direct FP32 accumulation |
| `fp32+fp32` | FP32 instruction buffer + FP32 long-term buffer |
| `fp32+fp32` | FP32 instruction buffer + FP32 long-term buffer |

## 15. Epilogue

After processing all K/V tiles, the kernel calls `normalize_d`:

```text
O = RO / d
```

The epilogue then:

1. applies per-channel `v_scale`;
2. optionally restores `v_mean`;
3. converts FP32 accumulator values to FP16/BF16;
4. stores output through shared memory;
5. optionally stores LSE.

LSE is based on the online-softmax state:

```text
LSE = m + log2(d)
```

The Python wrapper converts it from base 2 to natural logarithm and adds the K-smoothing correction when needed.

## 16. Bindings and build

The pybind registration is in:

`csrc/qattn/pybind_sm89.cpp`

The extension source list and architecture flags are in:

`setup.py`

On an SM120 machine, setup.py includes:

```text
-gencode arch=compute_120a,code=sm_120a
```

Therefore:

- the source organization and pipeline are SM89-style;
- NVCC emits SM120 machine code;
- the resulting extension runs natively on SM120;
- it is not yet a Blackwell-specific retuning of the kernel.

## 17. Suggested reading sessions

### First pass: control flow

Read only:

1. `sageattn`
2. `sageattn_qk_int8_pv_fp8_cuda`
3. the FP16 instruction-buffer launcher
4. the main loop of `qk_int_sv_f8_attn_kernel`

Goal: understand the sequence of QK, softmax, and PV stages.

### Second pass: quantization and layouts

Read:

1. `per_thread_int8`
2. `QuantInt8Kernel`
3. `per_channel_fp8`
4. V transpose and E4M3 quantization
5. scale-index calculations in the attention kernel

Goal: reconstruct every tensor shape and scale index.

### Third pass: Tensor Core fragments

Read:

1. `compute_int_qk`
2. INT8 MMA wrapper
3. `RS_32_to_8`
4. `compute_fp8_sv_inst_buf_fp16_accu`
5. FP8 MMA wrappers

Goal: understand fragment layouts and accumulator types.

### Fourth pass: numerical behavior

Read:

1. K smoothing
2. `update_mdo`
3. denominator accumulation
4. two-level PV accumulation
5. `normalize_d`
6. LSE correction

Goal: understand where approximation error enters and how stability is maintained.

## 18. Useful source-navigation commands

From the repository root:

```bash
rg -n "def sageattn_qk_int8_pv_fp8_cuda" sageattention/core.py
rg -n "def per_thread_int8|def per_channel_fp8" sageattention/quant.py
rg -n "QuantInt8Kernel|TransposePadPermuteKernel|MeanScaleKernel" csrc/fused/fused.cu
rg -n "qk_int_sv_f8_attn_kernel" csrc/qattn
rg -n "compute_int_qk|update_mdo|RS_32_to_8|compute_fp8_sv|normalize_d" csrc/qattn/attn_utils.cuh
rg -n "s8s8s32|f8f8f32" csrc/mma.cuh
```

To inspect generated PTX/SASS after building:

```bash
cuobjdump --dump-sass sageattention/_qattn_sm89*.so
```

Search the output for integer and FP8 MMA instructions. Keep in mind that exact SASS mnemonics depend on the target architecture and CUDA version.

## 19. Verification while reading

Use the existing comparison benchmark:

```bash
python bench/bench_qk_fp8_pv_fp8.py --seq-len 1024
```

For deeper debugging, add temporary checks one stage at a time:

1. compare INT8 dequantized Q/K with BF16 Q/K;
2. compare one QK tile with a PyTorch matrix multiplication;
3. compare online-softmax `m` and `d` with a reference;
4. compare FP8-dequantized V with BF16 V;
5. compare the final output and LSE.

Avoid adding device-side printing to the fully unrolled production kernel initially; it substantially changes register usage and timing.
