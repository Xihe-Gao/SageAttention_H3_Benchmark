# FP8 QK + FP8 PV CUDA Kernel Walkthrough（参考扩展）

本文参考 `Official_INT8_FP8_CUDA_Kernel_Walkthrough.md` 的源码阅读结构，说明本项目加入的 FP8 QK + FP8 PV CUDA kernel。**该路径不是 SageAttention 官方独立实现**，而是在官方 INT8 QK + FP8 PV kernel 的 launcher、template 和 attention 流程上扩展出的 FP8 QK 参考实现。

源码来自本仓库的 `third_party/SageAttention` submodule；相对路径均相对于该目录。

## 1. 与官方 INT8+FP8 路径的差异

| 部分 | 官方 INT8+FP8 | 本参考 FP8+FP8 |
|---|---|---|
| Python Q/K 量化 | `per_warp_int8` | `per_thread_fp8`，输出 E4M3 |
| Q/K dtype | INT8 | FP8 E4M3 |
| QK MMA | INT8×INT8→INT32 (`IMMA`) | FP8×FP8→FP32 (`f8f8f32` MMA) |
| QK scale | `/127`，INT8 scale | `/448 + 1e-7`，FP8 scale |
| QK accumulator | INT32 score fragment | FP32 score fragment |
| V 量化 | per-channel E4M3 | 相同 |
| PV MMA | E4M3×E4M3 | 相同 |
| PV accumulator | FP32 配置（含可选两级路径） | 当前 wrapper 直接 FP32 累加，不启用 short-term buffer |
| CUDA launcher | `DataType::kInt8` | `DataType::kE4M3` |

除 Q/K 的量化和 QK MMA 外，两条路径共享 online softmax、probability FP8 转换、PV 计算和输出归一化逻辑。

## 2. Python API 调用链

入口位于 `sageattention/core.py`：

```python
sageattn_qk_fp8_pv_fp8_cuda(
    q, k, v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32",
    smooth_k=True,
)
```

与官方函数 `sageattn_qk_int8_pv_fp8_cuda` 的主要差异：

1. Q/K 调用 `triton.quant_per_warp_fp8.per_thread_fp8`，而不是 `quant.per_warp_int8`。
2. 使用 `qk_quant_gran="per_thread"` 和当前源码支持的 `pv_accum_dtype="fp32"`。
3. Q、K 输出为 `torch.float8_e4m3fn`，scale 为 FP32。
4. 根据 `smooth_q`、`smooth_v` 选择基础或 mean-fused custom op。

V 仍调用官方的 `per_channel_fp8`，因此 V 路径与 INT8+FP8 保持一致。

## 3. FP8 Q/K 量化

实现位于 `sageattention/triton/quant_per_warp_fp8.py` 的 `per_thread_fp8`。

Q 使用与官方 INT8 per-thread 路径相同的跳步分组：每个 32 行 warp block 分成 8 个线程组；K 每个 64 行 block 分成 4 个线程组。不同之处是量化格式：

```text
amax  = max(abs(x_group))
scale = amax / 448 + 1e-7
q_fp8 = saturate_to_e4m3(x_group / scale)
```

`smooth_k` 和 `smooth_q` 的减均值在输入 dtype（本次为 BF16）中执行，随后 Triton 再将中心化结果提升到 FP32 计算 scale。这一点用于与官方 INT8 路径及仿真保持一致。

输出布局为：

```text
Q: torch.float8_e4m3fn + q_scale(float32)
K: torch.float8_e4m3fn + k_scale(float32)
```

## 4. Python custom-op wrapper

` sageattention/sm89_compile.py` 将 C++ 扩展注册为 PyTorch custom op。FP8 路径对应的入口包括：

```text
qk_fp8_sv_f8_accum_f32_fuse_v_scale_attn
qk_fp8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn
```

与官方 INT8 路径相比，wrapper 接收的 Q/K tensor dtype 不同，但 V、scale、layout、causal 和 LSE 参数组织方式基本相同。

## 5. C++ binding 与 CUDA launcher

PyTorch 扩展绑定位于：

```text
csrc/qattn/pybind_sm89.cpp
```

它将 Python custom-op 名称映射到 `.cu` 导出的函数。launcher 位于：

```text
csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu
csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu
```

官方路径实例化：

```cpp
qk_int_sv_f8_attn_kernel<..., DataType::kInt8, ...>
```

FP8 参考路径实例化：

```cpp
qk_int_sv_f8_attn_kernel<..., DataType::kE4M3, ...>
```

`DataType::kE4M3` 会让 kernel 使用 FP8 QK 的代码分支；PV 仍然走同一个 FP8 V 计算流程。

## 6. CUDA kernel 主流程

核心模板位于：

```text
csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh
```

对每个 Q block 和 K tile，流程是：

```text
加载 FP8 Q/K tile
    ↓
compute_int_qk(..., DTypeQK = kE4M3)
    ↓
FP8 Q × FP8 K → FP32 score fragment
    ↓
乘以 Q/K scale 和 sm_scale
    ↓
online softmax：更新 running max、denominator、output state
    ↓
将 probability 转换为 FP8
    ↓
FP8 probability × FP8 V → FP32 PV accumulator
    ↓
按 FP32 denominator 归一化并写回 BF16/FP16
```

这里函数名仍叫 `compute_int_qk`，是历史命名；当 `DTypeQK == kE4M3` 时实际执行的是 FP8 分支，而不是整数点积。

## 7. QK MMA 与 PV MMA

`csrc/qattn/attn_utils.cuh` 中的 FP8 分支调用：

```cpp
mma::mma_sync_m16n16k32_row_col_f8f8f32(...)
```

底层 wrapper 位于 `csrc/mma.cuh`，使用 FP8 输入、FP32 输出/累加的 MMA 形式。与官方 INT8+FP8 路径相比，差异只发生在 QK：

```text
官方：INT8 Q × INT8 K → INT32
参考：FP8  Q × FP8  K → FP32
```

两条路径的 PV 都是：

```text
FP8 probability × FP8 V → FP32 accumulator
```

源码中的 `RO` 为 `DTypeSVAccum=float`。需要特别注意：FP8 wrapper 虽然复用了官方 INT8+FP8 的通用 `qk_int_sv_f8_attn_kernel` template，但在 launcher 中传入 `use_inst_buffer=false`，因此实际调用的是 `compute_fp8_sv()`，直接将 FP8 PV MMA 结果累加到 `RO`；它不会调用 `compute_fp8_sv_inst_buf()`，也没有 FP32 short-term instruction buffer。`fp32+fp32` 是官方 INT8+FP8 的两级累加配置名，不能据此推断当前 FP8+FP8 wrapper 自动启用了同样的临时缓冲路径。

## 8. Smooth 选项

- `smooth_k`：Q/K 量化前减去 K 的序列均值；对 softmax row 只产生常数平移。
- `smooth_q`：Q 量化前减去 Q 均值，并通过 `q_mean_k_bias` 加回校正项。
- `smooth_v`：V 量化前减去 V 均值，最终输出加回 V mean。
- `smooth_qv`：同时启用 Q 和 V 校正。

这些选项只改变预处理、校正和融合入口，不改变 FP8 QK/PV 的基本 MMA 数据类型。

## 9. 如何与官方路径对照阅读

建议先阅读 `Official_INT8_FP8_CUDA_Kernel_Walkthrough.md`，再按以下差异定位源码：

1. `core.py`：将 `per_warp_int8` 替换为 `per_thread_fp8`。
2. `quant_per_warp_fp8.py`：检查 E4M3 scale 和跳步分组。
3. `.cu` launcher：比较 `DataType::kInt8` 与 `DataType::kE4M3` 实例化。
4. `attn_utils.cuh`：比较 INT8 `IMMA` 分支和 FP8 `f8f8f32` 分支。
5. `mma.cuh`：确认 QK/PV 的 FP8 MMA wrapper 与累加类型。

该 FP8+FP8 路径是用于 SageAttention H3 的实验性参考实现；性能和数值结果应通过独立的 PyTorch simulation 及 BF16 reference 进行验证。
