#!/usr/bin/env python3
"""Generate the eight-table SageAttention QKV benchmark report."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/sageattention_benchmark_results.json"
OUTPUT = ROOT / "docs/SageAttention_QKV_Benchmark报告.md"

METHODS = [
    ("bf16_sdpa", "BF16 SDPA benchmark"),
    ("int8_qk_fp8_pv", "int8_qk_fp8_pv"),
    ("fp8_qk_fp8_pv", "fp8_qk_fp8_pv"),
    ("fp8_qk_fp8_pv_smooth_q", "fp8_qk_fp8_pv + smooth_q"),
    ("fp8_qk_fp8_pv_smooth_v", "fp8_qk_fp8_pv + smooth_v"),
    ("fp8_qk_fp8_pv_smooth_qv", "fp8_qk_fp8_pv + smooth_qv"),
    ("sim_int8_qk_fp8_pv", "int8_qk_fp8_pv（仿真）"),
    ("sim_fp8_qk_fp8_pv", "fp8_qk_fp8_pv（仿真）"),
    ("sim_fp8_qk_fp8_pv_smooth_q", "fp8_qk_fp8_pv + smooth_q（仿真）"),
    ("sim_fp8_qk_fp8_pv_smooth_v", "fp8_qk_fp8_pv + smooth_v（仿真）"),
    ("sim_fp8_qk_fp8_pv_smooth_qv", "fp8_qk_fp8_pv + smooth_qv（仿真）"),
]

PAIRS = [
    ("sim_int8_qk_fp8_pv", "int8_qk_fp8_pv（仿真 vs 算子）"),
    ("sim_fp8_qk_fp8_pv", "fp8_qk_fp8_pv（仿真 vs 算子）"),
    ("sim_fp8_qk_fp8_pv_smooth_q", "fp8_qk_fp8_pv + smooth_q（仿真 vs 算子）"),
    ("sim_fp8_qk_fp8_pv_smooth_v", "fp8_qk_fp8_pv + smooth_v（仿真 vs 算子）"),
    ("sim_fp8_qk_fp8_pv_smooth_qv", "fp8_qk_fp8_pv + smooth_qv（仿真 vs 算子）"),
]


def f6(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    data = json.loads(INPUT.read_text())
    samples = data["samples"]
    env = data["environment"]
    sim = data["simulation"]
    cross = data["simulation_kernel_cross_validation"]

    lines = [
        "# SageAttention：MiniMax H3 QKV 仿真与 CUDA 算子交叉验证",
        "",
        "本文档使用同一批 MiniMax H3 真实 Q/K/V 激活，对 BF16 SDPA、SageAttention CUDA 算子和独立的纯 PyTorch 算法仿真进行数值与耗时测试。",
        "",
        "## 第一节：数据获取",
        "",
        "### 1.1 ComfyUI 工作流",
        "",
        "数据来自 ComfyUI 官方 `MiniMax H3: Text to Video` 模板调整后的 API 工作流。生成参数为 1344×768、243 帧、24 fps、seed=0；采样器为 `res_multistep`，scheduler 为 `simple`，共 20 步。工作流使用以下模型：",
        "",
        "- Diffusion model：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`",
        "- Text encoder：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`",
        "- Video VAE：`minimax_h3_video_vae_fp16.safetensors`",
        "- Audio VAE：`minimax_h3_audio_vae_fp32.safetensors`",
        "",
        "完整 prompt 保存在 `config/prompt.txt`，可执行 API 工作流保存在 `config/workflow_api.json`。",
        "",
        "### 1.2 捕获位置和触发方式",
        "",
        "QKV 捕获逻辑直接加入 `ComfyUI/comfy/ldm/minimax/model.py`，由环境变量 `COMFYUI_QKV_CAPTURE_CONFIG` 指向 `config/capture_config.json` 后启用。配置指定 step `[5, 19]` 和 transformer layer `[3, 25, 47]`，两者均采用从 0 开始的索引，因此一共得到 2×3=6 个样本。",
        "",
        "每次 MiniMax H3 diffusion forward 开始时，捕获逻辑根据 sigma 序列维护 step 计数；进入 transformer block 循环时记录当前 layer。只有 step 和 layer 同时命中配置才保存，正常推理路径与 attention 计算本身不被替换。",
        "",
        "### 1.3 保存的张量",
        "",
        "捕获点位于 Q/K RMSNorm 和 RoPE 之后、attention 调用之前；V 是 projection 输出。因此保存的数据正是 attention 算子的直接输入，而不是线性层之前的 hidden state。Q、K、V 原本是 packed projection 的 view，保存前执行 `detach().contiguous()` 建立独立存储，再移动到 CPU 并显式转换为 BF16，最后分别写为 `q.pt`、`k.pt`、`v.pt`。",
        "",
        f"6 个样本均为 layout `B,H,S,D`，shape `{samples[0]['shape']}`，保存 dtype 为 `torch.bfloat16`。step 5 的 sigma 为 {samples[0]['sigma']:.9f}，step 19 的 sigma 为 {samples[3]['sigma']:.9f}。每个目录还保存 `metadata.json`，记录 step、layer、sigma、shape、运行时 dtype 和捕获位置。",
        "",
        "## 第二节：FP8 算法仿真",
        "",
        "算法仿真实现在 `SageAttention_Sim/sageattention/h3_fp8_smoothq_sim`，全部由普通 PyTorch 运算组成，不调用本文测试的 SageAttention CUDA attention kernel。其目的不是获得性能，而是把量化和 online-softmax 数据流写成易检查、易修改的参考实现。",
        "",
        "仿真包含以下步骤：",
        "",
        "1. 始终执行 `smooth_k`：在 BF16 中计算并减去 K 的序列均值。该操作只给每个 softmax row 引入常数平移，不需要输出校正。",
        "2. Q/K 按 per-thread group 计算 FP32 scale。INT8 模式使用除以 127、round-half-away-from-zero 和饱和转换；FP8 模式使用除以 448 和 E4M3 饱和转换，再反量化到 FP32 进行参考矩阵乘。",
        "3. V 按 channel 计算 FP32 scale 并转换为 E4M3；`smooth_v` 时先减均值，并在最终输出中加回。仿真复现了 V 长度向 16 对齐后 padding 参与统计的细节。",
        f"4. K 维以 `cta_k={sim['cta_k']}` 分块，执行 FlashAttention 风格的 online softmax：维护 running max、分母和输出累加器；softmax probability 在每个 tile 内转换为 E4M3 后参与 PV。",
        "5. `smooth_q` 时对 Q 减均值，同时加入 `mean_q @ K'^T` 校正项。Q 均值平移不像 K 均值平移那样具有 softmax 不变性，因此该校正不可省略。",
        "6. PV 在 FP32 中累加，除以未量化 softmax probability 的 FP32 分母，乘回 V scale、加回可选 V mean，最终输出转换回 BF16。",
        "",
        f"`block_q={sim['block_q']}` 只用于限制内存峰值；query row 相互独立，因此它不改变仿真算法结果。",
        "",
        "## 第三节：FP8 算子实现",
        "",
        "CUDA 算子实现在独立的 `/workspace/SageAttention` feature branch 中。它与第二节的仿真不是由同一套执行代码切换出来的两种模式：仿真是纯 PyTorch 参考路径；算子由 Triton 量化 kernel、CUDA 预处理 kernel 和 CUDA attention kernel 组成。两套实现分别产生输出，再通过 BF16 基准及彼此之间的 relative L2 做交叉验证。",
        "",
        "`sageattn_qk_fp8_pv_fp8_cuda` 的执行过程如下：",
        "",
        "1. Python API 校验 HND/NHD layout、dtype、head dimension 和配置，并计算可选 `smooth_k`。",
        "2. 与 INT8 CUDA 路径一致，`smooth_k` 以及可选 `smooth_q` 的 mean/subtraction 先在输入 dtype（本次为 BF16）中完成；Triton 随后加载已经中心化的 Q/K，提升到 FP32 计算 scale，再转换为 E4M3。分组同样与 INT8 per-thread 一致：Q 在每 32 行内分成 8 个跳步组，K 在每 64 行内分成 4 个跳步组；scale 使用 `/448 + 1e-7`。`smooth_q` correction 使用 BF16 Q mean 和每个 K row 对应的 per-thread scale。",
        "3. CUDA `per_channel_fp8` 将 V 转置、padding 并按 channel 转换为 E4M3，同时产生 FP32 scale 和可选 V mean。",
        "4. CUDA attention kernel 使用 E4M3 QK MMA 和 FP32 score accumulator，逐 `CTA_K=64` tile 更新 online softmax；probability 转换为 E4M3 后执行 E4M3 PV MMA，并以 FP32 累加输出。",
        "5. 根据开关分派到基础、fuse_q_mean、fuse_v_mean 或同时融合两者的四个入口，最后写出 BF16/FP16 output。",
        "",
        "INT8 对照路径使用 `sageattn_qk_int8_pv_fp8_cuda`：Q/K 为 INT8，QK 使用整数点积；P/V 仍为 E4M3。本次配置为 `qk_quant_gran=per_thread`、`pv_accum_dtype=fp32`，因此 PV 使用 FP32 accumulator。FP8-only 路径同样使用 per-thread Q/K scale 和 FP32 PV accumulator。",
        "",
        "这种独立实现方式是有意的：如果仿真和算子共享量化结果或核心计算代码，二者可能同时继承同一个错误而仍然得到一致结果；独立实现再比较输出，更适合发现分组、scale、平滑校正、padding、累加精度和舍入顺序方面的问题。",
        "",
        "## 第四节：测试方法",
        "",
        f"- GPU：{env['gpu']}，compute capability {'.'.join(map(str, env['compute_capability']))}",
        f"- PyTorch：{env['torch']}；PyTorch CUDA：{env['torch_cuda']}；Python：{env['python']}",
        f"- SageAttention：`{env['sageattention_branch']}` @ `{env['sageattention_commit']}`",
        "- 输入：6 个完整长度 BF16 Q/K/V 样本；HND layout；非因果 attention；没有进行 token 截断或抽样",
        "- BF16 reference：`torch.nn.functional.scaled_dot_product_attention`",
        f"- 算子计时：CUDA Events；预热 {data['warmup']} 次，随后正式运行 {data['repeats']} 次。表中算子“耗时”是 5 次正式运行的算术平均值",
        f"- 仿真计时：CUDA Events；不预热，每个样本的每种配置完整运行 {sim['repeats']} 次。因此仿真“耗时”是单次值，不是多次平均",
        f"- 计时范围：{data['benchmark_scope']}。包含量化、平滑、校正和 attention API 内的 GPU 工作，不包含磁盘读取及 CPU→GPU 加载",
        f"- 相对 BF16 的误差：`{data['relative_l2']}`",
        f"- 仿真与算子交叉误差：`{cross['metric']}`",
        "- 所有 L2 统计均将输出分块转为 FP32，平方和使用 FP64 累计，以控制临时显存和规约误差",
        "",
        "## 第五节：测试结果",
        "",
        "表 1 汇总 6 个样本：耗时取各样本耗时的算术平均；relative L2 的平均、最小和最大值在 6 个样本间统计。BF16 是误差参考，所以其 relative L2 定义为 0。",
        "",
        "### 表 1：六个样本汇总",
        "",
        "| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |",
        "|---|---:|---:|---:|---:|",
    ]

    for key, label in METHODS:
        elapsed = [s["methods"][key]["mean_ms"] for s in samples]
        errors = [s["methods"][key]["relative_l2"] for s in samples]
        lines.append(
            f"| {label} | {mean(elapsed):.3f} | {f6(mean(errors))} | "
            f"{f6(min(errors))} | {f6(max(errors))} |"
        )

    lines += [
        "",
        "表 2 直接比较每个仿真输出和与其配置对应的 CUDA 算子输出。分母是 CUDA 算子输出的 L2 norm；该表不使用 BF16 作为中间参照。",
        "",
        "### 表 2：仿真输出与 CUDA 算子输出的直接交叉验证",
        "",
        "| 配置 | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |",
        "|---|---:|---:|---:|",
    ]
    sample_names = [s["sample"] for s in samples]
    for key, label in PAIRS:
        errors = [cross["samples"][name][key]["relative_l2"] for name in sample_names]
        lines.append(
            f"| {label} | {f6(mean(errors))} | "
            f"{f6(min(errors))} | {f6(max(errors))} |"
        )

    lines += [
        "",
        "仿真慢不代表 FP8 算法本身慢。纯 PyTorch 仿真需要通过 Python 循环逐块执行 matmul、exp2、规约、类型转换和中间 tensor 创建，算子之间还会产生额外的调度与显存读写开销。真实 CUDA kernel 则将量化、MMA、online softmax、概率 FP8 转换和 PV 累加融合在 GPU 内部，利用 shared memory、寄存器和异步流水完成，因此仿真速度不能用于评价最终 CUDA 算子性能。",
        "",
        "当前的微小差异来自仿真与 CUDA 算子两条独立执行路径中不同的累加与舍入顺序。CUDA 算子按照固定的 MMA fragment 和 warp 规约顺序执行 E4M3 QK、online softmax 与 E4M3 PV；仿真则将数据反量化后交给 PyTorch matmul 和张量规约。虽然两者使用相同的量化数据和 FP32 accumulator，但浮点加法不满足结合律，不同的运算分块和累加顺序会产生微小的舍入差异，并可能在 probability 转换为 E4M3 时被进一步传播。",
        "",
    ]

    for table_number, sample in enumerate(samples, 3):
        shape = "×".join(map(str, sample["shape"]))
        lines += [
            "",
            f"### 表 {table_number}：{sample['sample']}（step={sample['step']}，layer={sample['layer']}）",
            "",
            f"输入：`{sample['dtype']}`，shape `{shape}`，sigma={sample['sigma']:.9f}。",
            "",
            "| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, label in METHODS:
            result = sample["methods"][key]
            error = result["relative_l2"]
            lines.append(
                f"| {label} | {result['mean_ms']:.3f} | {f6(error)} | "
                f"{f6(error)} | {f6(error)} |"
            )

    lines += [
        "",
        "## 可复现文件",
        "",
        "- 捕获配置：`config/capture_config.json`",
        "- ComfyUI API 工作流：`config/workflow_api.json`",
        "- Prompt：`config/prompt.txt`",
        "- 原始结果：`results/sageattention_benchmark_results.json`",
        "- CUDA 算子 Benchmark：`scripts/benchmark_sageattention.py`",
        "- 仿真 Benchmark：`scripts/benchmark_sageattention_sim.py`",
        "- 仿真/算子交叉验证：`scripts/cross_validate_sim_kernel.py`",
        "- 报告生成器：`scripts/generate_benchmark_report.py`",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
