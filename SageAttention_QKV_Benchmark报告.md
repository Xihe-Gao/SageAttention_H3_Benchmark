# SageAttention：MiniMax H3 QKV 仿真与 CUDA 算子交叉验证

本文档使用同一批 MiniMax H3 真实 Q/K/V 激活，对 BF16 SDPA、SageAttention CUDA 算子和独立的纯 PyTorch 算法仿真进行数值与耗时测试。

## 第一节：数据获取

### 1.1 ComfyUI 工作流

数据来自 ComfyUI 官方 `MiniMax H3: Text to Video` 模板调整后的 API 工作流。生成参数为 1344×768、243 帧、24 fps、seed=0；采样器为 `res_multistep`，scheduler 为 `simple`，共 20 步。工作流使用以下模型：

- Diffusion model：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Text encoder：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video VAE：`minimax_h3_video_vae_fp16.safetensors`
- Audio VAE：`minimax_h3_audio_vae_fp32.safetensors`

完整 prompt 保存在 `prompt.txt`，可执行 API 工作流保存在 `workflow_api.json`。

### 1.2 捕获位置和触发方式

QKV 捕获逻辑直接加入 `ComfyUI/comfy/ldm/minimax/model.py`，由环境变量 `COMFYUI_QKV_CAPTURE_CONFIG` 指向 `capture_config.json` 后启用。配置指定 step `[5, 19]` 和 transformer layer `[3, 25, 47]`，两者均采用从 0 开始的索引，因此一共得到 2×3=6 个样本。

每次 MiniMax H3 diffusion forward 开始时，捕获逻辑根据 sigma 序列维护 step 计数；进入 transformer block 循环时记录当前 layer。只有 step 和 layer 同时命中配置才保存，正常推理路径与 attention 计算本身不被替换。

### 1.3 保存的张量

捕获点位于 Q/K RMSNorm 和 RoPE 之后、attention 调用之前；V 是 projection 输出。因此保存的数据正是 attention 算子的直接输入，而不是线性层之前的 hidden state。Q、K、V 原本是 packed projection 的 view，保存前执行 `detach().contiguous()` 建立独立存储，再移动到 CPU 并显式转换为 BF16，最后分别写为 `q.pt`、`k.pt`、`v.pt`。

6 个样本均为 layout `B,H,S,D`，shape `[1, 56, 73923, 128]`，保存 dtype 为 `torch.bfloat16`。step 5 的 sigma 为 0.972972989，step 19 的 sigma 为 0.387096822。每个目录还保存 `metadata.json`，记录 step、layer、sigma、shape、运行时 dtype 和捕获位置。

## 第二节：FP8 算法仿真

算法仿真实现在 `SageAttention_Sim/sageattention/h3_fp8_smoothq_sim`，全部由普通 PyTorch 运算组成，不调用本文测试的 SageAttention CUDA attention kernel。其目的不是获得性能，而是把量化和 online-softmax 数据流写成易检查、易修改的参考实现。

仿真包含以下步骤：

1. 始终执行 `smooth_k`：在 BF16 中计算并减去 K 的序列均值。该操作只给每个 softmax row 引入常数平移，不需要输出校正。
2. Q/K 按 per-thread group 计算 FP32 scale。INT8 模式使用除以 127、round-half-away-from-zero 和饱和转换；FP8 模式使用除以 448 和 E4M3 饱和转换，再反量化到 FP32 进行参考矩阵乘。
3. V 按 channel 计算 FP32 scale 并转换为 E4M3；`smooth_v` 时先减均值，并在最终输出中加回。仿真复现了 V 长度向 16 对齐后 padding 参与统计的细节。
4. K 维以 `cta_k=64` 分块，执行 FlashAttention 风格的 online softmax：维护 running max、分母和输出累加器；softmax probability 在每个 tile 内转换为 E4M3 后参与 PV。
5. `smooth_q` 时对 Q 减均值，同时加入 `mean_q @ K'^T` 校正项。Q 均值平移不像 K 均值平移那样具有 softmax 不变性，因此该校正不可省略。
6. PV 在 FP32 中累加，除以未量化 softmax probability 的 FP32 分母，乘回 V scale、加回可选 V mean，最终输出转换回 BF16。

`block_q=2048` 只用于限制内存峰值；query row 相互独立，因此它不改变仿真算法结果。

## 第三节：FP8 算子实现

CUDA 算子实现在独立的 `/workspace/SageAttention` feature branch 中。它与第二节的仿真不是由同一套执行代码切换出来的两种模式：仿真是纯 PyTorch 参考路径；算子由 Triton 量化 kernel、CUDA 预处理 kernel 和 CUDA attention kernel 组成。两套实现分别产生输出，再通过 BF16 基准及彼此之间的 relative L2 做交叉验证。

`sageattn_qk_fp8_pv_fp8_cuda` 的执行过程如下：

1. Python API 校验 HND/NHD layout、dtype、head dimension 和配置，并计算可选 `smooth_k`。
2. 与 INT8 CUDA 路径一致，`smooth_k` 以及可选 `smooth_q` 的 mean/subtraction 先在输入 dtype（本次为 BF16）中完成；Triton 随后加载已经中心化的 Q/K，提升到 FP32 计算 scale，再转换为 E4M3。分组同样与 INT8 per-thread 一致：Q 在每 32 行内分成 8 个跳步组，K 在每 64 行内分成 4 个跳步组；scale 使用 `/448 + 1e-7`。`smooth_q` correction 使用 BF16 Q mean 和每个 K row 对应的 per-thread scale。
3. CUDA `per_channel_fp8` 将 V 转置、padding 并按 channel 转换为 E4M3，同时产生 FP32 scale 和可选 V mean。
4. CUDA attention kernel 使用 E4M3 QK MMA 和 FP32 score accumulator，逐 `CTA_K=64` tile 更新 online softmax；probability 转换为 E4M3 后执行 E4M3 PV MMA，并以 FP32 累加输出。
5. 根据开关分派到基础、fuse_q_mean、fuse_v_mean 或同时融合两者的四个入口，最后写出 BF16/FP16 output。

INT8 对照路径使用 `sageattn_qk_int8_pv_fp8_cuda`：Q/K 为 INT8，QK 使用整数点积；P/V 仍为 E4M3。本次配置为 `qk_quant_gran=per_thread`、`pv_accum_dtype=fp32`，因此 PV 使用 FP32 accumulator。FP8-only 路径同样使用 per-thread Q/K scale 和 FP32 PV accumulator。

这种独立实现方式是有意的：如果仿真和算子共享量化结果或核心计算代码，二者可能同时继承同一个错误而仍然得到一致结果；独立实现再比较输出，更适合发现分组、scale、平滑校正、padding、累加精度和舍入顺序方面的问题。

## 第四节：测试方法

- GPU：NVIDIA GeForce RTX 5090，compute capability 12.0
- PyTorch：2.10.0+cu130；PyTorch CUDA：13.0；Python：3.12.13
- SageAttention：`feature/qk-fp8-pv-fp8` @ `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`
- 输入：6 个完整长度 BF16 Q/K/V 样本；HND layout；非因果 attention；没有进行 token 截断或抽样
- BF16 reference：`torch.nn.functional.scaled_dot_product_attention`
- 算子计时：CUDA Events；预热 2 次，随后正式运行 5 次。表中算子“耗时”是 5 次正式运行的算术平均值
- 仿真计时：CUDA Events；不预热，每个样本的每种配置完整运行 1 次。因此仿真“耗时”是单次值，不是多次平均
- 计时范围：end-to-end GPU kernels; excludes disk I/O and CPU-to-GPU load。包含量化、平滑、校正和 attention API 内的 GPU 工作，不包含磁盘读取及 CPU→GPU 加载
- 相对 BF16 的误差：`||candidate.float-reference.float||_2 / ||reference.float||_2`
- 仿真与算子交叉误差：`||simulation.float-kernel.float||_2 / ||kernel.float||_2`
- 所有 L2 统计均将输出分块转为 FP32，平方和使用 FP64 累计，以控制临时显存和规约误差

## 第五节：测试结果

表 1 汇总 6 个样本：耗时取各样本耗时的算术平均；relative L2 的平均、最小和最大值在 6 个样本间统计。BF16 是误差参考，所以其 relative L2 定义为 0。

### 表 1：六个样本汇总

| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|---:|
| BF16 SDPA benchmark | 714.887 | 0.000000 | 0.000000 | 0.000000 |
| int8_qk_fp8_pv | 291.239 | 0.017821 | 0.012721 | 0.023637 |
| fp8_qk_fp8_pv | 358.578 | 0.039084 | 0.029672 | 0.058567 |
| fp8_qk_fp8_pv + smooth_q | 401.235 | 0.038537 | 0.029095 | 0.058537 |
| fp8_qk_fp8_pv + smooth_v | 359.042 | 0.038700 | 0.029545 | 0.058089 |
| fp8_qk_fp8_pv + smooth_qv | 404.814 | 0.038144 | 0.028960 | 0.058058 |
| int8_qk_fp8_pv（仿真） | 21825.628 | 0.017825 | 0.012725 | 0.023639 |
| fp8_qk_fp8_pv（仿真） | 21705.220 | 0.039087 | 0.029673 | 0.058571 |
| fp8_qk_fp8_pv + smooth_q（仿真） | 25153.182 | 0.038540 | 0.029097 | 0.058542 |
| fp8_qk_fp8_pv + smooth_v（仿真） | 21709.569 | 0.038699 | 0.029544 | 0.058089 |
| fp8_qk_fp8_pv + smooth_qv（仿真） | 25208.749 | 0.038144 | 0.028959 | 0.058058 |

表 1 中 FP8-only CUDA kernel 的耗时高于 INT8 QK + FP8 PV，主要原因是两者的 QK 硬件路径不同：INT8 路径使用 `IMMA` 执行 INT8×INT8→INT32，整数 Tensor Core 吞吐较高，scale 可在后续阶段处理；FP8-only 路径使用 `QMMA ... F32.E4M3.E4M3` 执行 FP8×FP8→FP32，需要承担 FP32 score 累加、scale 处理以及额外的寄存器和数据搬运开销。两种路径的 PV 都是 FP8 输入、FP32 累加，因此主要性能差异来自 QK；`smooth_q` 和 `smooth_qv` 还增加了均值校正计算，所以耗时进一步上升。

仿真慢不代表 FP8 算法本身慢。纯 PyTorch 仿真需要通过 Python 循环逐块执行 matmul、exp2、规约、类型转换和中间 tensor 创建，算子之间还会产生额外的调度与显存读写开销。真实 CUDA kernel 则将量化、MMA、online softmax、概率 FP8 转换和 PV 累加融合在 GPU 内部，利用 shared memory、寄存器和异步流水完成，因此仿真速度不能用于评价最终 CUDA 算子性能。

表 2 直接比较每个仿真输出和与其配置对应的 CUDA 算子输出。分母是 CUDA 算子输出的 L2 norm；该表不使用 BF16 作为中间参照。

### 表 2：仿真输出与 CUDA 算子输出的直接交叉验证

| 配置 | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|
| int8_qk_fp8_pv（仿真 vs 算子） | 0.001819 | 0.001287 | 0.002671 |
| fp8_qk_fp8_pv（仿真 vs 算子） | 0.001816 | 0.001290 | 0.002661 |
| fp8_qk_fp8_pv + smooth_q（仿真 vs 算子） | 0.001816 | 0.001290 | 0.002663 |
| fp8_qk_fp8_pv + smooth_v（仿真 vs 算子） | 0.001204 | 0.000829 | 0.001652 |
| fp8_qk_fp8_pv + smooth_qv（仿真 vs 算子） | 0.001204 | 0.000829 | 0.001653 |

当前的微小差异来自仿真与 CUDA 算子两条独立执行路径中不同的累加与舍入顺序。CUDA 算子按照固定的 MMA fragment 和 warp 规约顺序执行 E4M3 QK、online softmax 与 E4M3 PV；仿真则将数据反量化后交给 PyTorch matmul 和张量规约。虽然两者使用相同的量化数据和 FP32 accumulator，但浮点加法不满足结合律，不同的运算分块和累加顺序会产生微小的舍入差异，并可能在 probability 转换为 E4M3 时被进一步传播。


### 表 3：step_05_layer_03（step=5，layer=3）

输入：`torch.bfloat16`，shape `1×56×73923×128`，sigma=0.972972989。

| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|---:|
| BF16 SDPA benchmark | 710.675 | 0.000000 | 0.000000 | 0.000000 |
| int8_qk_fp8_pv | 288.598 | 0.023637 | 0.023637 | 0.023637 |
| fp8_qk_fp8_pv | 357.589 | 0.058567 | 0.058567 | 0.058567 |
| fp8_qk_fp8_pv + smooth_q | 399.959 | 0.058537 | 0.058537 | 0.058537 |
| fp8_qk_fp8_pv + smooth_v | 358.431 | 0.058089 | 0.058089 | 0.058089 |
| fp8_qk_fp8_pv + smooth_qv | 404.070 | 0.058058 | 0.058058 | 0.058058 |
| int8_qk_fp8_pv（仿真） | 22324.836 | 0.023639 | 0.023639 | 0.023639 |
| fp8_qk_fp8_pv（仿真） | 21972.248 | 0.058571 | 0.058571 | 0.058571 |
| fp8_qk_fp8_pv + smooth_q（仿真） | 25374.859 | 0.058542 | 0.058542 | 0.058542 |
| fp8_qk_fp8_pv + smooth_v（仿真） | 21927.223 | 0.058089 | 0.058089 | 0.058089 |
| fp8_qk_fp8_pv + smooth_qv（仿真） | 25420.338 | 0.058058 | 0.058058 | 0.058058 |

### 表 4：step_05_layer_25（step=5，layer=25）

输入：`torch.bfloat16`，shape `1×56×73923×128`，sigma=0.972972989。

| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|---:|
| BF16 SDPA benchmark | 715.965 | 0.000000 | 0.000000 | 0.000000 |
| int8_qk_fp8_pv | 292.091 | 0.012721 | 0.012721 | 0.012721 |
| fp8_qk_fp8_pv | 358.836 | 0.029672 | 0.029672 | 0.029672 |
| fp8_qk_fp8_pv + smooth_q | 401.562 | 0.029095 | 0.029095 | 0.029095 |
| fp8_qk_fp8_pv + smooth_v | 359.181 | 0.029545 | 0.029545 | 0.029545 |
| fp8_qk_fp8_pv + smooth_qv | 405.073 | 0.028960 | 0.028960 | 0.028960 |
| int8_qk_fp8_pv（仿真） | 21957.811 | 0.012725 | 0.012725 | 0.012725 |
| fp8_qk_fp8_pv（仿真） | 21844.342 | 0.029673 | 0.029673 | 0.029673 |
| fp8_qk_fp8_pv + smooth_q（仿真） | 25406.900 | 0.029097 | 0.029097 | 0.029097 |
| fp8_qk_fp8_pv + smooth_v（仿真） | 21996.330 | 0.029544 | 0.029544 | 0.029544 |
| fp8_qk_fp8_pv + smooth_qv（仿真） | 25834.512 | 0.028959 | 0.028959 | 0.028959 |

### 表 5：step_05_layer_47（step=5，layer=47）

输入：`torch.bfloat16`，shape `1×56×73923×128`，sigma=0.972972989。

| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|---:|
| BF16 SDPA benchmark | 714.399 | 0.000000 | 0.000000 | 0.000000 |
| int8_qk_fp8_pv | 291.642 | 0.015967 | 0.015967 | 0.015967 |
| fp8_qk_fp8_pv | 358.624 | 0.037717 | 0.037717 | 0.037717 |
| fp8_qk_fp8_pv + smooth_q | 401.426 | 0.036588 | 0.036588 | 0.036588 |
| fp8_qk_fp8_pv + smooth_v | 359.073 | 0.037573 | 0.037573 | 0.037573 |
| fp8_qk_fp8_pv + smooth_qv | 404.821 | 0.036427 | 0.036427 | 0.036427 |
| int8_qk_fp8_pv（仿真） | 21674.793 | 0.015972 | 0.015972 | 0.015972 |
| fp8_qk_fp8_pv（仿真） | 21621.641 | 0.037719 | 0.037719 | 0.037719 |
| fp8_qk_fp8_pv + smooth_q（仿真） | 25108.342 | 0.036591 | 0.036591 | 0.036591 |
| fp8_qk_fp8_pv + smooth_v（仿真） | 21556.275 | 0.037573 | 0.037573 | 0.037573 |
| fp8_qk_fp8_pv + smooth_qv（仿真） | 25000.881 | 0.036427 | 0.036427 | 0.036427 |

### 表 6：step_19_layer_03（step=19，layer=3）

输入：`torch.bfloat16`，shape `1×56×73923×128`，sigma=0.387096822。

| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|---:|
| BF16 SDPA benchmark | 715.604 | 0.000000 | 0.000000 | 0.000000 |
| int8_qk_fp8_pv | 290.641 | 0.021527 | 0.021527 | 0.021527 |
| fp8_qk_fp8_pv | 358.477 | 0.038896 | 0.038896 | 0.038896 |
| fp8_qk_fp8_pv + smooth_q | 401.002 | 0.038869 | 0.038869 | 0.038869 |
| fp8_qk_fp8_pv + smooth_v | 358.843 | 0.037847 | 0.037847 | 0.037847 |
| fp8_qk_fp8_pv + smooth_qv | 404.637 | 0.037818 | 0.037818 | 0.037818 |
| int8_qk_fp8_pv（仿真） | 21606.660 | 0.021530 | 0.021530 | 0.021530 |
| fp8_qk_fp8_pv（仿真） | 21584.615 | 0.038900 | 0.038900 | 0.038900 |
| fp8_qk_fp8_pv + smooth_q（仿真） | 25068.676 | 0.038872 | 0.038872 | 0.038872 |
| fp8_qk_fp8_pv + smooth_v（仿真） | 21640.936 | 0.037846 | 0.037846 | 0.037846 |
| fp8_qk_fp8_pv + smooth_qv（仿真） | 24994.779 | 0.037817 | 0.037817 | 0.037817 |

### 表 7：step_19_layer_25（step=19，layer=25）

输入：`torch.bfloat16`，shape `1×56×73923×128`，sigma=0.387096822。

| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|---:|
| BF16 SDPA benchmark | 717.822 | 0.000000 | 0.000000 | 0.000000 |
| int8_qk_fp8_pv | 292.501 | 0.013763 | 0.013763 | 0.013763 |
| fp8_qk_fp8_pv | 358.961 | 0.030420 | 0.030420 | 0.030420 |
| fp8_qk_fp8_pv + smooth_q | 401.842 | 0.030022 | 0.030022 | 0.030022 |
| fp8_qk_fp8_pv + smooth_v | 359.473 | 0.030145 | 0.030145 | 0.030145 |
| fp8_qk_fp8_pv + smooth_qv | 405.257 | 0.029743 | 0.029743 | 0.029743 |
| int8_qk_fp8_pv（仿真） | 21632.811 | 0.013770 | 0.013770 | 0.013770 |
| fp8_qk_fp8_pv（仿真） | 21629.195 | 0.030424 | 0.030424 | 0.030424 |
| fp8_qk_fp8_pv + smooth_q（仿真） | 24960.023 | 0.030026 | 0.030026 | 0.030026 |
| fp8_qk_fp8_pv + smooth_v（仿真） | 21555.812 | 0.030144 | 0.030144 | 0.030144 |
| fp8_qk_fp8_pv + smooth_qv（仿真） | 25044.496 | 0.029742 | 0.029742 | 0.029742 |

### 表 8：step_19_layer_47（step=19，layer=47）

输入：`torch.bfloat16`，shape `1×56×73923×128`，sigma=0.387096822。

| 方法 | 耗时 (ms) | 平均 relative L2 | 最小 relative L2 | 最大 relative L2 |
|---|---:|---:|---:|---:|
| BF16 SDPA benchmark | 714.860 | 0.000000 | 0.000000 | 0.000000 |
| int8_qk_fp8_pv | 291.962 | 0.019312 | 0.019312 | 0.019312 |
| fp8_qk_fp8_pv | 358.981 | 0.039234 | 0.039234 | 0.039234 |
| fp8_qk_fp8_pv + smooth_q | 401.617 | 0.038110 | 0.038110 | 0.038110 |
| fp8_qk_fp8_pv + smooth_v | 359.250 | 0.039001 | 0.039001 | 0.039001 |
| fp8_qk_fp8_pv + smooth_qv | 405.029 | 0.037861 | 0.037861 | 0.037861 |
| int8_qk_fp8_pv（仿真） | 21756.855 | 0.019315 | 0.019315 | 0.019315 |
| fp8_qk_fp8_pv（仿真） | 21579.279 | 0.039237 | 0.039237 | 0.039237 |
| fp8_qk_fp8_pv + smooth_q（仿真） | 25000.289 | 0.038113 | 0.038113 | 0.038113 |
| fp8_qk_fp8_pv + smooth_v（仿真） | 21580.836 | 0.039000 | 0.039000 | 0.039000 |
| fp8_qk_fp8_pv + smooth_qv（仿真） | 24957.490 | 0.037861 | 0.037861 | 0.037861 |

## 可复现文件

- 捕获配置：`capture_config.json`
- ComfyUI API 工作流：`workflow_api.json`
- Prompt：`prompt.txt`
- 原始结果：`sageattention_benchmark_results.json`
- CUDA 算子 Benchmark：`benchmark_sageattention.py`
- 仿真 Benchmark：`benchmark_sageattention_sim.py`
- 仿真/算子交叉验证：`cross_validate_sim_kernel.py`
- 报告生成器：`generate_benchmark_report.py`
