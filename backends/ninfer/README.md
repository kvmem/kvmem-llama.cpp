# NInfer · Windows 16GB · Qwen3.8-27B GSQ-RCO

**简体中文** | [English](README.en.md)

**RTX 5070 Ti / RTX 5080 / RTX 5090 · 本地推理与托盘管理**

https://github.com/user-attachments/assets/9695e989-e3f2-4727-8735-48d00910d90a

**[下载引擎和模型（夸克网盘）](https://pan.quark.cn/s/28b896c4b0c0)** · **[单独打开视频](https://github.com/user-attachments/assets/9695e989-e3f2-4727-8735-48d00910d90a)** · **[5070 Ti 中文指南](docs/rtx-5070ti-windows.md)** · **[English guide](docs/rtx-5070ti-windows.en.md)** · **[下载与模型转换说明](docs/rtx-5070ti-windows-downloads.md)**

面向希望在 **Windows、约 16GB 独立显存**上运行 Qwen3.8-27B GSQ-RCO 的用户。本分支整理了 CUDA 13 Native 引擎、Windows 托盘管理器与日常启动配置：通过网页管理模型、保存参数、查看运行监控，并提供 **GSQ-RCO GGUF → `.ninfer`** 的转换步骤。

下载目录包含 **Swift S 和 Swift XXS 两套模型**，模型合计约 **20 多 GB 磁盘空间**，完整运行目录当前约 **23.5 GB**，建议预留至少 **30 GB 磁盘空间**。这是文件占用，不是显存要求。

**本仓库提供的预编译引擎与配套 `.ninfer` 模型成品，发布支持范围仅限 RTX 50 系列；本分支实测硬件为 RTX 5070 Ti 16GB。** RTX 5080、RTX 5090 的显存容量、可用上下文和合适参数因硬件而异，本分支尚未逐卡实测。

**独立显存 16GB 及以上的 RTX 30 / RTX 40 系列，理论上也可以尝试，但本分支没有相应实测。** 其他系列需要针对目标显卡从源码构建引擎，并按相应配方转换模型；RTX 50 系列成品不能直接当作通用部署包。RTX 30 / 40 的常用构建架构分别为 `86` / `89`，当前成品引擎仅包含 `sm_120a` 机器码。不要直接套用相同的上下文容量或性能结论。可以把本仓库和对应部署指南交给 AI，协助完成源码构建、GGUF 转换和启动配置。

本仓库：**[ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco)**。上游汇总仓库：**[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all)**。原始推理引擎：**[Neroued/ninfer](https://github.com/Neroued/ninfer)**。

本分支在上游基础上提供 Windows / RTX 5070 Ti 构建、显存策略和管理器。下方保留上游项目的重要信息和贡献者出处，方便继续查阅。

---

## 上游项目概览

以下内容译自上游介绍。其中的硬件、模型、历史测试限制与性能数字属于上游工作，**不是本分支 Windows 16GB 运行包的实测或保证**。当前 RTX 5070 Ti 参数与性能请看 [Windows 部署指南](docs/rtx-5070ti-windows.md)。

上游汇总线将多个 [NInfer](https://github.com/Neroued/ninfer) 分支整合到一起，覆盖 RTX 3090、RTX 4090、RTX 5090 和 RTX PRO 6000 Blackwell，并在此基础上继续开发。基础来自 [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090) 的 `master`：包括 v0.11.0 和多 GPU 流水线阶段，其中大部分由 [Warlax](https://github.com/WarlaxZ) 编写，延续了 [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) 从 Neroued NInfer 开始的工作。

另外还整合了 [TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090) 的补丁、[UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) 及其贡献者的思路、提交给 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的开放 PR，以及以下作者的工作：

- [IMGillusion](https://github.com/IMGillusion/ninfer-disk-kv)
- [Mirko Covizzi](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile)
- Ian Ranson（[Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)）
- [tmark00](https://github.com/tmark00/ninfer)
- David Oelfke（[gzenz/ninfer](https://github.com/gzenz/ninfer)）

各项改动保留原作者署名；[维护者与改动对应表](docs/maintainer/consolidated-line.md)列出作者和涉及文件。

下文之外的引擎说明，例如构建、软件包、服务 API、支持模型和参数，可参考 **[NInfer-3090 原始 README](https://github.com/ashalliants/ninfer-3090#readme)** 与 **[NInfer-4090 原始 README](https://github.com/UDPSendToFailed/ninfer-4090#readme)**。上游新增的、会影响数值或服务行为的功能通常需要手动开启，有三项例外：

- 按显卡实测 device profile 选择内核路线；`--device-profile off` 保留编译时的路线表。
- 将 prefill chunk 对齐到显卡 SM 的完整调度批次；`NINFER_PREFILL_ALIGN=0` 保留用户请求的 chunk。
- TertiumOrganum1 的三值权重预填充 tile；`NINFER_T2_A8_TILE=off` 恢复被替换的内核。

## 上游参考性能

下列数字测于 2026 年 9 月，每种显卡各一张，采用 greedy 采样；除特别标注外，均为单请求。具体设置和完整数据见[参考测试](docs/performance/reference-2026-09.md)。

| 项目 | RTX 3090 | RTX 4090 | RTX 5090 | RTX PRO 6000 |
|---|---:|---:|---:|---:|
| **Ternary Bonsai 2 27B**，短对话（DFlash2，7 个草稿） | 202 tok/s | 256 tok/s | 397 tok/s | 381 tok/s |
| 读入 261K-token 文档后的解码速度（最快草稿方案） | 90 tok/s | 123 tok/s | 218 tok/s | 218 tok/s |
| 261K-token 输入的首 token 延迟 | 215 s | 102 s | 82 s | 78 s |
| 填满后仍找到全部三个检索目标的最大上下文 | 970,752 | 958,464 | 978,944 | 1,048,576\* |
| 八请求并发总吞吐（MTP，3 个草稿） | 551 tok/s | 824 tok/s | 1,063 tok/s | 1,155 tok/s |
| **Qwen3.8-27B**，短对话（DFlash2，7 个草稿） | 118 tok/s | 149 tok/s | 236 tok/s | 237 tok/s |
| 填满后仍找到全部三个检索目标的最大上下文 | 417,792 | 405,504 | 872,448 | 1,048,576\* |
| 八请求并发总吞吐（MTP，3 个草稿） | 329 tok/s | 442 tok/s | 690 tok/s | 739 tok/s |

\* 这是引擎上限。RTX PRO 6000（96 GB）在各种 KV 格式和草稿方案下都能以该容量启动；填满后，两个模型均找到三个目标中的两个。

- **与同卡此前的 `master` 比较：**261K-token Bonsai 输入在 RTX 3090 上从 315 秒降至 215 秒，4090 从 138 秒降至 102 秒，5090 从 115 秒降至 82 秒；24 GB 显卡上随后进行的 `rk4v4` 解码快了 11–13%。短上下文解码仍受读取权重带宽限制，速度基本不变。Qwen3.8 在 RTX 5090 上的 8K–32K 输入预填充反而慢了 8–10%。
- **RTX 3090 的 device profile：**262K 下的 `rk4v4` 验证注意力比编译路线快 3.2 倍。三张卡上的快速 prompt 内核配合 FP16 P·V 累加，使 prompt 注意力耗时减少 19–30%。
- **草稿长度：**DFlash2 的 7 个草稿最适合短回答；长文档后的最佳范围为 3–7。MTP 目前最多支持 15 个草稿，但最快通常为 3–5。
- **超过原生窗口：**在约 880K tokens 下，Bonsai 2 在所有卡上都找到三个预埋代码；达到 1,048,576 tokens 时，仅 RTX 5090 和 RTX PRO 6000 能容纳，但会漏掉约 943K 位置的目标。
- **RTX PRO 6000：**96 GB 显存让所有配置都能以 1,048,576-token 上限启动，并至少剩余 50 GiB。与 RTX 5090 相比，预填充快 4–6%，解码慢 2–3%。

## 上游汇总线增加了什么

- **GGUF 块量化格式。** 支持按张量选择 ggml 量化类型的 Qwen3.8-27B GGUF，例如 ISTA-DASLab 的 GSQ-RCO，无需重新量化。`qwen3_8_27b_gguf` 配方原样复制每个量化张量的数据块；运行时直接计算全部 15 种 dense ggml 块格式。解码和验证使用向量内核，每列只解码一次权重；prompt 使用 llama.cpp 的整数 Tensor Core 内核。MTP、DFlash2、Vision 的使用方式与官方产物一致。3.5-bit GSQ-RCO IQ3_S 的 WikiText-2 困惑度为 7.071，与模型卡的 7.07 一致，官方产物为 7.286；权重占用从 15.9 GiB 降为 10.95 GiB。IFBench 为 80.3%，AIME 2025/2026 均为 100%，GPQA-Diamond 为 88.4%；官方产物分别为 77.7、96.7、96.7、87.4。无推测解码时，RTX 3090 为 59.9 对 40.3 tok/s，RTX 5090 为 107.5 对 88.1；RTX 4090 开启 MTP 后为 146 对 109 tok/s。详见 [GGUF 块格式](docs/gguf.md)。
- **按显卡实测的设备路线。** 每种操作和宽度优先查显卡 profile，再查针对单卡调优的编译路线表。已内置 RTX 3090、4090、5090 及 RTX PRO 6000 Blackwell 三个版本（Workstation、Max-Q、Server）的实测 profile。其他 GPU 首次启动自动校准一次，约 20–40 秒，随后保存结果；`ninfer-calibrate` 可按需重测。3090 的 262K `rk4v4` 验证注意力快 3.2 倍；三张卡都启用 FP16 P·V 累加（该注意力耗时减少 15–17%，困惑度不变）与快速 prompt 内核（prompt 注意力耗时减少 19–30%）。当两个候选 token 分数接近时，同一 greedy 请求单独执行或与其他请求组成批次，可能比以前更容易给出不同答案。`--device-profile off` 保留此前 master 的编译调度。详见[设备 profile](docs/device-profiles.md)。
- **Blackwell 默认构建中的 FP8 / NVFP4 加速。** 所有 `120a` 构建都会编译 FP8 A8 与 NVFP4 W4A4 Tensor Core 单元，`mma.sync` 兼容路径也能使用其专用路线。此前 NVFP4 产物会在该路径启动失败，FP8 预填充则走反量化路线。RTX PRO 6000 上，Qwen3.8-27B NVFP4/FP8 产物的 4096-token 预填充为 11,822 tok/s，与 Native 构建差距在 1.4% 内；Qwen3.6-35B-A3B NVFP4 为 30,938 tok/s。
- **更快的长上下文注意力。** INT8 系列 small-T 内核增加了分层策略，在 producer warp 之间分摊 QK 乘积，并提前一整轮读取下一块 key。快速 prompt 内核支持 `rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8`；所有 prompt 内核都将 chunk 对齐到完整 SM 调度批次，沿用 Ian Ranson 快速内核的思路。RTX 3090 的 131K `rk8v4` 输入耗时从此前 master 的 101 秒降至 76 秒。
- **MTP 最多 15 个草稿。** 超过 8 列验证窗口时，按上下文区间分别构建 CUDA Graph，使 `--spec mtp --draft-tokens 10..15` 能正常启动；此前会在 graph update 时失败。
- **BF16 KV 与 CUDA Graph。** 修复此前 master 中默认 BF16 KV 的 MTP 启动失败，以及 Qwen3.8 在无推测解码、512/1024 上下文时的启动失败。原因是 Graph 规划器把 BF16 使用 prompt 内核（最多 128 keys）和使用 small-T 内核的窗口共用了同一 executable；现在会向 attention op 查询每次捕获实际选择的路线。
- **完整参考测试。** 覆盖 RTX 3090、4090、5090 上的 Ternary Bonsai 2 27B / Qwen3.8-27B：完整窗口、各卡能启动并填满的最大上下文、1–15 个草稿、多请求并发，以及同机器上的此前 master。见 [2026 年 9 月参考测试](docs/performance/reference-2026-09.md)。
- **Ternary Bonsai 2 27B。** PrismML 的[三值 Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) 使用 `t2_g128_fp16` 权重，每权重 2.125 bits，从 PQ2_0 GGUF 导入时不重新舍入。检查点中的 Hadamard 旋转融合到各投影输入的 norm 和 gate；token 表与输出头仍保持三值。`bonsai2_27b_ternary` 配方加入 ProCreations 为 Bonsai 训练的 MTP 头、DFlash2 adapter，以及精确 proposal 头。
- **三值投影的整数激活。** 解码、推测验证以及不超过 192 tokens 的 prompt 使用 s8 激活 small-T 内核；更长输入使用 int8 激活 GEMM，将不整齐的尾部补齐到成本最低的 tile。输出头、draft 头、proposal 头也走同一路线。
- **RTX 3090 调优。** 旋转 producer 每个 1024 点变换使用 4 个 warp；INT8 KV 的 small-T 注意力按完整 SM 批次启动分片；GDN 记录将窗口暂存共享内存；三值目标模型的 DFlash2 adapter 使用 Q4。
- **DFlash2 与 Vision 叠加驻留。** 图像编码可以借用 drafter 的显存，使 DFlash2、Vision 和完整 262,144-token 窗口同时装入一张 24 GB 卡。
- **服务修复。** 强制 `tool_choice` 会在生成提示中打开指定调用，模板默认思考让位于该要求。上下文缓存无法容纳请求时只让该请求失败（HTTP 429），不拖垮引擎。Paged KV 耗尽会报告页号，连续三次耗尽将引擎标为不健康。修复私有缓存回收、demand window 和零价值候选的捕获搜索，减少长 agent 会话重新预填充。
- **构建。** 测试适配 CUDA 13 的 `cudaGraphGetEdges`。

### 来自 TertiumOrganum1 的工作

来源：[TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090)。

- **`rk4v4-e8` KV。** Key 与 `rk8v4` 一样先旋转，再按 8 维组映射到 int4 的 E8 格点；Value 保留 `rk8v4` 的 int4 平面。E8 KV 编码最早由 UDPSendToFailed 与 Daniel Parker 在 NInfer-4090 中引入。每 token、每 KV head 占 280 bytes，而 `rk8v4` 为 408。Ternary Bonsai 2 的 262,144 窗口可节省 2.0 GiB；两个并发槽各放一个完整窗口（共 524,288 tokens，`rk8v4` 为 519,744）后还剩 5.7 GiB。131K / 250K 的三个预埋目标仍全部找到；快速语料困惑度从 5.631 变为 5.650。
- **三值预填充的 128×64 int8 tile。** 激活按 token 和 128 列组量化，int32 求和覆盖整个权重组。与被替换的内核（`NINFER_T2_A8_TILE=off`）相比，Bonsai 2 在 8K、32K、64K 下的预填充分别快 33%、21%、15%，快速语料困惑度保持 5.631。
- **工具调用。** 对格式损坏的调用区域尽量恢复已读内容，避免把标记泄漏到答案中。
- **共享前缀捕获。** 若替换共享前缀捕获实际释放的空间少于预估，就放弃该候选；此前会使引擎持续失败并返回 503，直到重启。
- **构建。** `sm_120a`（RTX 50 系列）可使用 `mma.sync` 兼容路径构建。

### 来自 NInfer-4090 的工作

来源：[NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090)，除标注外作者为 UDPSendToFailed，已在汇总线中重新实现。

- **全程序 CUDA 构建**（Matt Anderson）。Core 和 ops 静态库不再使用可重定位设备代码，ptxas 能内联按动态 lane 计算的 shuffle，并跨循环安排加载；需要 stack frame 的内核减少约五分之一，服务端二进制增大约四分之一。
- **从共享内存读取 scale。** INT8 注意力的 query/key/value scale 改为从共享内存读取，不再从计算出的 lane 做 shuffle。配合全程序构建，`rk8v4` 验证注意力耗时减少 9–21%；Bonsai 2 在 64K 下的 MTP 步耗时减少 6.6%，8K 及以上预填充快 2–7%，答案不变。
- **服务 socket 使用 `TCP_NODELAY`。** 流式 token 写出后立即发送。
- **可选 SFU sigmoid / SiLU / softplus。** `-DNINFER_SFU_SIGMOID_SILU=ON` 用 `ex2.approx` 和正确舍入的倒数替代 `expf` 与除法。Bonsai 2 在 8K 以上预填充快约 2%，快速语料困惑度从 5.6306 变为 5.6309，MTP 解码不变。`-DNINFER_SFU_SOFTPLUS=ON` 对 GDN decay gate 的 softplus 使用同样思路；当 e^x 小于 1/16 时改用 log1p 级数，保留长记忆 head 的慢衰减精度，困惑度为 5.6302。
- **超过 262,144 的 key。** 当分片跨度超过暂存的 64 个页 ID 时，small-T 注意力从块表读取每页物理索引，将可见 key 上限提高到 1,048,576。
- **四倍原生窗口与 YaRN。** 原生 262,144 模型的 `--max-context` 可达到 1,048,576。超出原生窗口后可用普通 RoPE，或通过 `--rope-yarn` 使用 YaRN；默认 factor 为 Qwen 文档中的 `--max-context / 262144`，计算与 Hugging Face / vLLM 一致。Bonsai 2 配合 `rk2v4-e8` 的三目标测试，在文档 33%、66%、90% 位置放入代码：500,000 tokens 不开 YaRN 找到全部三个；开启后在 131,072、500,000、1,000,000 下都只找到两个。因此默认不开，除非普通 RoPE 已无法回答。`--rope-yarn-factor F` 可为所有位置固定 factor，不随窗口变化。
- **`rk2v4-e8` KV**（与 Daniel Parker 合作，他也通过 Neroued/ninfer#173 向上游提出该方案）。旋转并按 G64 缩放的 key 每 8 维仅占两个字节：一个记录最近的 E8 240 根向量，另一个记录 4-bit 对数半径和有符号残差轴。每 token、每 KV head 为 216 bytes，`rk4v4-e8` 为 280，`rk8v4` 为 408。这是 24 GB 卡上能与 Bonsai 2 同时容纳 1,048,576 tokens 的格式；不开推测剩 2.8 GiB，MTP 剩 1.3 GiB。两个完整 262,144 窗口剩 7.7 GiB。代价是质量：快速语料困惑度从 5.631 增为 5.820（`rk4v4-e8` 为 5.651），DFlash2 草稿接受率从 54.4% 降为 51.8%，解码慢 4%。131,072 / 250,000 tokens，以及 1,048,576 窗口中的 500,000-token 输入，都找到三个目标。
- **Windows D3D12 驻留内存池**（与 keylimesoda 合作）。`-DNINFER_D3D12_RESIDENCY=ON` 提供 `--wddm-evictable-budget`：设备内存池来自最高优先级驻留的共享 D3D12 heap，再导入 CUDA；KV 容量按 WDDM 可驱逐其他分配来安排。上游记录当时没有 Windows 测试机，因此只做过 MinGW 语法检查，并非运行验证。

其他来自 4090 分支的思路包括：服务端默认思考级别、最多 15 个 MTP 草稿、`/metrics` 和 `/slots`（Sergiusz Michalik）及 `/props`、从 `NINFER_WEBUI_DIR` 编译内嵌 WebUI、仅受上下文约束的输出上限、块采样器候选驻留共享内存、可选 bf16 残差加法（`-DNINFER_BF16_RESIDUAL_ADD=ON`）、分块 GDN 预填充的向量存储，以及受限分片编译和 ptxas 报告构建选项。

### 来自其他分支的工作

- **Host 缓存下方的磁盘层**（[IMGillusion](https://github.com/IMGillusion/ninfer-disk-kv)）。`--disk-kv-path DIR` 将被驱逐会话的 KV 页与状态图像写入按 prompt 摘要索引、带 CRC 检查和 LRU 管理的文件，重启后仍保留。`--disk-kv-restore` 可从相同已存前缀恢复，只预填充剩余部分。Bonsai 2 + MTP 中，一个 17,444-token 输入被另两个请求驱逐后，从磁盘恢复耗时由 9.6 秒降至 1.2 秒，重启后为 1.0 秒，答案一致；DFlash2 与无推测解码也可恢复。Windows 的 `-DNINFER_DIRECTSTORAGE=ON` 构建可用 `--disk-kv-directstorage` 经 DirectStorage 读取；上游当时与 D3D12 一样尚未实测。
- **自适应 MTP**（[Mirko Covizzi](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile)）。`--adaptive-mtp` 根据实测草稿存活率和轮次成本，每轮在 K 个草稿中验证 3..K 个，并为各宽度准备 CUDA Graph。RTX 3090 上的 Bonsai 2、K=5，59% 轮次验证 5 个，23% 验证 4 个，18% 验证 3 个；没有超过固定 K=3：短输入为 200 对 204 tok/s，8K 为 150 对 163。各宽度 Graph 也占显存，因此 Huihui 的 198,400-token 缓存在开启后无法再装入 24 GB 卡。宽度改变可能让接近同分的 token 结果不同，与切换固定窗口时类似。
- **快速 INT8 prompt 注意力**（Ian Ranson，[Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)）。每个 warp 将 query 行、score、output 留在寄存器中，每 64-key tile 用 FP16 累加 P·V。汇总线将其扩展到 `rk8v4` 和打包 key 编码，并让 device profile 在更快时开启；三张实测卡的 prompt 注意力耗时减少 19–30%。`--fast-prefill-kernel` 可强制开启。Bonsai 2 的 64K `rk8v4` 快速语料困惑度从 5.2074 变为 5.2079，131K 三目标测试全部找到。
- **Agent 工具调用格式。** 除 Qwen 格式外，支持 `<function name=...>`、`<invoke name=...>`、`<function_calls>`、`<param name=...>`，并经过同一恢复流程；来自 Pavel Kochubey 的上游 PR #300，经 Wallawalla47 引入。
- **结构化输出。** `--structured-output` 启用 xgrammar，支持推测解码；来自 Andrey Shvartsman 的上游 PR #294。
- **首 token 对数概率。** 开启 `--first-token-logprobs` 后，Chat Completions 请求可使用 `top_logprobs` 获取第一个生成 token 及其候选的对数概率（IMGillusion）。
- **滚动保留。** `--context-cache-policy rolling` 让一个长会话不断向前移动已缓存边界（IMGillusion）。
- **释放分叉检查点。** `--release-diverged-checkpoints` 优先丢弃所属会话已经不再使用的私有检查点；Ian Ranson 根据 pkochubey 的上游 PR #300 实现。
- **Blackwell NVFP4 专家权重库。** Mykhailo Dementii 的上游 PR #286–#290。Qwen3.6-35B-A3B NVFP4 检查点可用 `--recipe qwen3_6_35b_a3b_nvfp4` 转换，并运行于任意 `120a` 构建。RTX 5090 Native 构建（`-DNINFER_SM120_NATIVE=ON`）上，20.6 GB 纯文本产物的 4K 预填充为 27,663 tok/s、解码 397 tok/s。预填充把激活量化为 4 bits，走 Blackwell 独有 W4A4，所以 sm_8x 构建会拒绝这种权重库。
- **N-gram 复制草稿**（remesis、Ian Ranson）。搭配 drafter 时，可根据最近 12 个 token 的匹配，从已有 prompt、工具结果或输出中复制最多 15 个 token 供验证（`--ngram-draft-tokens`、`--ngram-min-match`）。开启 `--spec` 后默认启用；每个复制 token 都由目标模型验证，属于精确推测。可选 RAM 归档 `--ngram-archive-mib` 为同一 `X-NInfer-Draft-Session` 后续请求保留已完成请求的复制来源。
- **Hybrid 前缀缓存**（Ian Ranson，[Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)）。`--use-alt-prefix-caching` 用按内容寻址、跨请求共享的 64-token KV 块和稀疏状态快照替换检查点目录，按可用显存和一个 `--host-cache-mib` Host 预算安排容量。围绕默认目录还增加了可选近期访问驱逐、先降级到 Host（`--recency-eviction`），按需扩展回答的 Device KV 租用量（`--kv-lease-growth`），统一 Host 保留预算（`--host-cache-mib`），对重写会话自动设置消息边界锚点（`--auto-long-anchors`）；默认还会复用中止请求已预填充内容，并在目录满时以 LRU 替换自动共享前缀。
- **请求接纳与缓存驱逐**（Gideon Zenz、David Oelfke、Ian Ranson）。`--thorough-admission-search` 最多花 250 ms 检查所有复用候选；`--value-aware-demote` 按检查点重建成本排序驱逐；`--concurrent-prefill` 在其他请求预填充时接纳新请求；`--recover-invariant-failures` 在内部不变量失败后继续服务。
- **草稿与采样**（Gideon Zenz）。`--mtp-attention-window N` 让 MTP 草稿头只关注开头 64 个 key 和最近 N 个，避免草稿读取量随历史持续增长，最终 token 仍由目标模型验证。思考结束后的采样可切到单独预设（temperature 0.2），支持服务端 `--post-thinking*` 或请求中的 `post_thinking` 对象。
- **服务功能**（Gideon Zenz、Ian Ranson）。`GET /stats` 提供全部 Engine 计数和等待队列，可用主端口或独立 `--stats-port`；[`tools/monitor`](tools/monitor/README.md) 提供终端仪表盘与卡死 watchdog；支持请求日志轮转（`--request-log-max-mib`）。还提供 `--assistant-prefill`、`--unconstrained-response-format`、`--lenient-assistant-history`、`--derive-session-keys`；Anthropic 流每个心跳发送协议 `ping` 事件；分组 `--help`、`--log-colours`、统计面板 `--log-stats-panel`，以及每个产品二进制中的 build id。
- **CPU 视觉与位置插值**（David Oelfke）。`--vision-residency cpu` 使用 Host FP32 权重和 CPU 线程编码图片，不分配设备 Vision 显存；`--rope-scaling-factor` 配合 `--rope-scaling-original-context` 对超过原生窗口的位置做插值。
- **内核与转换**（Ian Ranson、Duncan Betts）。包含兼容构建的 decode Graph 程序依赖启动（`-DNINFER_PDL=ON`）、长上下文上短 prefill 步的 split-KV 注意力、通用 BF16 GEMM fallback、混合格式 MTP 权重库、所有宽度的融合 RMSNorm 与 NVFP4 attention 输入，以及 ModelOpt NVFP4/FP8、Quasar NVFP4 和 `grouped_mse` scale 搜索转换器；还提供基于预构建 vcpkg 树的 Windows 原生构建。
- **统一 Linear 模板**（Neroued）。上游带 sliced-K 调度的 Q4/Q5/Q6/Q8 A16 模板与汇总线原路线并存。每张卡只在两轮实测更快的宽度使用它们：Q5 从约 8 列到 96（3090）、128（4090）、1024（5090），各形状快 1.5–1.7 倍；Q6 为 4–32 列；Q4 从 25 列开始；Q8 的适用宽度随卡变化。Q4 解码和验证宽度保留原内核。`NINFER_LINEAR_ROUTES=legacy|unified` 可强制路线表。
- **FP8 / NVFP4 / BF16 模板与融合投影**（Neroued）。上游统一模板及各格式融合投影，包括 attention/GDN 输入及卷积形式、LinearAdd、SwiGLU、Q8 pair、top-k heads、Q8 grouped convolution、context-KV materialization，都按同样原则与原路线并存：每种形状和 Op 测两轮，在哪些宽度更快才选用。3090 的原 FP8 / NVFP4 A16 路线会在宽输入上循环 small-T，统一模板在验证和预填充宽度分别快 1.7–7 倍、2.5–44 倍，4090 类似；5090 的 FP8/NVFP4/BF16 在多数 A16/A8/A4 宽度快 1.1–3 倍。Q8 投影多数宽度保留原路线；统一 SwiGLU 的 gate/up 投影在激活阶段保持 FP32。
- **两阶段 GDN 预填充**（Neroued）。16 tokens 及以上的 prompt chunk 可以先完成 Q/K 归一化、gate 因子和每 chunk 求解，再做 FP32 状态递推并写出结果，替代原 WY、状态传递和输出内核。3090/4090/5090 在 16–8192 tokens 各宽度的 GDN Op 快 1.4–4.3 倍，内置 profile 因而采用此路线；由于递推只占 prefill 一小部分，3090 的 Qwen3.8 27B 整体 Engine prefill 变化约 1%。`NINFER_GDN_TWO_STAGE=0|1` 可强制选择。
- **PackGQA**（Gideon Zenz）。INT8 prompt 内核可把同一 KV head 的 query heads 打包进 tile（`NINFER_PROMPT_PACK_GQA=1` 或 profile 的 `attn_pack_gqa`）。1024-token chunk 在 32K / 131K 上下文中，3090 快 2.7%，4090 / 5090 分别慢 0.5% / 3.9%，因此内置 profile 均未开启。
- **引擎与服务修复。** 包括 worker OOM 恢复（David Oelfke 编写、Ian Ranson 移植）；`--kv-headroom-mib`、`--cuda-graph-allowance-mib`、`--thinking-budget-message`（Ian Ranson）；`--webui-mcp-proxy` 转发 WebUI MCP 流量、E8 根码查表解码、按 SM 数量设置 RMSNorm 阈值（[tmark00](https://github.com/tmark00/ninfer)）；带拓扑分类的 MTP Graph profile（Mykhailo Dementii，上游 PR #221）；可直接打开的服务 URL 与 CORS 预检回显（pelebel、natpate）。还有 GGUF 转换源（giveen）、Q6 配方（bingchengcc）、稀疏 MoE/NVFP4/attention epilogue 调优（Mykhailo Dementii、Duncan Betts、MOVIBALE）、带引号标记与重复参数工具调用修复（Fedor Suchkov、adubkov）、Copilot 工具格式（Damian Sromek）。

[维护者与改动对应表](docs/maintainer/consolidated-line.md)列出各项改动涉及的文件和测试。

## 运行示例

以下是上游参考测试使用的命令，**不等于本分支 16GB 运行包的默认配置**。从下方模型表选择相应产物并交给 `ninfer-serve`。原始命令行服务默认在 `127.0.0.1:8080` 提供 OpenAI / Anthropic API，自动选择显卡 profile；[参考表](docs/performance/reference-2026-09.md)记录了这些配置的实测。

<details>
<summary>Ternary Bonsai 2 27B：单请求综合配置，DFlash2 五草稿与完整 262,144 窗口</summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

5 个草稿是综合选择：短回答 7 个更快，长文档后最佳为 3–7 个，见[草稿长度测试](docs/performance/reference-2026-09.md#draft-length)。图片输入可添加 `--vision --vision-residency overlay --vision-max-merged 12288`；图像编码借用 drafter 显存，完整窗口仍可装入 24 GB 卡。
</details>

<details>
<summary>Ternary Bonsai 2 27B：通过 proposal 头生成 MTP 草稿</summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```
</details>

<details>
<summary>Ternary Bonsai 2 27B：24 GB 卡的大上下文示例，958,464 tokens，<code>rk4v4</code></summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 958464 --kv-capacity 958464 --kv-dtype rk4v4 --gdn-state-fp16 --rope-yarn
```

RTX 5090 使用 `rk4v4` 可容纳 1,048,576-token 上限，包括 DFlash2 或 MTP；`rk8v4` 为 978,944。填到 1,048,576 时，无论 YaRN 还是普通 RoPE，都找到 33%、66% 的代码，但漏掉 90%（约 943K）的目标；约 880K 以内，三张卡均找到全部目标。
</details>

<details>
<summary>Ternary Bonsai 2 27B：自适应 MTP，加磁盘层保留被驱逐的会话</summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 5 --lm-head-draft --adaptive-mtp \
  --disk-kv-path /var/cache/ninfer --disk-kv-gib 64 --disk-kv-restore
```
</details>

<details>
<summary>Qwen3.8-27B：24 GB 卡，DFlash2 五草稿、245,760-token <code>rk4v4</code></summary>

```bash
ninfer-serve Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer --model-id qwen3.8-27b \
  --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

相同推测方案换成 `rk8v4`，RTX 4090 可放 167,936 tokens，3090 可放 176,128；RTX 5090 两种格式都能放完整 262,144。
</details>

<details>
<summary>为没有内置 profile 的显卡测量配置</summary>

```bash
ninfer-calibrate --print > my-gpu.json
```

引擎首次启动会自动执行；手动运行可在驱动或时钟变化后刷新 profile。详见[设备 profile](docs/device-profiles.md)。
</details>

## 上游模型产物

这张表保留上游发布的模型，体积、组件和要求各不相同；它们不是顶部下载目录中那两份 Swift text/MTP 模型的说明。

| 模型 | 产物 | 说明 |
|---|---|---|
| Ternary Bonsai 2 27B | [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 GiB。三值文本网络、token 表和输出头，含 Vision、Bonsai 专用 MTP 头、DFlash2 adapter 与精确 proposal 头。仅适用于此汇总线。 |
| Qwen3.8-27B GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 GiB。逐字节保留 ISTA-DASLab 的 3.5-bit GGUF 块，包含其 Q6_K MTP 头、Vision、DFlash2 adapter 和 proposal 头。仅适用于此汇总线。 |
| Qwen3.8-27B | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19 GiB，`groupwise-int`（Q4/Q5）；参考测试使用的官方产物。 |
| Qwen3.8-27B，abliterated 版本 | [WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 GiB，官方 `qwen3_8_27b` 配方，含 MTP、DFlash2 和 proposal 头。 |
| Qwen3.6-35B-A3B NVFP4 | [WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 20.39 GiB。原样保留 RedHatAI NVFP4 专家权重编码，含 Q8 投影、Vision、MTP、proposal 头，需要 `sm_120a` GPU。 |

原始 README 列出的官方 NInfer 产物也可在此汇总线加载。[权重转换说明](docs/weight-conversion.md)介绍 [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) 和 [GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) 产物的生成方法。

## 构建示例

<details>
<summary>Linux + CUDA 13.1</summary>

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve ninfer-calibrate
```

`CMAKE_CUDA_ARCHITECTURES`：RTX 30 系列为 `86`，RTX 40 系列为 `89`，RTX 50 系列和 RTX PRO 6000 Blackwell 为 `120a`，后者使用三值路线需要的 `mma.sync` 兼容路径。可选开关见 [Linux 构建指南](docs/rtx-3090-linux.md#build-options)。上游 Windows 构建、发行包、测试和基准流程参见 [NInfer-3090 README](https://github.com/ashalliants/ninfer-3090#readme)。本分支 Windows RTX 5070 Ti 的构建方法见顶部专用指南。
</details>

## 许可证

与上游一致，采用 Apache-2.0。Bonsai 产物中的权重来自 PrismML、ProCreations 和 Qwen，均为 Apache-2.0；相关声明见其模型卡。
