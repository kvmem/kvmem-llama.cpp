# NInfer 多卡文本与 Vision

## 实现来源

保持 NInfer 子模块基线 `6e9189c1089f6968e29fe2e1c51b4596e2346eed`，现有 KVMem 集成继续保存在 `backends/patches/ninfer-kvmem.patch`。不更换模型制品格式，不引入 TP/NCCL，也不复制 llama.cpp 的 ggml 搬运代码。

直接复用已有 StagePlan、StageLink、逐卡权重/workspace、共享 page-group allocator、逐卡 execution table、StateImage/GDN replay 和 MTP。补入上游的四个修复：

| 来源 | 提交 | 用途 |
| --- | --- | --- |
| ninfer-all | [d36b0cbd](https://github.com/iamwavecut/ninfer-all/commit/d36b0cbda26011abed280943d6574147a62ff016) | Host block-table shadow 重写前等待已排队复制 |
| ninfer-all | [b2f99a0f](https://github.com/iamwavecut/ninfer-all/commit/b2f99a0fa0b2ce26ad984d6be3eddaef2230e223) | 在持有 execution table 的设备上发布复制 |
| ninfer-all | [66c0f3b3](https://github.com/iamwavecut/ninfer-all/commit/66c0f3b32ab57415e5248da0fabd3699cd9bf51e) | KV/状态页复制时绑定实际持有内存的设备 |
| ninfer-3090 | [e7f5ce88](https://github.com/ashalliants/ninfer-3090/commit/e7f5ce88ff758025747a5f1fd826b419616ebcca) | rank 0 Vision handoff 与多卡文本流水线，resident tower 纳入分层规划 |

Vision 补丁的 startup hunk 按当前 KVMem 代码手工合并；CPU Vision 不计入 GPU tower 权重。Windows 物理多卡门禁及 KVMem 的多设备门禁被移除。KVMem 新增适配是按层 owner 的 mean-K/Q、candidate keys 和 ranges：每卡独立计算，Host 按全局 Attention 层顺序收集，prefill 的逻辑起点随原生 control link 分发。query token 数只增加一次，MTP/ngram 只提交已接受前缀。

真实 IQ3_S 制品验收还暴露了上游加载遗漏：`Bindings::place` 只放置主参数，48 层 GDN 共用的 `input_columns` 留在 rank 0。无 P2P 的第二张卡读取它，Compute Sanitizer 在 `quantize_matrix_kernel` 报告非法读取。为此扩展现有 Binder/Materializer，显式复制共享不可变辅助张量，并把每个 Use 的列索引／Hadamard signs 绑定到消费层所在卡。同一对象在同一 rank 只上传一次；主矩阵跨层共用且要求不同 rank 时仍拒绝。单卡加载不申请副本。物理两卡的副本归属及逐字节检查、模型 Use 重绑定测试均已通过。

## 预算与启动

一个逻辑历史、一个选页结果、一份 H 预算。Main 页池保留 C×(B+R)，各卡持有属于自己层的 KV；MTP 页池和草稿层留在 rank 0。每卡统计与 workspace 纳入该卡的精确容量曲线，最终容量受最紧张的卡约束。

统一启动器使用 `-Gpu 'GPU-a,GPU-b'`，可选 `-StageLayers 40,24`；直接运行服务使用 `--devices 0,1`。UUID 顺序决定层顺序，resident Vision、MTP、embedding/head 都在第一张卡。`/props.execution` 返回最终设备和层数。启动例子见 [多后端说明](multi-backend.md)。

## 验收环境与执行

2026-10-06 至 10-07，原生 Windows，CUDA 13.2，MSVC 14.44，sm_120a，8 路编译。制品为本地 `Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`。

| 顺序 | 卡 | UUID |
| --- | --- | --- |
| 0 | RTX 5060 Ti 16GB | GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c |
| 1 | RTX 5050 Laptop 8GB | GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29 |

两卡均为 CC 12.0，互相没有 P2P。边界传输复用上游 portable pinned-host ring。早期独立 StageLink 探针已通过 20,480 bytes／1 MiB、eager／Graph／capture 后 eager 的逐字节检查；它不能代替模型验收。

本轮结果目录：`backend-p0p1-20261002/results/ninfer-multigpu-20261006`。`scripts/test-ninfer-multigpu.py` 提供 `core`、`native`、`text`、`vision`、`formats`、`edge` 分层运行，各层写入独立日志与 JSON，真实模型用例顺序执行。运行前检查 UUID 和显存占用。

```powershell
python scripts/test-ninfer-multigpu.py --phase text `
  --model D:/models/model-vision-mtp.ninfer --build build-backends/ninfer `
  --gpu 'GPU-first-uuid,GPU-second-uuid' --output D:/results/multigpu `
  --red D:/fixtures/red.png --blue D:/fixtures/blue.png
```

`native` 复用上游 greedy/prefix/MTP oracle；`text` 验证越窗检索、checkpoint、取消、Host 会话、并发、adaptive/fast；`vision` 验证原生与 KVMem 的 resident/CPU 图片、换图、复制草稿和图文并发；`formats` 补其他原生 KV 页恢复；`edge` 将第一段设为无 Attention 的 3 层，反转物理卡顺序并检查 MTP 会话恢复。

本机两卡分别有 36／20 个 SM，GGUF Stream-K 按 SM 数划分归约。独立 IQ3_S 矩阵探针在 655,360 个 FP32 输出中发现 438,117 个末位差异，91 个输出在转 BF16 后仍不同，relative RMS 为 `1.97184e-7`、relative max 为 `5.11032e-7`。这是原有算子的数值路径，未修改它来固定两卡归约顺序。

保留的上游随机 token oracle 显示：单卡与双卡短／长生成各 24 token 完全相同，前缀复用量同为 1,324；恢复后 8 token 的续写有差异，切分点不同也可能改变续写。两段都放在 5060 Ti 时，全部输出／续写与单卡逐 token 一致；实体两卡固定层归属时 Graph、eager 和强制 Host 中转彼此一致。因此这台异构 SM 数机器的 `native` 验收使用显式 `--same-placement-reference`：每种层切分仍要求 Graph／eager、传输和缓存续写逐 token 一致，不将跨卡归约当作位精确保证。原严格比较和失败日志保留在结果目录中。

## 当前结果

功能验收完成，启动器参数映射和无效组合检查通过。真实硬件资格限定于上述物理两卡，未将单卡 C8 的历史结果算作多卡 C8 结果。

| 层次 | 实际结果 |
| --- | --- |
| 核心 | 6 项通过：StagePlan/StageLink、物理多 rank 页池、容量、Engine/Serve 参数、辅助权重绑定、物理副本归属与字节比较 |
| 原生模型 | 固定层归属的 Graph/eager、Host 中转、40/24 不均匀切分和 MTP 生成／缓存续写逐 token 一致；严格跨卡切分 oracle 的数值限制见前文 |
| 文本 KVMem | 5 组通过：越窗检索、Host 会话／淘汰／取消／编辑、MTP、C2、adaptive、fast |
| Vision | 7 组通过：native/KVMem 的 resident/CPU、图片身份和旧图检索、换图、取消、精确复制 ngram、fast 图文 C2、多张实际 1024-token 图片 C2 |
| KV 格式 | 9 种均通过物理两卡会话检查：BF16、INT8、FP8、NVFP4、K8V4、RK8V4、RK4V4、RK4V4-E8、RK2V4-E8 |
| 边界切分 | 反转物理卡顺序，5050 首段 3 层、5060 Ti 后段 61 层；首段无 Attention，MTP/Graph 的 Host 会话、取消和编辑恢复通过 |
| HTTP | resident/CPU 各 5 请求通过：两路实际 1024-token 图片、颜色／会话码隔离、缓存复用、共享 H 上限、warm compact batch、流式文本及资源退役 |

文本覆盖 INT8 eager、INT8 Graph/MTP3、会话压力和取消恢复、C2 compact decode，以及 RK8V4/adaptive/fast。Vision 覆盖 native/KVMem resident eager ordinary、native/KVMem CPU Graph/MTP3 的图片身份、旧图检索、换图、多图、预算拒绝及取消恢复；这些身份夹具限制每图 256 merged tokens。RK8V4 Graph/自适应 MTP3/ngram15 的 C1 复制诊断在默认 prefill 路径通过；fast/RK8V4/resident/C2/固定 MTP3/ngram31→15 的精确文件复制、恢复、取消、恢复同伴与颜色检查通过，每图实际 1024 tokens。INT8/CPU/C2/自适应 MTP3 并发夹具每请求含两张各 1024-token 图片，B3072/R128/H1GiB，覆盖旧图检索、换图、取消恢复和同会话提交顺序；恢复阶段 peak=2、rows>rounds，页数据与 GDN 状态均通过逐字节检查。

首次多图片并发测试误用 B1536，按既有合同拒绝保留总计 2048 个图片 tokens；保留拒绝日志，改为 B3072 后完成上述验收，没有改变预算算法。

`memory_vision copy-compare` 的 RK8V4/fast/C1/自适应 MTP3/ngram15 诊断在双卡、同卡两段及单卡均失败：冷输出 token 1 为 271（两个换行），恢复为 198（一个换行），均为 91 tokens，正文和图片颜色相同。页数据与 GDN 状态搬运检查通过。该现象与已有 [单卡宽窗口记录](vision-ngram-all-kv-validation-20261006.md) 相同；根因尚未确定，不能宣称所有图文自由生成的冷／恢复输出位精确。关闭 fast 后，同一双卡复制诊断通过。保留全部失败日志和原断言；本轮没有更改算法或宽松比较来规避它。fast 图文/ngram 的并发资格另用现有 `memory_ngram` 源文件精确复制夹具检查，包含恢复、取消隔离及正文逐 token 断言。

HTTP 长自由回答的冷／恢复文字比较同样有差异：单卡 resident 两路均不相同，双卡 resident 为一相同一不同，双卡 CPU 两路均不相同；颜色和独立会话码均正确，搬运检查均通过。冷 prefill 与 warm compact decode 的 batch/verify 形状不同。此处只确认为单卡也能复现的数值／自由生成限制，没有宣称确定了全部根因。首次严格文字断言失败的请求及日志保留；HTTP 验收检查协议交付、语义身份、复用和预算，精确正文／token 恢复仍由 `memory_ngram` 夹具的原严格断言检查。没有修改生产算法。

`scripts/test-ninfer-multigpu-http.py` 可以重跑该接口验收，`--single-device-reference` 提供同工作负载的单卡对照。服务使用 40/24 分层，C2、INT8、B1536/R128/H1GiB、自适应 MTP3，记录 `/props` 的实际设备／层数及完整响应。首次 HTTP 启动因测试脚本给子进程传入相对日志路径失败，改为绝对路径后重跑；未修改服务行为。

本地 worker 为 `build-backends/ninfer/apps/ninfer-serve.exe`；`ninfer-serve-concurrency8.exe` 已同步为同一程序。按已有打包脚本的依赖闭包补齐 CUDA 13.2 的 cuBLAS/cuBLASLt，服务在仅含 Windows 系统目录的 PATH 下完成上述 HTTP 请求。可直接运行结果目录的 `start-multigpu.ps1`，使用本机 UUID、40/24、CPU Vision、C2、B3072/R128/H1GiB；模型路径已指向本地制品。`-DryRun` 可查看参数。该脚本是本地构建入口，旧 RC1 ZIP 未重新发布。

完整当前工作树已导出回 `ninfer-kvmem.patch` 与 `versions.json`，保留同目录 fast/lookup 改动，未改变真实暂存区；`prepare-backends.py --check --backend ninfer` 通过。启动器、服务构建、Python 语法及两个仓库的 `diff --check` 通过。汇总、具体命令、日志、模型和程序记录保存在结果目录。

不宣称性能提升，不将本地源码能力算作现有 RC1 ZIP 的能力。未新增多卡 strict/mixed、overlay Vision、DFlash/DFlash2、视频、并行 prefill 或冷快照支持；异构 compute capability 不在原生 DeviceContext 合同内。
