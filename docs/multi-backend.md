# KVMem 多后端运行

Windows 入口 `scripts/windows/start-backend.ps1` 在启动时选择 `llamacpp` 或 `ninfer`。两个 worker 分进程运行，各自拥有推理引擎和 GPU 内存；公共 KVMem 策略库直接链接在 worker 内。

## 启动

解压包后，在 PowerShell 中运行：

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model.ninfer' -Gpu 'GPU-your-device-uuid' `
  -Context 262144 -Budget 2048 -Reserve 512 -Prefill 512 -HostMiB 12288
```

`-Worker` 可指定本地构建的 `ninfer-serve.exe` 或 `llama-kvmem-server.exe`；省略时使用包内 `workers/<backend>/` 的程序。`-DryRun` 输出启动参数，`-Describe` 查询静态能力。默认监听 `127.0.0.1:18200`，前台运行，Ctrl+C 结束本次启动的 worker。安装 NVIDIA 驱动及 Microsoft Visual C++ 2015–2022 x64 Runtime；包内保留各引擎的 DLL，模型另行提供。

llama.cpp 示例：

```powershell
./scripts/windows/start-backend.ps1 -Backend llamacpp `
  -Model 'D:/models/model.gguf' -Gpu 'GPU-your-device-uuid' `
  -Budget 4096 -Reserve 1024 -KvType q8_0
```

已有的 llama.cpp 可执行文件、`start-server.ps1`、IQ3/IQ4 启动脚本仍可使用。原生参数通过 `-WorkerArgs @('--flag', 'value')` 传入，例如 llama 的 MTP、mmproj 和会话配置。后端参数保留各自含义；`.gguf` 与 `.ninfer` 不互换，也不迁移跨引擎 KV。`-HostMiB` 只对应 ninfer 的原生 KV payload 预算。

## ninfer 功能与组合

当前源码正在完成 P6–P10 及 ngram 验收，冻结的 P5 运行包仍只有单请求文本能力；新源码的参数不适用于旧包。新组合包括固定 MTP1–4（MTP2/4 的完整组合矩阵尚未完成，MTP4 已通过部分真实模型回归）、多个 Host 会话、最多四条文本请求、单活动请求图像，以及单活动文本请求的 RK8V4 和冷快照。最终可用组合以随包验收报告为准。原生非 KVMem 模式保持原来的参数入口。

```powershell
# 固定 MTP3，两个活动文本请求；B/R 是每请求预算，H 是全局预算。
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model-mtp.ninfer' -Gpu 'GPU-your-device-uuid' `
  -MtpDrafts 3 -Concurrency 2 -Budget 2048 -Reserve 512 -Prefill 512

# 图像需要包含视觉权重与预处理资源的制品；默认最多 256 个视觉 tokens/图。
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model-vision-mtp.ninfer' -Gpu 'GPU-your-device-uuid' `
  -Vision -VisionTokens 256 -MtpDrafts 3 -Budget 2048 -Reserve 512 -Prefill 512

# 单活动文本请求：RK8V4 与跨进程冷会话恢复。
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model-mtp.ninfer' -Gpu 'GPU-your-device-uuid' `
  -MtpDrafts 3 -KvType rk8v4 -DiskPath 'D:/cache/kvmem' -DiskMiB 4096
```

`-MtpDrafts` 默认 0，可选 1–4（固定草稿数；2/4 尚未完成完整组合矩阵）；`-NgramDrafts` 默认 0，可选 1–63，非零时要求 MTP 和单活动文本请求；`-NgramMinMatch` 默认 12，可选 4–64。例如 `-MtpDrafts 3 -NgramDrafts 31 -KvType rk8v4`。ngram 使用完整已提交 token 历史提出复制草稿，仍由目标模型验证；重复文本或代码更容易获益，不能承诺所有输入都加速。自适应 MTP、超过 4 的 MTP 草稿宽度、MTP 独立 attention window、视频、并发图像和 RK8V4 并发尚不在本轮组合内，服务会拒绝。`-DeviceProfile auto` 可选原生设备配置；磁盘快照首版要求 `off`，以固定跨进程执行配置。`-RetainedSessions` 控制非活动 Host 历史数量，默认 4，上限 16。

本次 ninfer 实测制品是 Qwen3.8-27B-GSQ-RCO-IQ3_S，设备为 RTX 5060 Ti 16GB、CUDA 13.2、`sm_120a`。8K/64K/128K/256K 档的早期口令检索和后续追问通过；这组人工构造用例不代表通用长文问答质量，也不表示其他 GPU 架构已验收。示例的 B=2048、R=512、H=12 GiB 是实际通过的配置。

- B 是每请求选中历史的 GPU token 预算，R 是追加工作空间，两者按 64-token 页对齐；C 条活动请求的 Main 页池按 C×(B+R) 配置。MTP 另有原生草稿页池，Main/MTP 的 Host payload 都计入 H。
- prefill 是 128 的倍数且不超过 R。追加压力触发选页，逻辑历史保留原始位置。生成上限由请求输出预算和逻辑上下文上限共同决定。
- H 只限制原生 KV payload。统计索引、最多三个检查点的 GDN StateImage、模型和临时缓冲另计，H 不等于进程总内存上限。
- 长期历史保存在普通 Host 内存，CUDA 搬运复用一个不超过 B+R 原生页大小的 pinned 缓冲区；其用量单独通过 `transfer_staging_bytes` 报告。
- 请求在执行前检查整段合法历史的 H 需求，已知预算不足返回错误。编辑历史不匹配可信前缀时从头计算。
- 如果单请求可容纳，但并发请求的全局 H 预留暂时不足，后续请求进入原生 FIFO 队列；不会让每个请求重复独占 H。服务入口默认排队时限 600 秒，Engine API 默认 30 秒，可显式配置。
- 图像区域完整保留，连同 sink 和追加页必须能装入 B；不满足时在准入阶段拒绝，可增大 B 或降低 `VisionTokens`。换图按内容、位置和精确前缀使旧 checkpoint 失效，不依赖文件名。
- 长输入的同查询工具续写可恢复 query 之后的完整端点，并保留冻结 Q/选页；新查询恢复合法的查询前检查点，重新执行 probe 与 replay。取消只保留已有完整 Base，不发布部分生成端点。
- 未超过 B+R 的短输入保存输入末尾前一 token 的 Base 和正常生成结束时的执行端点。短转长时，越过新 query 探测起点且不具有同查询附件的旧检查点不被复用。
- 复用与全量计算可能经过不同的原生分块边界，不能承诺两者浮点状态或所有 greedy tokens 完全相同。相同 checkpoint 的重复请求、取消恢复，以及明确答案的检索测试分别验收。

## API 与状态

`POST /v1/chat/completions` 使用原生 ninfer 的 tokenizer、chat template、采样器和协议实现。`stream: true` 返回 SSE，断开请求会触发取消；下次提交完整 messages 可恢复可信历史。`GET /health` 查询就绪状态，`GET /v1/models` 查询模型，`GET /props` 提供 `backend`、`kvmem` 预算和能力；`/stats`、`/v1/load` 提供原生运行统计及 KVMem 当前历史的 payload、mean 统计、checkpoint 图像用量。历史换入/换出计数随历史替换重置。

ninfer 直接启动参数为 `--kvmem-budget B --kvmem-gen-reserve R --kvmem-host-mib H`，其中 H 的单位为 MiB。这会关闭原生前缀缓存目录，由 KVMem 管理当前会话历史；`--no-prefix-reuse` 同时关闭 KVMem 的跨请求复用。`--kvmem-verify-transfers` 逐字节检查恢复内容，用于功能诊断，会增加传输开销。

ninfer 现在保留正常生成结束时的已执行端点，以及 Frontend 确认的模板重建边界。未改写的 assistant/工具回放可复用生成内容；最后一个尚未执行的输出 token 留给下一轮计算。每个会话共用一份 Host KV 历史，最多三个完整状态点。长历史工具续写在查询 span 和精确输入身份一致时保留原 Q/选页，只计算新增尾部；新用户查询重新执行检索，改写后的历史仅恢复仍精确匹配的完整状态点。恢复更早状态时放弃较晚状态点，防止共享历史被覆写后仍错误命中。此补全属于 ninfer 适配，llamacpp 路径保持原有行为。

冷快照使用 `--kvmem-disk-path PATH --kvmem-disk-mib N`。启动器设置 `-DiskPath` 时会开启 `--derive-session-keys`，从 system 和首条 user 消息派生会话键；Responses API 也可显式传 `prompt_cache_key`。Chat Completions 当前接受该字段但不映射为原生会话键，应使用派生键。它只保存具有会话键的非活动文本历史，按模型制品身份、KV/状态布局、执行配置与精确输入前缀恢复。只在完整校验通过后分配新的设备页与版本；不会恢复旧指针或 GPU 句柄。进程重启、Host 淘汰后再次提交同会话的完整 messages 可触发读取。损坏、截断、版本不匹配或写入失败会计入 `disk_errors`，请求可重新计算。

磁盘目录是单 worker 所有的冷缓存；实际文件放在专用子目录 `ninfer-kvmem-v4`，保存共享原生 KV、各完整状态点、mean 统计与冻结 Q/选页。旧 v3 记录不作为 v4 来源，重新计算后可生成新记录。磁盘配额包含这些记录和原子替换的临时文件，替换期间新旧记录都收费。配额不足时淘汰最旧非活动文件；仍无法容纳则跳过保存。它不是在线 NVMe 分层，也不使用 DirectStorage。状态提供 `disk_hits`、`disk_writes`、`disk_errors`；`history_hits`、`history_misses` 与 `history_evictions` 为进程累计计数。

已知 H 或逻辑上下文不足会在请求执行前返回错误，服务可以继续使用。实际 CUDA 驱动/设备错误仍沿用原生 ninfer 的失败处理，可能需要重启 worker；没有承诺在设备故障后继续复用同一会话。

## 源码构建和打包

引擎源码通过 `backends/llamacpp` 和 `backends/ninfer` 子模块引用既有项目，固定提交及 KVMem 补丁由主仓库管理。先运行 `python scripts/prepare-backends.py`；Windows 构建入口会自动准备所选后端。克隆、补丁维护和源码包说明见 [仓库工作流](multi-backend-repository.md)。

公共核心可用 `KVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF` 单独构建，llama 目标继续使用现有顶层 CMake。ninfer 在自己的构建目录中增加 `-DNINFER_KVMEM_SOURCE_DIR=<kvmem checkout>/kvmem`，构建 `ninfer-serve`；两边独立配置 CUDA 与编译器，公共核心维持 C++17，ninfer 使用 C++20。

在 VS 2022 Developer PowerShell 中运行 `scripts/windows/package-backends.ps1`，传入 `-LlamaWorker`、`-NinferWorker`、`-NinferSource`、`-OutputDir` 和 `-CudaPath`，可加 `-VcpkgInstalled` 与 `-ValidationReport`。输出目录必须不存在。脚本为每个 worker 收集 DLL，检查干净 PATH 下的启动，并写入许可文件、二进制校验值和验证报告；模型不会复制。打包成功只证明依赖可加载，推理结果以配套验证报告为准。
