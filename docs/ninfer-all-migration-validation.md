# ninfer-all 迁移验收记录

日期：2026-10-07。状态：**M0–M6 迁移完成，核心测试通过**。

本代码分支收录已完成验收的迁移源码、固定上游版本和集成补丁。原混合工作区保留；迁移阶段没有覆盖原后端源码或替换运行服务。后续 Q2 制品验证和服务配置见下文。实施过程和恢复方法见 [迁移计划](ninfer-all-migration-plan.md)。构建产物、模型、个人 DSH 配置和原始会话日志不随源码提交。

## 固定版本与可复现性

| 项目 | 最终值 |
|---|---|
| ninfer 上游 | `https://github.com/iamwavecut/ninfer-all.git` |
| ninfer 基线 / gitlink | `8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3` |
| 集成补丁 | `backends/patches/ninfer-kvmem.patch`，761034 字节 |
| 补丁 SHA-256 | `667b7ae9e7357dbb9cbb2d0c352cdc8fdd2a9bf6583248faf99ed0da0afc631f` |
| 准备后完整树 | `54485065cbdf4397227f6b524c08ea50c7b9ee02` |
| 父仓库基线 | `9b92d869ba4ff02fa1d7830687b5e05d93791bbe` |
| llama.cpp 基线 | `7fe450e19305b828c199d602c23a8337aaa1f03b`，本轮未修改 |
| 最终 worker SHA-256 | `4cc165bfd41ed66ae0863621f4070c268d66e8bf8df72a30df065248364867e2` |

`backends/versions.json`、`.gitmodules`、父仓库工作树的 gitlink 和完整补丁已一致固定。独立检出 `m6-clean` 从上述 ninfer 原始提交准备，完整树校验和重复准备通过；有本地源码改动时，准备保留改动，严格检查明确拒绝未记录的源码差异；把 manifest 改回旧 Ryan pin 的负向检查也明确拒绝。最终候选工作树自身的 `--check` 通过。

Windows 构建 wrapper 默认改为 8 并行，定向配置检查确认其指向候选公共 KVMem core。CI 新增 Linux/Windows 固定源码准备与重复校验，接入原有聚合门槛；**没有推送或运行远端 CI**。本机环境为 VS2022/MSVC 14.44、CUDA 13.2.86、Release `120a`/SM120 native；本轮编译均使用 8 并行。

## 迁移内容与支持边界

111 个文件的原 KVMem 补丁已三方适配，实际处理 26 个冲突文件。保留公共 KVMem、Main/MTP 原生 KV 字节、64-token 页、B/R/H、query probe/replay、KV/GDN checkpoint、Host 会话和图像身份。权重加载保留上游 tail/VMM 通路，多卡复制和 Vision 绑定按实际设备/rank 归属适配。上游新的 logprobs、ngram 输入索引、prefill layer-ready 回调和状态发布顺序也保留。

Ryan 原有 strict/mixed 内存策略已在 M1 保留和适配，default/strict/mixed 的实际短生成均通过；五个新增策略文件来自固定 Ryan blob。完整来源清单见 `m1-results/report.json`，不混入 Ryan 后来提交。

新上游 Q2_0/Q4_0/Q5_0 的生产内核、格式和物化逻辑直接采用；旧工作区 21 个量化增量逐文件处理，保留有独立价值的 golden/选行/reader 测试，不搬回未完成的旧 Q2_0 内核。QType 数字编号采用上游，v3 制品继续以格式名称持久化；旧制品已验证可读。

`qwen4_exp`、suspend、graft 和 router 尚未适配 KVMem，入口明确拒绝。DFlash、overlay Vision、并行 prefill、磁盘冷快照和跨后端 KV 转移仍遵循原支持边界。量化格式支持不等于新增模型架构支持。

## 核心运行验证

`core-results` 中 **25 个选定程序调用全部以 0 退出**。以下是验收范围；耗时仅为执行记录，不作性能比较。

| 范围 | 最小核心证据 | 结果文件前缀 |
|---|---|---|
| 原生 payload / 算子 | 九种 KV 格式的 Main/MTP 原始容器与字节；RoPE、INT8 sparse attention、mean-K/query | `payload`、`rope`、`sparse-int8`、`token-sums` |
| 参数、状态及预算 | frontend checkpoint；Engine/CLI/serve 参数；容量、缓存默认值、schema、加载报告、资源 Host/FIFO | `frontend`、`options`、`cli-options`、`serve-options`、`capacity`、`cache-defaults`、`schema`、`load-report`、`resources` |
| 实际越窗 | B128/R128，物理容量 256；输入 257 token，生成 130 token；重选两次，KV/GDN `payload_exact=1`、`state_exact=1` | `ordinary-window` |
| 普通原生对照 | 同一真实制品，原生 INT8/eager，无 KVMem；输入 257、生成 130 token | `native-control` |
| 会话与 Host 压力 | A→B→A 精确恢复；Host 压力淘汰 4 次；取消、编辑及关闭索引后的清理；结束时 resident pages 为零 | `sessions` |
| 工具历史 | thinking/tool 的 canonical input checkpoint、丢弃输出后追加、重复与取消恢复 | `tool-history` |
| Graph/MTP/ngram | 真实 Q4_0 MTP 制品，RK8V4/Graph、自适应 MTP4、ngram31；贪心/固定种子采样、取消、输出限制、编辑与代码复制 | `graph-mtp-ngram` |
| C8 | 同一制品，RK8V4/Graph、自适应 MTP4、ngram15；两个短批次，每批 8 路/48 输出上限；逐路精确重放、历史复用、实际并发解码、Main/MTP 页归零和 H 上界 | `c8` |
| 单卡图像 | 实际 IQ3_S Vision/MTP 制品；resident Vision 覆盖图像身份、重复、替换、取消/多图及预算拒绝后恢复；CPU Vision 运行 query-only 的重复、取消和多图恢复子集 | `vision-resident`、`vision-cpu` |
| 双卡文本 | 5060 Ti + 5050，层划分 40/24；RK8V4/Graph、自适应 MTP4，冷请求、重复、取消与恢复，状态字节和实际 draft 校验 | `pipeline-text` |
| 双卡 C2 图像 | 实际 Vision/MTP 制品，CPU Vision，RK8V4/Graph、自适应 MTP3；混合并发、重复、取消恢复、替换及保留；图像替换实际触发 `identity_rejected=2` | `pipeline-c2-images` |
| 三格式数值和旧制品 | Q2_0/Q4_0/Q5_0 + Q8_0 控制，K=1536/5120，T=1/8/9；独立 host byte decoder 和 FP64 oracle，linear/add/SwiGLU/embedding；旧三格式 writer 制品由新 reader/materializer 校验 | `gguf-numerics`、`old-writer` |

功能测试先查占用，以 UUID 选择空闲设备。数值检查使用 RTX 5050 Laptop `GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29`；较大图像/MTP 使用 RTX 5060 Ti `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`；双卡使用上述两张实际设备。没有性能实验。

## 转换、真实格式与交付入口

- **16 项 CPU 转换测试通过**（5.91 秒），覆盖三格式 golden bytes、选行、BF16 GDN 边界和 PQ2_0/PTQ1_0 ternary 编码。
- 增加固定 Bonsai 27B 的 `bonsai2_27b_ternary_gguf_mtp` recipe，支持独立 PTQ1_0/PQ2_0 文本来源和现有 GGUF MTP。复用并严格校验 dense MTP schema、形状和格式，保留原编码导入及 attention/MLP 分组。
- 实际 PTQ1_0 文本 + Q4_0 MTP 转为 v3 制品，790 个对象；9 个 MTP 矩阵的 36 个取样行与 GGUF 原始编码字节一致，含 interleaved 行映射。五个 norm 实际与 HF companion 比较，证实该 Bonsai r3 GGUF 存储值需减 1（F16 对照最大误差约 0.000488）；这一约定限制在该 recipe，未推广到任意 GGUF。
- 该真实 Q4_0 MTP 制品已用于上述 Graph/MTP/ngram、C8 和双卡文本测试。未通过修改制品标签来模拟格式。**Q2_0/Q5_0 没有本地真实模型样本，仅完成编码、制品互操作、物化与独立 GPU 算子验证。**
- 新 writer 的三格式制品到新 reader/materializer 通过。`qwen4_exp` 的 metadata-only 负向夹具在加载权重前被 KVMem 明确拒绝；它不是实际模型验证。
- 最终 worker 的 PE 依赖递归闭包为本地 10 个文件（含两个 cuBLAS DLL）。清理 PATH 后 `--help` 及实际 HTTP `/health`、`/props`、生成、`/stats` 通过；显式 KV 容量 256/B128/R128/max_context2048 生效，生成 `1, 2, 3, 4, 5`，Ctrl+Break 正常退出码 0。没有替换生产服务。

## 实际失败与修复

以下问题均保留证据并说明来源，没有把本次适配回归归因于旧 KVMem/llama.cpp。

| 问题及来源 | 触发、修复和证据 |
|---|---|
| M1 构建兼容 / 上游 | MSVC 的 C++20 `shared_ptr::unique()` 与 NVFP4 嵌套 lambda kernel 别名实际编译失败，分别改为等价 use_count 和直接启动同一 kernel。前者有状态测试；后者仅编译/链接验证，该快速 prompt 数值路径未独立运行。M1 完整记录见计划。 |
| Vision 接口迁移回归 | 新 bind 需要 hidden size；传入实际 text hidden size，worker 与两种 Vision 核心运行通过。 |
| HTTP 接口迁移回归 | 新多模型 lease 已取代旧 `service_`；props 改用选中模型的 `lease.service()` 和解析后 options，最终实际 HTTP 验证通过。 |
| 上游测试在 Windows 的命名冲突 | `near`/`far` 宏使资源/frontend 测试编译失败；仅改局部名称，原断言通过。 |
| 显式 KV 容量迁移回归 | 最终 worker 的 max_context2048/kv_capacity256 被新上游的普通原生长上下文检查提前拒绝。KVMem 改由原有 C×(B+R) 规则校验；新增 C1/C2 有效显式容量和不匹配拒绝的聚焦参数测试通过，最终 HTTP 运行通过。 |
| 独立准备副本的换行设置 | 首次手工克隆继承 CRLF，导致补丁/完整树检查失败；恢复固定 Git blob 的精确字节并关闭该副本的 autocrlf。最终 LF 补丁从干净基线准备与校验通过，不是源代码逻辑问题。 |
| M1 实际显存/预算不足 | 初始 Legacy 零 Host 配置及 Hybrid/Graph 预留被正确拒绝；选择有效的小保留槽配置完成原生短生成，没有改变预算规则。后续 KVMem Graph/MTP 和双卡测试已完成。 |

**没有确认新的旧 KVMem/llama.cpp 逻辑缺陷，因此没有需要用户决定保留或改变的旧行为。**

## 未扩大验证的范围

没有跑全套 CTest、18 格式全宽度矩阵、KV×MTP×Graph×Vision×并发×设备全排列、超长上下文档位扫描或性能比较。九种 KV 的容器/原始字节覆盖不等于九种真实模型全流程；运行覆盖以 INT8 和 RK8V4 的代表组合为准。未新测 llama.cpp/NVMe、DFlash、overlay Vision、并行 prefill、strict 自动 OOM 恢复矩阵或新 qwen4 架构；没有新增这些能力声明。远端 CI、独立 CLI 可执行文件和快速 NVFP4 prompt 数值路径也未运行。

## 本机证据

以下为操作者本机 `ninfer-all-migration-20261007/` 目录内的证据路径；原始结果本地留存，不是本代码分支中的文件。

| 路径 | 内容 |
|---|---|
| `m0-snapshot/capture.json` | 原父仓库和两个后端的恢复快照、HEAD、index 与工作区文件哈希 |
| `m1-results/report.json`、`m2-results/rebase/` | Ryan 来源、M1 检查、三方适配输入与冲突记录 |
| `core-results/*.json`、`*.log` | 25 个程序的命令、GPU UUID、运行前占用、退出码与原始日志 |
| `m5-results/conversion-final-tests.log` | 16 项 CPU 测试 |
| `m5-results/mtp-norm-source-check.json`、`real-q4-raw-row-check.json` | 真实来源 norm 与编码行证据 |
| `m6-results/frozen-candidate.json`、`preparation-checks.json` | 最终补丁/完整树与独立准备校验 |
| `m6-results/quant-increment-disposition.json` | 21 个量化增量文件的最终去留与上游字节一致性 |
| `m6-results/writer-and-family-guard.json` | 新制品互操作和 metadata-only 架构拒绝 |
| `m6-results/worker-dll-closure.json`、`final-worker-http.json` | 最终 binary/DLL 哈希、清理 PATH 后 HTTP 与正常退出 |
| `m6-results/original-inputs-audit.json`、`report.json` | 原源码/index 保留审计与最终验收汇总 |

## 后续真实 Q2 制品和本地连接验证

同一最终 worker 已在 RTX 5060 Ti 上运行 [ModelScope 的 Q2LynnStyle-MTP 制品](https://www.modelscope.cn/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer/tree/master/GGUF-NInfer)。下载源固定为仓库 revision `4fc03123dec322d855458de3831fc7b5788fb863`，文件为 `GGUF-NInfer/Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer`，大小 13272798720 字节，SHA-256 为 `cdd82b69027faba0bda619224ee851ccd0ecf0dfb176bad85a17d2df07926921`。

该模型的 Q2 名称表示混合精度：文本包含 IQ2_S、Q2_K、IQ3、IQ4 等格式，MTP 矩阵为 Q4_0；**不含 Q2_0**。这项结果补充真实低比特模型和 Q4_0 MTP 的运行证据，不能替代 Q2_0/Q5_0 的真实模型验收。

两个定向配置均采用 C1、max_context2048、B128/R128、物理 KV 容量 256、H1024 MiB、prefill128 和 FP16 GDN，冷请求与精确重复通过，KV/GDN 字节校验通过，进程正常退出：

- INT8 KV/eager、关闭 MTP/ngram：输入 765 tokens、输出 30 tokens；重复复用 758 tokens，输出一致。
- RK8V4/Graph、MTP3/ngram15：输入 33 tokens、代码输出 84 tokens；MTP 草稿 66 tokens、接受 64 tokens，重复输出一致。

后续按原 IQ3 服务参数启动 Q2，使用 INT8 KV、Graph、固定 MTP4、关闭 ngram、max_context204800、C1、B49152、H6144 MiB、最大输出32768。原 R32768 在单卡 INT8 配置下被启动容量检查拒绝：运行时需 3373348352 字节、可用 2650800128 字节；调整为 R8192 后正常启动，输出上限仍为 32768。按用户要求将 sink/recent 各设为 10240 tokens，实际 `/props` 校验通过。

DSH 默认连接指向该 Q2 模型。使用已安装的 DSH 适配器完成两次短流式请求，验证工具调用、模拟工具结果与 assistant replay。此检查覆盖模型接口和工具协议，**不覆盖真实 PowerShell 执行器**。随后独立诊断发现 DSH Desktop 0.2.0-rc.2 的 Windows 沙箱 PTY 启动失败；普通 PowerShell PTY 和带有效输入管道的同一只读沙箱命令可执行，沙箱 PTY 子进程以 127 退出。具体原生调用尚未确认，未改动 DSH 代码或文件保护规则，也没有把它归为 KVMem 推理回归。

后续原始证据在操作者本机 `q2-real-model-20261007/`：`test-results.json`、`service/live-check.json`、`dsh/adapter-verification.json` 和 `dsh/pty-diagnosis-summary.json`。没有运行性能测试或新增长上下文扫描。
