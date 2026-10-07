# KVMem 两后端实现差异

更新日期：2026 年 10 月 7 日。适用于 `feat/multi-backend-framework` 当前工作区的 KVMem 集成。主仓库提交 `98b5910` 之后的 ninfer 改动在 `backends/patches/ninfer-kvmem.patch`，准备后的树哈希在 `backends/versions.json`。

llama.cpp 与 ninfer 共用 KVMem 策略和内存契约，但保留各自的推理、页池、请求处理和参数解析。切换后端时，需要同时核对参数名称、默认值、计量范围和执行行为。下表比较的是本项目的 `llama-kvmem-server` 与 ninfer KVMem worker，不是上游两个引擎的全部能力。

除“统一启动入口”一节外，默认值均指直接启动 worker 且未显式覆盖的配置。旧预编译包的功能以随包说明和验收报告为准；当前源码的参数不能自动用于旧包。运行方法见 [多后端运行](multi-backend.md)，源码准备方法见 [仓库工作流](multi-backend-repository.md)。

## ninfer 保留的粒度约定

用户于 2026 年 10 月 6 日决定：ninfer 当前的 64-token block 可以保留，本次不修改其大小，将它记录为后端实现差异。

| 概念 | llama.cpp KVMem | ninfer KVMem |
|---|---|---|
| 检索块 block | `--kvmem-block-tokens N` 可配置，默认 128 tokens | 当前固定 64 tokens，未开放对应启动参数 |
| 原生物理页 page | 本项目的 KV 搬运按适配器 block 和原生布局处理，不要求与 ninfer 页大小相同 | 原生页池要求每页 64 tokens |
| block 与 page 的关系 | 依据本后端的 block 与 KV 布局执行 | 当前一块对应一页；统计、选择、搬运和冷快照都按此关系实现 |
| recent 保留量 | `--kvmem-recent-tokens N`，转换为 `N / block_tokens` 个完整块，默认 0 | `--kvmem-recent-tokens N`，按 64-token 页向下取整；N 必须至少 64。省略时为 64，即最新一页 |
| sink 保留量 | `--kvmem-sink-tokens N`；默认 0 表示一块，正数向下取完整块且至少一块 | `--kvmem-sink-tokens N`，规则相同。默认 0 表示首个 64-token 页；正数向下取整且至少一页 |

block 是检索统计和选中历史的单位，page 是原生 KV 分配和搬运的单位，两者概念不同。保留 ninfer 的 block=64。`--kvmem-recent-tokens 1024` 在这个粒度下保留最新 16 页；不足一页的余数被舍去。ninfer 必须保留至少一页 recent，以维护尚未写满的追加页；0–63 会被启动脚本、服务参数和 Engine 校验拒绝。

代码位置：[llama 参数](../tools/llama-kvmem-server.cpp)、[llama block 与保留量映射](../src/adapter/llama-memory-kvmem.cpp)、[ninfer 原生页约束](../backends/ninfer/src/core/paged_kv_cache.cpp)、[ninfer 检索配置](../backends/ninfer/src/models/qwen3_5/program/storage/memory_statistics.cpp)、[ninfer 快照恢复配置](../backends/ninfer/src/models/qwen3_5/program/storage/memory_snapshot.cpp)。

## 工作集和检索行为

| 项目 | llama.cpp KVMem | ninfer KVMem |
|---|---|---|
| 启用 KVMem | worker 默认开启，提供 `--kvmem` / `--no-kvmem` | `--kvmem-budget B` 中 B>0 开启；默认 B=0 关闭，关闭时不能残留 R/H 等辅助配置 |
| 历史选择预算 B | `--kvmem-budget`，依据 block、设备容量和适配器配置计算有效值 | 同名参数；B 至少 128，必须为 64 的整数倍 |
| 生成预留 R | `--kvmem-gen-reserve`，默认 256；服务还将 R 用作输出硬上限 | 同名参数；KVMem 下至少 64，必须为 64 的整数倍；作为可复用追加空间 |
| Main KV 工作空间 | 根据本后端容量规则配置 B+R 的槽池 | 每条活动请求按 B+R 配置 Main 页池；要求 B+R 不超过逻辑上下文；MTP 另有原生页池 |
| 单次输出超过 R | KVMem 开启且 R>0 时，服务把默认和请求输出预算截到 `min(context, R)` | 允许；输出受请求预算、逻辑上下文和结束条件约束 |
| 生成时追加空间不足 | 本服务当前用上述 R 上限限制单次输出；不能从参数名推断已支持 ninfer 的持续追加行为 | 下一步紧凑 KV 视图超过 B+R 时，归档、重选最多 B、回填并继续追加；MTP 会为下一验证批次提前检查空间 |
| 选择方法 | `--kvmem-method recency\|retrieval`，默认 retrieval | 没有同名选项；query 完整时按 mean-K 检索，query 尚未完整时保留必选页并按最近历史补足 |
| 检索 query | 由最后用户 span 等逻辑确定；`--kvmem-query-max-tokens` 默认从 span 尾部取最多 512 tokens，`--kvmem-query-last` 提供 64-token 回退 | 使用 Frontend 标记的最后真实用户内容 span；没有对应的可配置长度上限 |
| query 重放和策略 | 开放 `--kvmem-query-replay legacy\|auto`、`--kvmem-query-policy legacy\|user` | 使用 ninfer 的 query probe、checkpoint 和 replay 流程，没有上述策略开关 |
| 必须保留的历史 | sink、配置的 recent 和本请求其他 mandatory 区域占用 B | 配置的 sink、recent 和完整图片区域占用 B；其余页参与选择。省略 recent 时仍保留最新一页 |

recent 是 B 内的保留份额，R 是追加空间。增大 recent 不会自动增大 B+R；它会减少可供其他历史参与检索的份额。配置必须给 sink、recent、图片等必选区域留下足够的 B。

### ninfer 超过 R 后如何继续生成

以普通 decode、已选满 B=4096、R=512 为例：

1. 当前选中的历史 KV 和新追加 KV 逐渐占满 B+R。
2. 下一次写入会超过容量时，先将尚未归档或有更新的已提交 KV 保存到 Host，包括已生成内容。
3. 从旧输入与已生成历史中重新选择最多 B。保留配置的 sink 和 recent；省略时是首个页和最新一页。其余页按已有 query 的检索分数选择。
4. 按原始位置顺序回填选中页，形成紧凑 GPU 视图，再使用空出的容量追加。逻辑位置和累计输出计数继续增长，当前 GDN 状态不因 KV 换页而重置。

未选中的 KV 仍可保存在 Host，但暂时不参与当前注意力。长输出可以超过 R，不代表整段输出的 KV 始终在 GPU，也不代表每次换页都会完整保留最近 R 个 tokens。

RTX 5050 Laptop 已通过 [超过 R 的真实输出验证](reserve-output-5050-validation-20261006.md)：B=256、R=128、NVFP4 普通 decode 下，分别请求并生成 512/1024 tokens，结束原因均为 length；生成途中分别完成 2/6 次重选，KV/GDN 搬运校验通过，Main 页池保持 384-token 容量。这组功能结果不代表其他 KV 格式、MTP 或 Graph 已完成相同验证。

llama 的 R 输出限制是原服务路径已有的实现约束，尚未认定为缺陷。是否解除该限制及如何实现追加空间复用，需要单独确定行为目标和回归范围。

代码位置：[llama 输出上限](../tools/kvmem-webui.h)、[llama 服务与 query 处理](../tools/llama-kvmem-server.cpp)、[ninfer 选页和原生 KV 搬运](../backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp)、[ninfer decode 空间检查](../backends/ninfer/src/models/qwen3_5/program/decode.cpp)。

## 启动参数和默认值

| 配置 | llama.cpp worker | ninfer worker |
|---|---|---|
| 模型文件 | `-m` / `--model PATH`，使用 `.gguf` | 第一项位置参数为 `.ninfer` 路径 |
| 上下文 | `-c` / `--ctx-size`，默认 2048 | `--max-context`，默认 8192；两边 KVMem 预算均按每请求或每 lane 解释 |
| 默认输出预算 | `-n` / `--n-predict`，默认 -1，随后受服务 generation limit 限制；启动参数不接受 0 | `--default-max-tokens`，默认 8192；0 表示不另设输出上限，仍受逻辑上下文限制 |
| 预填充 | `-b` / `--batch-size` 逻辑 batch 默认 512；`-ub` / `--ubatch-size` 默认继承 batch | `--prefill-chunk` 默认 1024，必须为 128 的正整数倍；KVMem 下不超过 R |
| 主模型 KV 格式 | `--kv-dtype` 设置 K/V；当前 worker 接受 f16/f32/q8_0/q5_0/q4_0，默认 q8_0；`--cache-type-k/v` 可分别配置 | KVMem 开放 bf16/int8/fp8/nvfp4/k8v4/rk8v4/rk4v4/rk4v4-e8/rk2v4-e8；原生默认 bf16，统一脚本默认 int8；没有相同的独立 K/V 开关 |
| MTP KV 格式 | `--spec-kv-dtype` 独立配置，默认 f16 | KVMem Main/MTP 页池使用相同 KV 格式 |
| API 模型名 | `-a` / `--alias NAME` | `--model-id ID` |
| 聊天模板 | `--chat-template` 接收 Jinja 正文，`--chat-template-file` 接收文件；提供 `--chat-template-kwargs` | `--chat-template` 接收文件路径；同名参数的输入类型不同 |
| 主模型设备 | `--device CUDA0,CUDA1,...` 等原生设备名；支持相应的设备与切分配置 | `--device N` 选择一张 `CUDA_VISIBLE_DEVICES` 内的设备。多卡用 `--devices A,B,...`（2–8 张），与 `--device` 互斥 |
| 多卡切分 | 当前 KVMem 开放相应的 layer/tensor 路径，具体组合受模型、MTP 和 lane 配置约束 | 源码使用上游 layer pipeline，不是 TP 或 NCCL。启动器 `-Gpu` 的 UUID 顺序就是 stage 顺序；第一张卡承担 embedding、head、MTP 和 resident Vision，CPU Vision 仍走 Host 编码。可选 `--stage-layers` 或 `-StageLayers`，省略时按各卡空闲内存规划。各卡 compute capability 必须相同，显存可以不同。一份逻辑历史、一份选页结果和一份 B/R/H；各卡只持有自己层的 KV。`mixed`/`strict` 显存策略仍要求单卡。overlay Vision、DFlash、并行 prefill 和冷快照不因多卡而开放。验收见 [多卡验收](ninfer-multigpu-validation-20261006.md) |
| 长上下文位置缩放 | 当前自定义 worker 没有开放 RoPE/YaRN 启动覆盖项，沿用原生配置和模型元数据 | 开放 `--rope-yarn`、`--rope-yarn-factor`、`--rope-scaling-factor` 等原生选项；增大 context 与启用位置缩放是不同设置 |

KV 格式名称代表不同字节布局和量化方法。例如 ninfer int8 与 llama q8_0 都包含 8-bit 数据，但不能视为相同格式或直接互换 KV。跨后端 KV 迁移不支持。

代码位置：[llama worker 参数](../tools/llama-kvmem-server.cpp)、[llama 服务配置](../tools/kvmem-server-options.h)、[ninfer worker 参数与准入](../backends/ninfer/src/serve/serve_options.cpp)、[ninfer 默认配置](../backends/ninfer/src/serve/serve_options.h)。

## 思考和采样

| 配置 | llama.cpp worker | ninfer worker |
|---|---|---|
| 默认思考开关 | 普通聊天默认关闭；`--enable-thinking` 开启，`--no-think` 关闭，请求可覆盖 | 普通聊天默认开启；`--no-thinking` 关闭，请求可覆盖；未显式请求思考的结构化输出有关闭思考的特殊规则 |
| 默认思考预算 | `--reasoning-budget`：默认 -1 不限，0 立即结束，正数限制思考 token | `--default-thinking-budget`：默认未设置；显式值必须为正数，没有相同的 -1/0 启动语义 |
| 思考预算提示语 | `--reasoning-budget-message` | `--thinking-budget-message`；预算计数和结束控制沿用原生实现，不能仅靠别名承诺相同行为 |
| 默认 reasoning effort | `--reasoning-effort` 传入模板默认参数 | `--default-reasoning-effort` 验证 none/minimal/low/medium/high/xhigh/max，并映射到模型支持的强度 |
| 常用采样项 | temperature、top-p、top-k、min-p、presence/frequency penalty、seed | 对应常用项均已开放；temperature 不提供 llama 的 `--temp` 别名 |
| top-k 范围 | 非负整数，0 关闭 | 当前范围 0..20，0 关闭；不能直接复用 top-k=40 等配置 |
| repetition penalty | `--repeat-penalty` / `--repetition-penalty` 已开放 | 当前服务没有对应启动项 |
| seed 范围 | uint32 | uint64；相同 seed 不保证跨引擎输出一致 |
| 思考后采样 | 当前 worker 没有 ninfer 的对应启动项 | `--post-thinking` 和相关参数可切换思考结束后的采样配置 |

同一 Qwen dense 模型的基础 thinking/non-thinking 温度和 top-p 预设接近，但思考开关不同会选择不同预设。ninfer 还按模型架构选择默认值，不能将某个 dense 模型的比较推广到全部模型。

代码位置：[llama 采样和思考预算](../tools/kvmem-chat-sampling.h)、[llama 默认思考开关](../tools/llama-kvmem-server.cpp)、[ninfer 思考解析](../backends/ninfer/src/serve/translate.cpp)、[ninfer 模型采样预设](../backends/ninfer/src/models/qwen3_5/frontend/frontend.cpp)。

## 推测解码和请求并发

| 配置 | llama.cpp worker | ninfer KVMem worker |
|---|---|---|
| MTP 启用 | `--spec-type draft-mtp`；默认 none | `--spec mtp`；默认关闭；原生 DFlash 选项在 KVMem 组合中被拒绝 |
| 草稿长度 | `--spec-draft-n-max`，默认 3；当前 KVMem replay 模式要求 1..5 | `--draft-tokens`，KVMem 开放固定或自适应 MTP1..15 |
| MTP 状态策略 | `--kvmem-mtp-state snapshots\|auto\|replay`；默认 replay；tensor 路径只开放 snapshots | 使用 ninfer 的原生状态与事务流程，没有对应策略选项 |
| 自适应 MTP | 当前 worker 未开放对应参数 | `--adaptive-mtp`，必须启用 MTP，并遵守各组合限制 |
| ngram 草稿 | 当前 worker 未开放 ninfer 的对应参数 | `--ngram-draft-tokens` / `--ngram-min-match`；KVMem 要求 MTP；1..8路可用，启动并发大于1时，配置的16..63自动降至15并提示；单路保留宽度，文本及 resident/CPU 图片均放行九种原生 KV |
| MTP 隐含的 ngram 默认值 | 启用本 worker 的 MTP 不会接通 ninfer ngram | 直接启动时，启用 spec 且未指定 ngram 宽度会默认 15；统一脚本的 `NgramDrafts` 默认 0 并显式传入 |
| 活动请求数 | `-np` / `--parallel`，默认 1，开放 1..4；NVMe/冷会话磁盘与部分设备组合受限制 | `--max-concurrency`，默认 1；KVMem 开放文本及 resident/CPU 图像 1..8，多路 ngram 宽度上限15；磁盘冷快照当前暂时禁用 |

固定草稿长度、自适应上限和 ngram 草稿长度是不同配置。比较两个后端的 MTP 时，应显式列出这些设置，不能仅用“MTP 已开启”代表同一执行方式。

代码位置：[llama 推测和 lane 准入](../tools/llama-kvmem-server.cpp)、[ninfer 组合准入](../backends/ninfer/src/serve/serve_options.cpp)、[ninfer ngram 默认规则](../backends/ninfer/src/product/speculative_options.h)。完整 ninfer 组合边界见 [多后端运行](multi-backend.md)。

## Host 会话和磁盘存储

| 配置 | llama.cpp KVMem | ninfer KVMem |
|---|---|---|
| Host 历史数量 | `--kvmem-conversations N`，默认 1；计活动和非活动 Host stores，并发时至少为 lane 数 | `--kvmem-sessions N`，默认 4，范围 1..16；计保留的非活动 Host 历史，活动请求另占原生资源 |
| Host 历史预算 | `--kvmem-conversations-gb` / `--kvmem-session-ram-gb`，单位 GiB；会话 RAM 软上限，默认 0 不限，活动会话可能超过软上限 | `--kvmem-host-mib`，单位 MiB；Main/MTP 原生 KV payload 的全局硬预算，KVMem 下必须非零；不含 mean 索引、GDN checkpoint、模型及其他缓冲 |
| 在线 KV 存储层 | 原有 `--kvmem-cpu-gb`、`--kvmem-nvme-gb`、raw-K/V 等配置；含义与冷会话配额分开 | 当前集成采用 Host KV 归档和有界 pinned 搬运缓冲；冷磁盘不是在线 NVMe 换页层 |
| 非活动会话冷缓存 | `--kvmem-session-cache-dir` + `--kvmem-session-nvme-gb`，单位 GiB；要求多会话，不能与在线 CPU/NVMe arena 配置混用 | 当前按用户要求暂时禁用；启动器、serve CLI、Engine 均拒绝开启，内存历史复用仍可用 |
| 冷快照身份 | llama 的会话存储与 checkpoint 规则 | ninfer 的模型、原生布局、执行配置、精确前缀及完整状态点；不能跨后端读取 |

会话数量、Host KV payload 预算和进程总 RAM 是三个不同指标。统一界面可以统一单位和展示方式，但应保留各预算的覆盖范围、硬软限制及活动会话计数规则。

代码和说明：[llama 会话 RAM](multi-conversation-kv-cache.md)、[llama 冷会话缓存](session-disk-cache.md)、[ninfer 内存配置](../backends/ninfer/include/ninfer/types.h)、[ninfer Host 历史保留](../backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp)、[ninfer 冷快照](multi-backend.md)。

## 视觉和服务配置

| 配置 | llama.cpp worker | ninfer KVMem worker |
|---|---|---|
| 视觉资源 | 单独的 `--mmproj PATH`；提供视觉设备、CPU/GPU offload、图片 token 和视频参数 | 视觉权重与预处理资源在 `.ninfer` 内；`--vision` 启用，开放 resident/CPU；KVMem 当前拒绝视频和 overlay |
| CPU 视觉 | `--no-mmproj-offload`，或相应视觉设备配置 | `--vision-residency cpu` / `--vision-cpu`；允许最多 8 条活动图文请求，CPU 编码本身串行；启动脚本每图默认 1024 tokens |
| API key | `--api-key` 支持多个 key，另有 `--api-key-file` | `--api-key` 配置一个 key，没有对应的 key 文件入口 |
| 超时 | `--timeout`，默认 1800 秒，设置 HTTP 读写超时 | `--pending-timeout-ms`，默认 600000 毫秒，限制准备和请求准入等待；含义不同 |
| UI 开关 | `--no-ui`，同时接受 `--no-webui` 别名 | `--no-webui` |
| 日志级别 | `--verbosity` / `--log-verbosity`，数值 0..5；另有 KVMem trace 开关 | `--log-level` 使用 trace/debug/info/warning/error/critical/off；另有请求 JSONL 与统计配置 |

代码位置：[llama 服务参数](../tools/kvmem-server-options.h)、[llama 服务配置](../tools/llama-kvmem-server.cpp)、[ninfer 服务参数](../backends/ninfer/src/serve/serve_options.cpp)。

## 统一启动入口

`scripts/windows/start-backend.ps1` 已统一后端选择和部分参数映射，但不消除上述执行差异。

| 入口配置 | 当前映射和限制 |
|---|---|
| 两边共用 | Backend、Model、Worker、Gpu、ListenHost、Port、Context、Budget、Reserve、Prefill、MaxTokens、KvType，以及 WorkerArgs。ninfer 的 `-Gpu` 接受 1–8 个不重复 UUID，顺序即 layer stage 顺序 |
| 共用脚本默认值 | 端口 18200、Context=262144、B=4096、R=1024、Prefill=128、MaxTokens=1024、Concurrency=1；KV 默认 llama q8_0 / ninfer int8；ninfer 视觉 token 默认 1024。相同 MaxTokens 仍受各后端的输出限制 |
| 只为 ninfer 映射 | HostMiB、StageLayers、MtpDrafts、AdaptiveMtp、FastPrefill、NgramDrafts/MinMatch、Concurrency、RetainedSessions、Vision/Residency/Tokens、DeviceProfile、ThinkingBudget、SinkTokens、RecentTokens；显式传给 llama 会被脚本拒绝。省略 ThinkingBudget 不限制思考。SinkTokens 默认 0（一页），RecentTokens 默认 64（一页）。DiskPath/MiB 当前暂时禁用。设备参数只能走 `-Gpu` 和 `-StageLayers`，思考预算只能走 `-ThinkingBudget`，sink/recent 只能走 `-SinkTokens` 和 `-RecentTokens`，不能再从 WorkerArgs 传 `--device`、`--devices`、`--stage-layers`、`--default-thinking-budget`、`--kvmem-sink-tokens` 或 `--kvmem-recent-tokens` |
| llama 的对应能力 | MTP、lane、会话、视觉和冷磁盘等已有能力仍可用，当前需要通过 WorkerArgs 传原生参数；脚本拒绝对应入口选项不表示后端没有该能力 |
| 默认 ninfer 快捷脚本 | `start-ninfer.ps1` 调用统一启动器。Context=204800、B=36864、R=16384、Host=12288MiB、Prefill=256、输出上限16384、思考预算8192、sink 0（一页）、recent 64（一页）、单路、INT8、自适应MTP上限4、ngram31、CPU视觉、每图1024 tokens、保留1个非活动 Host 会话。可用 `-Gpu`、`-StageLayers`、`-SinkTokens` 和 `-RecentTokens`。这些值与共用脚本默认值不同 |

代码位置：[共用启动脚本](../scripts/windows/start-backend.ps1)、[ninfer 快捷脚本](../scripts/windows/start-ninfer.ps1)。

## 参数统一的边界

| 事项 | 当前约定或后续方向 |
|---|---|
| ninfer block 与物理页 | 已决定保留当前 64 tokens，作为实现差异展示；不为参数外观一致而修改原生页池 |
| recent、sink | ninfer 已按整页向下取整规则开放 `--kvmem-sink-tokens` 和 `--kvmem-recent-tokens`。recent 至少 64 tokens，省略时仍是一页，与 llama 允许 0 的规则不同。query 长度仍未开放 |
| 思考、输出预算、MTP、并发 | 优先统一用户入口和默认值说明；R 输出限制等行为变化需要另行确定目标，不能只改别名 |
| 会话和内存预算 | 统一单位与展示，并明确预算范围、活动/非活动计数、硬预算与软上限 |
| 原生 KV 格式和多卡实现 | ninfer 源码已开放 layer pipeline；llama 保持自己的 layer/tensor 路径。两者不等价，也不迁移 KV。按包、模型和功能组合校验 |

维护参数时，应同步更新 worker 解析、统一入口、能力说明和本表。参数名称相同但语义不同的项目，应先说明行为，再决定是否统一。
