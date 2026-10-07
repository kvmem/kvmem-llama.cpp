# ninfer-all 上游迁移评估与实施计划

日期：2026 年 10 月 7 日。状态：M0–M6 已完成，核心迁移验收通过。最终范围及覆盖边界见 [验收记录](ninfer-all-migration-validation.md)。

## 建议与难度

建议保留 `feat/multi-backend-framework` 的公共 KVMem 核心、独立 worker 和双后端入口，将 ninfer 子模块的上游改为 `iamwavecut/ninfer-all`，在固定提交上重新适配 KVMem。先完成现有能力的迁移验收，再完成 Q2_0、Q4_0、Q5_0 的本项目验收。这里按此前约定，将本次消息中的“q2,q4,q4”理解为这三种权重量化格式。

整体难度为**中高，约 4/5**。主要工作是页池、请求执行状态和会话复用的适配，不需要重写公共 KVMem，也不是只更换仓库地址。现有补丁涉及 111 个文件，直接应用到候选上游时有 45 个文件失败。可直接应用的另外 66 个文件仍需要检查调用约定和行为。

一个有利变化是：**候选上游已经实现三种量化格式的转换注册、vector/MMQ 运算和 embedding 路径**，并提供独立 CPU 解码检查。因此原来的“继续自行接三种内核”应改为“采用上游实现，保留本地有价值的测试，补齐转换配方和 KVMem 场景验收”。这会减少后续自维护代码，但不能代替本机 Windows/SM120 验收。[量化实现](https://github.com/iamwavecut/ninfer-all/tree/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/src/ops/linear/gguf)、[上游数值测试](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/tests/ops/linear/test_gguf.cpp)

按一人集中开发粗估，迁移及精简验收约需 **5–9 个有效工作日**；如果遇到 Windows 新构建故障或 checkpoint 行为变化，另留 2–4 天。这是拆分工作后的工程估计，不是完成时间承诺；M1 原生构建和 M2 KV 适配完成后应重新估算。纯量化验收预计只占其中约半天到一天，主要不确定性在 KVMem 适配。

## 本次核对的版本和证据

| 项目 | 本次分析快照 |
|---|---|
| 父仓库分支 | `feat/multi-backend-framework` |
| 父仓库 HEAD | `9b92d869ba4ff02fa1d7830687b5e05d93791bbe`，另有工作区改动 |
| 当前 ninfer 上游 | `Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco` |
| 当前 ninfer 基线 | `6e9189c1089f6968e29fe2e1c51b4596e2346eed` |
| 当前集成补丁 SHA-256 | `b07edbfd0065c310ebc1655c0e8a9e404db75a084d2e1f8a4c02e79b16c7fcc8` |
| 清单中的准备后树 | `d467b3069a847dbeff76eadad28bf89a6103ee12` |
| ninfer-all 候选提交 | `8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3`，`master` 当时的 HEAD |
| 候选提交时间 | 2026-10-07 07:05:31，北京时间 |
| Ryan 远端当前 HEAD | `b06908ba3caa4f73269274fc7984b96f16d4295c`；与本地锁定基线不同 |

GitHub 提交接口在 `2026-10-01T00:00:00Z` 至查询时刻返回 ninfer-all 至少 100 条提交，Ryan 分支返回 4 条。ninfer-all 的第一页已达 100 条上限，最早一条仍在 10 月 3 日，因此不能把 100 当作总数。这个时间窗口支持“近期更新更快”的判断，不能由提交数推断稳定性或长期维护承诺。[ninfer-all 提交记录](https://github.com/iamwavecut/ninfer-all/commits/master/)、[Ryan 提交记录](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco/commits/main/)

静态文件比较结果如下，统计口径为 Git 跟踪的普通文件，不含构建产物：

| 比较项 | 结果 |
|---|---:|
| 当前基线文件 / 候选上游文件 | 2491 / 2605 |
| 候选上游相对基线新增 / 删除 / 修改 | 193 / 79 / 536 |
| 当前 KVMem 补丁文件 | 111 |
| 其中修改已有文件 / 新增文件 | 87 / 24 |
| 87 个已有文件中上游也已修改 | 78 |
| 逐文件补丁检查成功 / 失败 | 66 / 45 |
| 工作区相对准备后树的额外差异 | 21 个文件，均属于本轮量化接入及其测试、文档 |

失败数表示补丁上下文不能直接应用，不是已经确认的 45 个功能缺陷；成功数也不表示语义兼容。此次只在独立上游检出中运行 `git apply --check`，没有应用补丁。

本机研究证据保存在 `C:\Users\leyew\AppData\Local\KVMem\ninfer-all-research-20261007\`：`audit.json` 包含所有文件的检查结果，`audit.py` 可重现统计，两份 `*-recent-commits.json` 保存活动记录，`upstream/` 是固定提交的干净源码。2026-10-03 的旧研究仅作历史参考，不替代本次快照。

## 迁移范围与架构边界

迁移单位是 `backends/ninfer` 及其集成补丁。父仓库继续管理公共核心、llama.cpp 适配器、两个独立 worker、构建和打包入口；llama.cpp 的上游和基线不随本次迁移改变。仍采用“固定上游提交 + 本地补丁”的准备方式，沿用 `scripts/prepare-backends.py` 的版本和完整树校验。

ninfer-all 的 Host/disk prefix cache 不能替代 KVMem 的检索工作集。候选上游原生路径仍要求显式 KV 容量至少覆盖 `max_context`；KVMem 则需要保留更长的 Host 历史，按 query 选择有限的 GPU 历史。需要继续维护的合同是：

- ninfer 的 64-token block/page，以及 sink、至少一页 recent、图片区域等必选页。
- 每请求的 B 历史预算与 R 追加空间；C 个活动请求的 Main 页池按 C×(B+R) 配置，MTP 使用独立页池。
- Host 原始 KV payload 的共享 H 预算、有界 pinned 搬运缓冲，以及各自可解释的内存统计。
- mean-K/query 统计、原始位置、query probe/replay、生成过程中换页和超过 R 后继续生成。
- Main/MTP KV 与 GDN StateImage 的一致 checkpoint、提交、取消恢复和会话隔离。
- 已放行的九种 KV 格式、MTP/ngram、resident/CPU Vision、C1–8 和同 compute capability 的多卡 layer pipeline。

以上能力范围以当前[多后端运行](multi-backend.md)、[后端实现差异](multi-backend-differences.md)和对应验收记录为依据。历史结果作为旧基线证据保留，不能直接写成候选上游已经通过。原生容量约束见[上游 startup](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/src/models/qwen3_5/program/planning/startup.cpp)。

候选上游还新增独立的 `qwen4_exp` 模型族、模型 suspend/resume、router 和新的缓存行为。源码更新会带入这些能力，但本轮不承诺它们已支持 KVMem。尤其不能因为 Flash-Next 使用 Q2_0，就把“量化格式可计算”当作“该模型的稀疏 attention、状态和 KV 已接入 KVMem”。KVMem 开启时，应明确拒绝未适配的模型族及会绕过现有状态管理的新组合；原生非 KVMem 路径维持上游行为。[上游维护者说明](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/docs/maintainer/consolidated-line.md)

本轮也不恢复已暂停的 ninfer KVMem 磁盘冷快照，不新增 DFlash、overlay Vision、并行 prefill 或异构 GPU 支持，不要求下载 Flash-Next 大模型。llama.cpp 的 NVMe 会话交换与本次 ninfer 迁移没有直接关系，不把其压力测试加入默认验收。

### 适配代码的职责划分

为降低后续更新成本，将 KVMem 的具体处理尽量集中在已有模型侧适配文件中，在上游执行流程中保留必要的显式调用。不会为了这次换上游新增通用插件框架或重写所有 backend 接口。

| 层级 | 应保留的职责 | 迁移时的约束 |
|---|---|---|
| 父仓库 `kvmem/` | 检索策略、历史目录、Host 原始 payload 和预算合同 | 不依赖 ninfer 的 Tensor、CUDA Graph 或 StateImage 类型 |
| ninfer core / artifact | 原生页池、设备与 stream、权重加载和布局 | 提供原始数据搬运；不把 KVMem 的 query/会话策略塞入通用加载器 |
| `program/storage/kvmem_window.cpp` 与 `memory_statistics.cpp` | Host 历史和原生页映射、选页、回填、K/Q 统计 | 适配新原生接口，保持公共预算和原始位置含义 |
| Qwen3.5 Program / frontend | query span、probe/replay、KV/GDN checkpoint 和执行提交 | 明确 prefill、decode、MTP、取消各自调用点；保留模型专有状态的所有权 |
| runtime / serve | 准入、FIFO、资源退役、参数、能力与状态报告 | 对未接入模型/功能组合明确报错，避免走到没有 KVMem 处理的执行分支 |
| 父仓库 scripts / manifest | worker 选择、工具链、固定版本、补丁和交付入口 | 两个引擎继续独立构建和运行，准备结果可校验 |

## 差异处理策略

### 将当前代码分成三部分

| 来源 | 处理方式 |
|---|---|
| Ryan 基线中已有的 Windows/SM120 和内存策略改动 | 单独与新上游比较；它们不在 KVMem 集成补丁中，不能只移植补丁就算保留完整能力 |
| 已记录的 KVMem 集成补丁 | 按构建、KV、执行状态、服务入口分组适配；重点审查冲突文件和它们的调用方 |
| 本轮额外的 21 个量化文件 | 保留当前现场和证据；采用上游已有生产实现，逐项迁移有增量价值的测试，避免重复实现 |

量化步骤 2 的转换/加载已完成，已有 10 项定向 Python 测试和三格式加载验证。步骤 3 的 GPU 接线只写入了代码，**尚未编译或运行数值测试**。迁移前须按这个状态归档，不能将其纳入“已验证基线”。原始记录位于 `C:\Users\leyew\AppData\Local\KVMem\backend-p0p1-20261002\results\quant-wiring-20261007\`，包括 `step2-result.json`、`step2.diff` 和 `step3-before/`。

### Windows 差异需要重新判断

10 月 3 日曾用最小 MSVC 编译确认旧 ninfer-all 的 device profile 单个长字符串触发 C2026。当前候选提交已经改成按行切分、每段不超过 8 KiB 的字符串数组再拼接。**旧缺陷不能继续列为当前已知阻塞，也不应机械搬回 Ryan 的字节数组实现**；M1 构建时验证新生成代码即可。[当前 CMake 实现](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/src/runtime/CMakeLists.txt)

候选上游支持 `120a`，要求 CUDA 至少 13.1；本机 CUDA 13.2 满足版本条件，但这不是编译成功的证据。MSVC 参数、CUDA launch/shared-memory 处理、NVFP4/FP8 编译路径、D3D12/WDDM 与依赖 DLL 仍需核对。使用本机 VS2022/MSVC 14.44、CUDA 13.2 和现有依赖安装，不能照搬上游 RTX 3090 主机的环境说明。[上游构建定义](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/CMakeLists.txt)

当前 Ryan 的 `--cuda-memory-policy strict/mixed` 在候选上游没有同名实现。当前 KVMem 模式本来只允许 `default`，因此它们不是 KVMem 算法迁移的必要条件；但原生 worker 已有这些入口。计划默认保留已公开的原生合同，作为独立兼容改动处理，不把托盘管理器等整个 Ryan 产品一并搬入。如果保留成本超出预估，应在阶段汇报中说明取舍，再决定是否调整范围，不能静默删参数或改成近似行为。

### 权重格式名称与内部编号分开处理

| 格式 | 本地未完成分支的 QType | 候选上游 QType | GGML 类型 | 元素数 / 字节数 |
|---|---:|---:|---:|---:|
| Q2_0 | 25 | 27 | 42 | 64 / 18 |
| Q4_0 | 26 | 25 | 2 | 32 / 18 |
| Q5_0 | 27 | 26 | 6 | 32 / 22 |

采用上游编号，并重新编译所有依赖对象；不沿用旧静态库或对象文件。v3 `.ninfer` 的 tensor 描述使用 `gguf_q2_0`、`gguf_q4_0`、`gguf_q5_0` 这些字符串，不直接持久化上述 C++ 枚举。因此编号顺序差异本身不要求重写模型文件。M5 仍需拿步骤 2 的旧 writer 夹具交给新 reader，验证名称、布局、行选择和实际字节；也要核查本地其他记录是否曾保存裸枚举，不能外推所有私有缓存都兼容。[格式定义](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/src/core/weight.h)、[制品 schema](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/src/artifact/schema.cpp)

三者是权重量化格式，与 `KvType` 的 q4/q8 或 NVFP4 等 KV 存储配置不是同一个合同；也不能用 ninfer 原生 `Q4_G64` / `Q5_G64` 替代 GGUF Q4_0 / Q5_0。

## 实施阶段

执行约定：用户于 2026-10-07 授权连续推进至迁移和核心测试完成，仅重大问题或需决定的旧逻辑问题暂停确认；阶段完成时汇报进展，无需逐阶段等待。用户已同意按本计划迁移，实验仅覆盖实际改动和关键风险，不为完整性增加无关实验。执行状态记录在文末。

### M0 固定迁移输入并保留恢复点

目标：建立可复现的旧状态和迁移候选，不影响当前运行服务。

1. 开始实现时重新确认父仓库、两个子模块、补丁和工作区状态，防止本分析之后的新改动被遗漏。
2. 导出父仓库及 ninfer 的已暂存、未暂存差异，并复制 Git 未跟踪的新增源码；单独归档量化步骤 2 与未验证步骤 3。仅 `git diff` 不足以保存新增文件。
3. 在隔离的迁移工作区使用本计划固定的 `8319e8f…`。若决定换更晚提交，先更新比较报告，再使用新快照，实施期间不追随浮动 `master`。
4. 建立逐项迁移清单，标明“上游已覆盖、需要适配、保留为独立差异、范围外”，特别列出 Ryan 基线自带的改动。

交付：恢复清单、源文件快照和明确的迁移目标。验收只检查文件完整性及版本对应，不编译、不跑 GPU。不要自动提交当前混合工作区，也不要用 reset 清理它。

### M1 确认原生 ninfer-all 可以在本机运行

目标：先区分上游构建问题与 KVMem 适配问题。

1. 为候选源码建立独立构建目录，使用 CUDA 13.2、`120a`、明确的 SM120 native 设置，默认 8 个编译任务。
2. 只构建 `ninfer-serve` 及实际需要的依赖；不构建全部 benchmark/test 目标。处理真实出现的 Windows 编译和链接问题。
3. 对照 Ryan 的必要兼容项，优先采用上游等价实现；对仍缺失且属于现有公开合同的项做小范围移植，包括非 KVMem 的 strict/mixed 参数链。
4. 用已有、可放入空闲 GPU 的 `.ninfer` 模型，关闭 KVMem，做一次原生加载和短生成；检查 worker 路径、依赖 DLL 和启动参数。strict/mixed 如保留，另作该路径必要的参数与加载检查。

验收：原生 worker 能在本机工具链构建、启动和生成；兼容差异有来源记录。没有这一步，不把后续构建/加载失败归因于 KVMem。该阶段不测速度，也不启用自动校准作为默认验收的一部分。

### M2 接回公共 KVMem 与原生 KV 页搬运

目标：恢复有界 GPU 工作集及 Host 原始字节存储。

主要位置：ninfer 的 `CMakeLists.txt`、`src/models/CMakeLists.txt`、`src/core/paged_kv_cache.*`、`src/core/device.*`、`program/planning/startup.*`、`program/storage/kvmem_window.cpp` 和 `memory_statistics.cpp`。

1. 接回 `NINFER_KVMEM_SOURCE_DIR`、公共库链接和模型侧适配源码；保留上游新的加载/执行目标划分。
2. 对照新页池的 page/head-major 布局、plane/rank 信息、设备 current/stream 和生命周期接口，适配导入导出及主/MTP 页池。不要以编译通过代替搬运正确。
3. 保留 B/R/H、64-token 粒度、sink/recent、原始位置、mean-K/query 统计、有限 pinned staging 和跨 rank 的同步语义。
4. 先让单卡 ordinary 路径完成一次越过 B+R 的输入与生成换页，核对实际页池上限和 Host 计费。

精简验收：复用 `ninfer_p1_payload`、`ninfer_p1_sparse` / `ninfer_p1_rope`、`ninfer_token_sums` 中受影响的检查；九种 KV 格式用小页夹具覆盖原始 codes/scales 往返和边界，不为每种格式启动真实模型。只在公共 KVMem 核心有改动时运行其对应单元测试。

交付：单卡原始 KV 搬运与检索窗口可用，逐字节检查通过，H/pinned/设备页统计有界。

### M3 恢复请求状态和 Host 会话复用

目标：在新上游缓存和执行调度中保留 KVMem 的 checkpoint 语义。

主要位置：`program/prefill.cpp`、`decode.cpp`、`transactions/commit.cpp`、`program_impl.*`、`state/state_image.*`、frontend/query span、`runtime/engine/engine_core.h` 和 `context_cache/resource_manager.h`。

1. 梳理新上游 endpoint anchors、分支恢复和 worker recovery 的调用顺序，明确 KVMem 的状态由谁捕获、提交、恢复和释放，避免一份请求同时由两套缓存机制恢复。
2. 移植查询前 checkpoint、冻结 query、probe/replay、canonical input 与正常生成端点的恢复逻辑；保留 Main/MTP KV、GDN、token 位置和图片身份的一致性。
3. 保留短输入转长输入、同 query 工具续写、换 query、编辑历史和取消后的已提交状态；检查未提交生成端点不被发布。
4. 对共享 H 预留、FIFO 等待、取消/异常退役作小范围验证，确认请求不会重复独占整个 H。

精简验收：选用现有 `ninfer_memory_sessions` / `ninfer_p4_history` 中覆盖上述边界的场景；同一模型、同一配置下完成 A→B→A、一次工具续写、一次取消恢复和一次历史编辑。至少一个长历史超过物理窗口，避免只验证没有触发 KVMem 的短请求。

交付：会话状态和资源生命周期检查通过。字节搬运要求精确；已有文档记录的自由生成浮点差异不能直接当作新回归，也不能用它掩盖页数据或状态损坏。疑似原 KVMem/llama.cpp 行为问题按 `legacy-findings.md` 记录来源、触发、证据与影响，再与用户确认处理；迁移引入的回归直接修复。

### M4 恢复已支持的组合与服务入口

目标：迁移后的 worker 达到现有对外功能范围，且新上游特性不会绕过 KVMem 的限制。

1. 接回 fixed/adaptive MTP、ngram、fast prefill、C1–8、resident/CPU Vision 和多卡 layer pipeline 的已有适配。
2. 保留并发 ngram 宽度限制、图片必选区域、各 rank 的 KV/GDN/统计数据、首卡职责，以及 Host 历史不按 GPU 数重复计费的规则。
3. 更新参数映射、能力报告、`/props` / 状态统计和启动脚本。检查新模型族、suspend、grafts/router 等新入口是否需要在 KVMem 模式明确拒绝；默认不放行未验证组合。
4. 保留现有不支持组合的明确错误，以及磁盘冷快照禁用状态。原生路径与 KVMem 路径的限制分别说明。

精简验收按分支选代表组合，不做笛卡尔积：INT8 ordinary；RK8V4 Graph+adaptive MTP+ngram；一组 C2 图文和一组极短 C8 调度；resident 与 CPU 各覆盖一次图片身份/恢复；两卡一次文本恢复加一次含图片/MTP 的短路径。能由已有测试同一服务覆盖的场景合并运行。

多卡验收必须使用两张物理卡，记录 UUID 顺序和层划分；单卡模拟两段只能作诊断，不能代替此项。如只能取得一张卡，可先报告单卡完成，多卡资格保持未验证，不把整个迁移标记为全部完成。

交付：支持表与实际准入一致、受影响协议/参数测试通过，并保留既有已知限制。llama.cpp 仅检查共享启动入口未被破坏；没有修改其实现就不重跑整套 llama 模型测试。

### M5 复用上游三种量化实现并完成接入验收

目标：Q2_0、Q4_0、Q5_0 在新基线上形成可说明的转换、加载与执行闭环。

1. 以新上游的类型编号、格式表、Q2_0 解码和 GPU 实现为唯一生产实现；逐项处理本地 21 个文件，去除重复接线，保留增加实际覆盖的测试。
2. 复用步骤 1 的 packed-byte golden cases，检查 Q2_0 的 `{-1,0,1,2}`、Q4_0 的低/高 nibble 顺序、Q5_0 的独立高位 mask，以及 scale、行选择和 BF16 cast。
3. 使用步骤 2 保存的三格式 `.ninfer` 夹具进行跨版本加载；增加反方向加载仅在它属于实际兼容目标时进行，不默认建立双向兼容承诺。
4. 上游数值测试已有独立 host decode，并以其与 GPU dequant 完全一致为前置检查，再做 FP64 product 比较。复用该 oracle，不能只比较两个生产 GPU 内核。
5. 当前上游 `test_gguf.cpp` 默认跑 18 格式、两种 K 和 11 种 token 宽度，且没有命令行筛选。为精简验收，可增加小范围筛选入口，保留默认完整集合；本轮只运行三种新格式加一个既有格式控制项，选择 T=1、8、9 覆盖 vector/MMQ 边界，再补一个真实权重的 K。同时覆盖 linear、linear_add、两种 SwiGLU 入口和 embedding。
6. 核对实际模型配方是否把对应 tensor 按 GGUF blocks 保留下来。现有本地样本中 Q4_0 出现在 Bonsai MTP 权重，不能据此宣称整个模型能走 dense GGUF 配方。先检查该 MTP 来源和转换路径，再做一次确实执行 Q4_0 权重的短生成/MTP 请求。

本地尚无 Q2_0/Q5_0 真实模型样本。三格式的制品和算子测试可以先完整验收；没有真实样本的格式必须注明“算子/制品已验证，真实模型未验证”，不能合成一份改了格式标签的模型冒充实测，也不为此默认下载大型模型。格式支持与某个模型架构支持分别报告。

交付：新的量化支持说明、保留测试清单和明确的模型覆盖范围。上游代码已存在不等于本机已验收；此前本地步骤 3 未测试的实现不再作为正确性依据。

### M6 更新固定版本和交付文档

目标：让新的开发检出能重建已经验收的源代码。

1. 在同一组改动中更新 `.gitmodules`、ninfer gitlink、`backends/versions.json` 的 upstream/base_commit、`ninfer-kvmem.patch`、patch SHA-256 和 prepared tree hash。
2. 补丁从新的上游基线导出，包含确实保留的 Ryan 兼容差异和所有新增文件。使用临时独立 index 计算树，不污染用户真实暂存区，不用 PowerShell 文本重定向改变补丁编码。
3. 保持现有单补丁消费流程；通过分组说明记录来源和用途，不在本轮额外引入补丁栈框架。准备脚本只在发现实际不足时修改。
4. 在独立干净检出中运行 `prepare-backends.py --backend ninfer` 和 `--check --backend ninfer`，再验证重复准备不会重复应用或覆盖开发改动。不能在旧混合工作区强行切换子模块。
5. 更新仓库工作流、运行支持表、量化说明和本次验收记录，区分源码能力、测试范围与已有 ZIP。确认一次新构建 worker 的启动和 DLL 闭包即可，不重跑已通过的全部用例。

交付：可复现的固定版本、集成补丁和一致文档。发布 ZIP、提交/推送代码、替换现有运行服务属于后续交付动作，不包含在本次分析中；实施时也不能因更新版本清单就宣称旧包已经升级。

## 精简验证原则

| 阶段 | 必要证据 | 默认不做 |
|---|---|---|
| 当前分析 | 源码和版本核对、只读补丁适用性、文档引用/格式检查 | 编译、GPU、真实模型、性能测试 |
| M0 | 快照完整性、版本及差异清单 | 执行测试 |
| M1 | worker 构建、原生加载和短生成、必要兼容项 | 全目标构建、全套 CTest、设备自动校准 |
| M2 | 小页原始字节、位置/统计检查、一次越窗请求 | 九格式分别跑长模型 |
| M3 | checkpoint/会话切换、工具续写、取消/编辑恢复 | 256K/百万 token 长压测 |
| M4 | 受影响准入/协议测试及代表组合 | KV×MTP×Graph×Vision×并发×设备全排列 |
| M5 | 既有 golden/互操作、三格式定向算子与可用真实模型 | 重做量化内核、跑全 18 格式宽度矩阵、下载 37 GB 新模型 |
| M6 | 准备/校验、源树重建和交付入口 | 重复整套验收、llama NVMe 压力测试 |

功能/正确性测试先查询当前占用，使用能容纳工作负载的空闲 GPU，以 UUID/型号识别，不固定数字索引。性能比较不是本轮迁移完成的默认门槛，不据上游其他 GPU 的成绩宣称本机提速；若后续需要比较，只使用 RTX 5060 Ti `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`，固定模型、B/R/H、KV、MTP、Graph、设备 profile、prefill 对齐与请求数据。另一张功能测试卡为 RTX 5050 Laptop `GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29`。

默认编译并行度为 8。迁移候选的构建 wrapper 已将默认值从 2 更新为 8；只有实际资源压力或构建失败才降低，并记录原因。测试通过后，仅因新改动、失败或未解决的实质风险才扩大或重复测试。

## 主要风险与完成条件

| 风险 | 判断 | 处理 |
|---|---|---|
| 45 个文件不能直接应用补丁 | 已确认的文本冲突 | 按职责适配；不按冲突数量估算功能缺陷数 |
| 上游已改变页池、StateImage、endpoint 与执行入口 | 高风险兼容区，M2–M4 核心验证已通过 | 原始字节、状态、会话和双卡生命周期已核对 |
| Ryan 自带功能遗漏 | strict/mixed 缺失已确认，其他项已按来源盘点 | 与 KVMem 补丁分开记录来源和去留 |
| 旧 MSVC profile 缺陷 | M1 已在本机编译验证当前上游分段修复 | 采用上游实现，不搬回旧字节数组实现 |
| 量化 QType 编号不一致 | 已确认；v3 使用名称存储 | 统一上游编号、重编译、旧夹具加载验证 |
| 新模型/服务入口绕过 KVMem 适配 | 未适配组合已有明确拒绝 | 明确能力边界和准入检查，不扩大支持声明 |
| 快速上游持续变化 | 已确认近期活跃 | 一次验收对应一个 SHA；不自动跟随最新提交 |
| 已有自由生成差异被误判或掩盖 | 旧文档已有范围限制 | 区分字节状态错误与数值路径差异，保留原严格夹具与失败记录 |

迁移完成需要同时满足：现有 KVMem 合同在候选版本上通过受影响的精简检查；新旧行为差异已解释；三格式的制品与算子证据完整，真实模型覆盖边界明确；固定版本和补丁能从干净检出重建。未测的多卡/模型功能、保留待定的参数，不能用“已接入”一笔带过。

回退以 M0 保存的父仓库状态、旧 submodule pin、补丁、manifest 和旧 worker 为单位。新旧构建目录与测试会话目录隔离，不承诺跨版本内存快照通用，不覆盖当前服务的会话文件。单阶段失败先修复或回到该阶段恢复点，不把半迁移状态写入可发布版本。

迁移后继续直接跟踪 ninfer-all 的固定提交，按实际需要选择升级；每次先查看与 KVMem 触点重叠的文件，再更新补丁和相应检查。可向上游整理通用 Windows 或算子修复以降低维护成本，但提交上游不作为本轮必需交付。

## 初始分析的边界

初始分析完成了公开远端版本查询、独立源码下载、源文件核对、补丁只读适用性检查和计划编写。该次分析没有切换现有 ninfer 子模块，没有应用迁移补丁，没有改生产代码，没有编译或运行模型，也没有停止或替换服务。

## 实施记录

### M0 已完成

2026-10-07，按用户确认开始迁移准备。恢复资料位于 `ninfer-all-migration-20261007\m0-snapshot\`，包含父仓库、ninfer、llama.cpp 三份 HEAD 源码归档，各自的 staged/unstaged 补丁、原始改动文件、版本及暂存区记录。分别保存了 9、129、50 个工作区文件，另保存 14 个量化证据文件；每个复制文件的字节及三份源码归档完整性已核对。

隔离迁移目录为 `ninfer-all-migration-20261007\kvmem\`。父仓库以 detached HEAD 保持 `9b92d869…`，并带入捕获的父仓库改动；其 ninfer 是独立检出，固定为 `8319e8f…`，尚未应用 KVMem 补丁。llama.cpp 在该工作树中暂未初始化，原目录及恢复资料已保留。

`migration-inventory.json` 逐文件记录 111 个 KVMem 补丁文件、21 个量化增量文件和 124 个 Ryan 基线差异文件的迁移阶段与处理方式。Ryan 差异根据共同祖先 `f118551f…` 到当前锁定基线的六个提交获取，不把 Ryan 远端后来提交混入输入。清单中的待核对项仍需实际适配，不能据文件列表宣称已迁移。

尝试创建完全独立的父仓库克隆时，旧历史对象库缺少 `8cfd9b0996784d2cc81536758605bb8d75a72956`，导致克隆的 repack 失败。已改用原仓库的隔离 Git 工作树；它共享 Git 对象库，但有独立源码目录和暂存区。M0 未修复旧历史对象库，也未改变原分支、原后端、真实暂存区或生产源码。源码归档和原始文件副本可独立恢复，因此恢复资料不依赖这次克隆成功。

恢复时优先在新目录展开对应 `baseline.zip`，对照 `manifest.json` 处理删除文件，再恢复 `working-files/` 的原始字节；各仓库的 staged/unstaged 补丁分别保留以重建暂存状态。父仓库的两个后端须使用各自的恢复资料，不能只恢复父仓库的空 gitlink 目录。不要将恢复操作直接覆盖到仍有新改动的开发目录。

M0 仅做版本、快照和文件完整性核对，没有编译、GPU、模型或性能实验。原目录的 HEAD、捕获的工作区字节、状态和真实 index 在准备结束时均通过未变检查；之后仅更新本计划的实施记录。

下一步为 M1。候选工作树的父仓库 gitlink、versions 和集成补丁仍描述旧基线，M6 才会形成新的可发布固定版本；M1 应直接指定候选 ninfer 源码和独立构建目录，暂不运行会自动准备旧基线的 wrapper。当前目录是迁移输入，不能视为已经可发布的 KVMem 后端。

### M1 已完成

2026-10-07，候选 `8319e8f…` 已在本机 VS2022/MSVC 14.44、CUDA 13.2.86、`120a`、SM120 native 下完成原生 `ninfer-serve` 构建、启动和短生成。构建始终使用 8 并行，只选择 worker 与复用五个现有小测试对象的 bundle，没有构建完整测试或 benchmark 目标。独立构建目录为 `ninfer-all-migration-20261007\build-native\`。

保留了 Ryan 原生 strict/mixed 的参数、分配器、容量规划、worker 驻留检查和 JSON 报告。五个新增源文件与固定 Ryan 提交的 Git blob 一致；适配时保留新上游默认路径的 graft 页计数、状态发布顺序与持久化报告字段。非默认内存策略与新模型休眠、路由、graft、`qwen4_exp` 的组合明确拒绝；不导入托盘管理器。现有的单 GPU、文本、非评分、非 WDDM 等限制继续成立。

| 实际问题或兼容项 | M1 处理及证据 |
|---|---|
| 中文 MSVC `/showIncludes` 依赖前缀不匹配 | 独立构建配置匹配实际前缀，Ninja 单文件头依赖验证通过；仅重建早期依赖记录不正确的 CXX 对象，保留同一候选的 CUDA 对象 |
| 上游 C++20 `shared_ptr::unique()` 导致 C2039 | 使用等价的 `use_count() != 1`；现有字符串状态分叉/事务测试通过 |
| 快速 NVFP4 prompt 嵌套 lambda 的 kernel 别名导致 C2326 | 直接配置和启动同一 kernel 模板，参数及计算不变；CUDA 编译、链接通过，其数值路径留待后续对应功能验证 |
| 上游单模型启动 catch 丢弃异常原因 | 恢复旧入口已有的原因日志；无效 Legacy 配置的实际失败复现确认日志包含原因，未改变准入逻辑 |
| 旧 profile 长字符串及其他旧 Windows kernel 绕行实现 | 当前上游分段 profile 与 staged descriptor/launch 实现通过完整 worker 编译，采用上游；仅修复上面实际出现的额外 NVFP4 launcher 问题 |

精简验证为 **5/5 小测试通过、3/3 原生短生成通过**。五个小测试覆盖 CLI 参数、服务参数到 Engine 的映射、KV 容量、缓存/内存策略归一化和字符串状态分叉。三个实际策略分别为 default、strict、mixed；均输出 `1, 2, 3, 4, 5`，提示词 22 token、输出 14 token，检查了后续解码、统计与 Ctrl+Break 正常退出，三个进程均以 0 退出，未强制终止。

运行时 5060 Ti 被占用，按 GPU 偏好使用空闲 RTX 5050 Laptop `GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29`，没有等待指定显卡。已有 Bonsai 模型使用 max_context 512、prefill_chunk 128、并发 1、BF16 KV、FP16 GDN、关闭 MTP/ngram/自动校准；采用 Host 保留为 0 的 Hybrid 缓存，仅 1 个 Device snapshot、0 个新 tap。default/strict 使用显式 512 KV，mixed 使用 auto 且实际解析为 512。strict 的驻留检查通过，Shared 为 413138944 B，与基线相等，64 MiB 预留验证通过。

首轮测试的 Legacy 零 Host 预算被正确拒绝：15 个必需状态图像需要 1176852480 B。随后 Hybrid 默认保留槽与 Graph 预留需要 448301312 B，但 5050 当时 CUDA 只报告 278458368 B 可用。保留了失败日志与命令，并选用最小保留槽、关闭 Graph 的有效配置完成 M1；没有修改预算逻辑或缩减 max_context。**本阶段 Graph 实际运行未验收，留到 M3 的执行状态验证**；没有做性能、分页压力、strict 自动 OOM 恢复矩阵、完整测试集或量化数值矩阵。独立 CLI 可执行文件未构建，M1 验证其参数链及本阶段实际 worker。

完整来源与结果在 `ninfer-all-migration-20261007\m1-results\report.json`；`cpu-tests.json`、`native-smoke.json`、逐策略请求/响应/统计和日志保留原始证据。`source-snapshot` 保存 35 个改动文件及新增源码。结束时确认原父仓库及两个后端的 HEAD、真实 index、除本计划外全部捕获的工作区文件均未改变，没有替换生产 worker。

以上是 M1 当时的记录。用户随后授权连续推进至迁移及核心测试完成；M2–M6 结果如下，候选工作树的 wrapper 现在使用已固定的新基线。

## M2–M6 完成记录

2026-10-07，111 文件 KVMem 补丁已适配至 `8319e8f…`；按 M1 已移植的策略做三方适配，实际处理 26 个冲突文件，保留上游新的 tail/VMM、ngram 输入索引、logprobs 和 layer-ready 回调。旧量化增量逐项采用上游生产实现，保留有独立价值的 golden/reader 测试，不带入旧未完成的 Q2_0 算子。

- M2：九种 KV 原始 Main/MTP payload、INT8 sparse attention、RoPE、mean-K/query 通过。实际 B128/R128 的 257-token 输入生成 130 tokens，跨过 R 后重选两次，KV/GDN 字节校验通过。
- M3：A→B→A、Host 压力淘汰、取消恢复、编辑、工具历史的 canonical input checkpoint 通过。
- M4：RK8V4 Graph、自适应 MTP/ngram31、C8 两个短批次、resident/CPU 图像，以及 5060 Ti＋5050 的 40/24 层文本恢复和 C2 图像 MTP 通过。图像替换触发身份失效，取消恢复及预算拒绝后的恢复通过。未适配入口已有明确拒绝及聚焦检查。
- M5：16 项 CPU 转换测试、Q2_0/Q4_0/Q5_0 与 Q8_0 控制的独立 GPU 数值检查、旧/新 writer 到新 reader 通过。实际 Q4_0 MTP 制品九个矩阵的 36 行原始字节一致，实际 MTP/ngram 请求通过。Q2_0/Q5_0 无本地真实模型样本。
- M6：固定 URL/gitlink/版本清单/完整补丁及准备后树；独立干净检出的准备、重复执行、本地改动保护、严格树校验与旧 pin 拒绝通过。最终 worker 的完整 DLL 闭包、清理 PATH 后 HTTP 生成/props/stats 和正常退出通过。

全部编译使用 8 并行。共执行 25 个选定核心程序调用及上述转换/互操作/交付检查，没有运行完整 CTest、全排列、长上下文扫描或性能实验。远端 CI 的源码准备校验已配置，未推送或运行远端 CI。

迁移结果保存在隔离工作树 `ninfer-all-migration-20261007\kvmem`。原混合工作区及其两个后端保留；迁移阶段未替换运行服务、提交/推送或发布 ZIP。用户随后授权将验收后的源码推送到独立分支 `feat/ninfer-all-migration`；后续真实 Q2 制品与服务验证见验收记录。最终版本信息以该候选的 `backends/versions.json` 为准；详细失败修复、核心配置和限制见 [验收记录](ninfer-all-migration-validation.md)。
