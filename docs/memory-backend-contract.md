# 公共内存契约

本契约在独立 C++17 核心中实现。最先接入的是 llama 的单 GPU、Flash Attention、RAM packed 路径，以及 ninfer 的 KVMem 工作集切换。当前产品组合以 [多后端运行](multi-backend.md) 为准，包括 ninfer 的九种 KV、文本和 resident/CPU 视觉，以及多卡 layer pipeline。公共类型的存在不表示所有后端都支持对应功能。llama 的其他物理配置保留原生兼容桥。

入口是 `kvmem/include/kvmem/memory_contract.hpp` 与 `native_kv_payload.hpp`，实现分别位于 `kvmem/src/host/`。新接口没有引擎、GPU runtime 或 Tensor 类型。

## 三种坐标

| 坐标 | 持有方与字段 | 用途 |
|---|---|---|
| 原始 token 位置 | `BlockDescriptor::original_start`、`MemoryStamp::frontier` | 历史身份、模型 RoPE 位置、提交进度 |
| 紧凑 attention 地址 | `ViewBlock::compact_start` | 按原始时间排序后的工作集读取和 causal 边界 |
| 物理页/槽 | 引擎私有 | 显存分配、地址表、native generation、stream 和 graph 生命周期 |

例如取原始位置 0、128、512 开始的三个完整 128-token 块，新视图的紧凑地址为 0、128、256；K 仍对应原始位置 0、128、512。公共契约不提供“把 K 重新旋转到紧凑位置”的操作。

这个示例是普通一维文本位置。`original_start` 始终是逻辑 token 序号；多模态的多轴 RoPE、rope delta 等真实位置元数据仍由引擎保存，公共层不能用这个序号伪造或覆盖它们。

`KvMemRemap` 的 `from_base/to_base/raw_refresh` 是旧 re-RoPE 设计留下的元数据。`KvMemStore::set_selection` 仍维护这些字段，旧测试也仍覆盖其行为；当前 llama 的 `apply_plan_to_kv` 使用 stage-in/out，`write_block_to_gpu` 恢复 packed K/V，`occupy_in` 使用保存的原始 row positions。`remaps` 在适配器中主要用于块遍历、trace 和统计，不能据其字段名推断产品又旋转了一次 K。

`MemorySession::prepare_reselect` 只调用既有 `pick_topk_blocks`，核对 policy/native catalog 的粒度、块身份和有效长度后，生成 `WorkingSetPlan`，不调用会修改旧 remap 元数据的 `set_selection`。选块算法没有复制。原生兼容桥尚在使用的旧执行元数据暂时保留。

## 职责

| 对象 | 负责 | 引擎继续负责 |
|---|---|---|
| `KvMemStore` | 既有排序、sink/recent 和强制保留策略 | 提供已提交的统计，选择前冻结 query |
| `WorkingSetPlan` | 块身份、版本、紧凑映射、换出/换入、预算 | 具体物理页号与临时 buffer |
| `MemorySession` | 版本检查、换页顺序、发布门槛、取消与失效 | 串行执行安全边界和线程同步 |
| `PayloadLayout` / `NativeKvPayload` | 原生 bytes、所有 planes、stride、alignment、格式版本 | 解释量化编码，执行实际复制 |
| `MemoryBackend` / `MemoryTransfer` | 必须实现的适配器契约 | 唯一的设备分配器、reservation、stream、event 和 graph |
| `CheckpointIdentity` | 精确检查后端/模型/布局/会话/执行历史/frontier | checkpoint 字节、GDN 还原与 query 重放 |

新后端不能继承旧 `KvMemBackend` 的默认空复制来表示已支持。`MemoryBackend`、`MemoryTransfer` 全部必需操作均为纯虚函数；缺失 backend、返回空 transaction 或缺少所需能力会明确失败。

## 身份、版本与位置边界

- `BackendIdentity` 固定后端、源码/实现版本、实际模型身份和 state schema。一个 backend 对象在绑定期间不能热换模型、布局或能力；更换必须创建新的会话/后端对象。
- `SessionIdentity` 包含会话 ID 和 incarnation。清空、编辑历史、重建会话时使用新 incarnation，不能只靠可重复的 block ID 区分历史。
- 每个块分别记录 `content_version` 与 `statistics_version`。Native payload 的复用只看 KV 身份/版本/有效长度，统计更新不会冒充 KV 写入；统计的来源与当前 revision 仍需适配器保证。
- `MemoryStamp` 覆盖 valid、incarnation、history revision、view version、逻辑 frontier 和 execution history。逻辑内容或统计变化推进 history revision；工作集发布推进 view version。任何相关状态变化必须反映在 stamp 中。
- prepare 使用完整逻辑 catalog；按时间连续，最后一个块可以不满。resident 必须表示当前完整版本，Host version=0 表示缺失，旧版本不能作为换入来源。输入中的未知 ID、重复 ID、漏掉 mandatory、粒度不匹配会报错，不静默修正。
- 即使没有选中某个冷块，它也必须保有当前 Host 副本；不能在新计划中默默接受已经丢失的冷历史。请求 Retrieval 策略时，后端必须声明 pre-RoPE 统计能力，否则明确拒绝；不把 sink+tail 能力当成检索已经接通。
- 选择结果按原始位置排序；compact 地址不会修改 `frontier`。多 query causal mask、图的 buffer 绑定和最大原始位置仍由实际引擎验证。
- `CheckpointIdentity::compatible_with` 采用严格相等。相同 token 文本不足以让 full/sparse/replayed GDN 状态互换。ninfer 的 query checkpoint 保存完整原生 StateImage、精确前缀和当时的驻留视图；恢复与重放仍由 Program 执行，公共核心不解释其字节。

## prepare → transfer → publish / abort

调用方必须暂停旧视图的新写入，并在引擎安全边界单线程驱动 `MemorySession`。异步设备传输可以进行，协调器本身不提供锁、调度器或后台线程。

```text
Idle / Published
   prepare(expected_stamp, selected, mandatory, budget)
Prepared
   advance()：start_spill
Spilling
   advance()：poll；Pending 保留所有源页
   Complete 后：start_restore，可回收 evict 页
Restoring
   advance()：poll；Pending 不允许发布
Ready
   publish()：检查版本 → 引擎发布 → 核对新 stamp
Published
```

1. 纯规划先检查布局、身份、所有选中来源及预算。`prepare` 必须预留所有目标 Host buffer、native 页 entitlement、复制暂存和新视图 metadata；失败必须释放预留，保持旧视图，不能提前启动复制。
2. `start_spill` 只复制尚无当前 Host 副本的 outgoing 块。源页在 poll 返回 Complete 之前不得释放；Complete 包含 stream/event 完成和内容版本确认。
3. `start_restore` 可以释放 evict 页并用同一原生页池恢复。适配器也负责按新紧凑顺序排列 retained 块，不引入第二套 GPU 分配器。
4. 所有恢复完成后进入 Ready，尚未自动发布。`publish` 必须在安全边界同时暴露页表、有效长度、Host/resident metadata 与新 view version。graph 不能引用已释放或仍 pending 的 buffer。
5. `abort()` 必须等待或取消在途工作，然后才释放 buffer/leases。能保证旧字节和旧视图都完整时返回 `PreviousViewPreserved`；否则返回 `SessionInvalidated`，协调器调用 backend.invalidate，拒绝继续推理直到建立新会话并重新 prefill。

每次推进和发布前检查 stamp；传输中出现过期计划会触发 abort。发布阶段抛异常或后端未公布预期版本时，强制失效，不能报告成功。析构和 disconnect 同样先 abort；`shared_ptr` 使后端生命周期长于 pending transaction。后端的 prepare/reservation 还需防止多个协调器同时取得同一 session 的所有权。

## 预算与原生 payload

`selected_tokens` 限制有效历史数量；`device_tokens_required` 按每个块的 native 分配粒度向上对齐，再加独立对齐的 `reserve_tokens`。配置必须尊重粒度，不把 32-token ninfer 配置偷偷当成 64 tokens。

`host_payload_bytes_required` 包括所有已占用的 Host 记录和此次换出的新记录，以 `record_bytes` 计量。旧 Host 副本占用的空间也必须计入。设备权重、GDN、workspace、graph、Host metadata、checkpoint 和传输临时空间不包含在这个字段中，由引擎自己的内存计划和 prepare 预留负责。

`PayloadLayout` 明确格式版本、block tokens、record bytes、alignment，以及每个 attention layer 的 K/V/可选 K-scale/V-scale 平面。平面不能重叠或越界；packed 行内 scales 保持在对应 opaque K/V bytes 内。`token_stride=0` 表示原生整页布局，公共层不猜测其内部张量顺序。

`NativeKvArchive` 是有容量约束的普通 Host byte archive；它保存的是字节序列，不能将其 vector 地址当作满足 native alignment 的设备或 pinned buffer。引擎可继续使用自己的 pinned Host store，并兑现相同版本/完成规则。ninfer 的页池无需替换为这个 portable archive。

`raw_kv_payload_layout/export_raw_kv_payload/import_raw_kv_payload` 为现有 `RawKvStore` 的 packed token-row K/V 提供独立边界：

- 只包含显式列出的 attention layers，缺失 KV 层不能伪造成全零记录；GDN 层不产生 KV planes。
- 不读写 mean-K sums，不调用重新量化或 RoPE；部分尾页导出将未使用的序列化字节清零。
- 导入前核对 backend/model/layout、原始位置和行 stride。旧 raw packed frontier 更长时，拒绝短恢复，调用方必须先显式 invalidate packed tail，避免暴露旧尾部。
- 该桥接适用于已有 packed-row 路径；其他 native layouts 和 checkpoint 恢复由实际适配器负责。
- 来源冻结与内容版本来自引擎；`RawKvStore` 没有模型或会话标识，桥接不能自行推断它们。导入中若发生分配或 I/O 错误，调用方必须将本次恢复视为失败并按事务规则处理。

## 两后端能力表

P3 首步已让 llama 的检索、预填充压力及会话 attach/detach 共用公共 `plan_residency`。
这个函数只计算原生块的保留、换出和换入集合，供 `WorkingSetPlan` 和旧槽位执行器共同使用；不解释位置、不分配内存。
适配器通过过渡类型 `NativeResidencyPlan` 调用它，准备阶段不修改 `in_working_set`，成功分配槽位后才提交驻留成员；`baked_pos`、`remap_count`、`remap_abs_delta` 不再参与产品执行。
关闭 stage-in 复用的配置仍会强制换出并恢复选中块；重复或未知 ID 明确报错。

llama 的预填充计划可包含已登记、尚未计算 KV 的行，而 `MemorySnapshot.frontier` 只允许已提交 KV；旧 Recency 路径也不承诺所有冷历史都有 Host 副本。`NativeResidencyPlan` 保留这些区别；已经接入 `MemorySession` 的路径使用下文的 evaluated / accepted frontier。

| 契约能力 | llama.cpp 当前产品 | ninfer KVMem 首版 |
|---|---|---|
| 原始位置与物理存储分离 | 原始 row metadata 和 packed 恢复 | 原始 RoPE position 与紧凑 cache position 分离 |
| native K/V payload | 既有 packed rows/raw mirror | BF16 / INT8，包含独立 scale planes，普通 Host 归档和有界 pinned 暂存 |
| 有限设备工作集 | 既有预算语义 | 显式 B+R，H payload 上限，执行前 admission 检查 |
| mean-K / query 统计 | 既有捕获与统一接受提交 | RMSNorm 后、RoPE 前捕获；同一公共 mean-K 选择算法 |
| GDN / checkpoint | 保留引擎专用实现 | 精确 query 前 StateImage、前缀与驻留视图，探测后恢复重放 |
| 搬运 / 图 | 保留专用实现 | 同步安全边界搬运；图开关、原生字节与状态一致性验收 |
| 连续多轮、编辑、取消 | 保留已有能力 | 可信 checkpoint 复用；失配或无 checkpoint 时从头计算 |
| MTP、并发、多模态、磁盘恢复 | 保留已有能力 | 首版不开放 |
| 接入 `MemorySession` | 单 GPU FA + RAM packed；其他配置保留兼容桥 | 所有已支持的 KVMem 工作集切换 |

上表的「ninfer KVMem 首版」只描述 P3 第一步。后来的 MTP、并发、视觉和多卡在 ninfer 适配器里实现，不以该行的「首版不开放」为当前产品边界。磁盘冷快照入口当前关闭。当前组合见 [多后端运行](multi-backend.md)。

## 验证与复现

```sh
cmake -S . -B build-host -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF -DKVMEM_ENABLE_NVME=OFF
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

P2 独立构建目标：Windows / MSVC 19.44、C++17，无 llama 子模块构建、无 ninfer 链接、无 GPU runtime。P2 的两个新增可移植测试与 P3 追加测试覆盖：

- `memory_contract_test`：packed-row、BF16 native page、INT8 多平面三种形状；按原始位置排列、尾页和生成预留；两次选页的数据恢复；mandatory、重复/未知块、过期来源与容量错误；pending 阶段禁止回收/发布；复制/恢复/发布失败；取消、disconnect、析构生命周期；复用原选择算法而不改变 legacy remap；checkpoint 隔离。
- `native_kv_payload_test`：全部 planes 和 padding 的精确复制、跨模型/会话/layout 拒绝、旧版本/短记录/坏布局、Host 预算；32/64/128 tokens 三种块大小 × 1/部分尾页/完整块共九种 RawKvStore 往返，验证源和目标 mean-K 保持不变。
- P3 `native_residency_test`：400 组独立集合检查、32-token 块及 5-token 尾块、四个物理槽位上的精确字节往返、强制重载、未计算的预填充行、换出完成前不释放槽位、会话换绑、分配/复制失败时保留 Host 源与回收未发布目标槽。验证原始位置和全部 legacy 重旋转元数据不变。

测试后端只在测试文件里模拟完成事件和故障，不能用作生产后端。这些测试证明公共库的行为与生命周期，不替代 P3/P4 的实际 GPU、GDN、取消、吞吐与质量验证。Windows 不运行既有 POSIX/NVMe 专用测试，本次没有修改其实现。


## P3.1b：生产原生事务与三种 frontier

`KvContent` 随 `ConvStore` 移动。它区分预留地址 `reserved`、已过图完成 fence 的 `evaluated`、已接受的 `MemoryStamp.frontier`。预填充登记位置不会生成有效 `BlockDescriptor`；分多次 ubatch 完成时，只能导出已经完成的前缀。MTP target 验证前通过显式 hook 标记推测执行，宽度为一也不会自动推进接受边界；`decode_mean_commit(n_keep)` 推进接受前缀，随后 truncate 丢弃拒绝尾部。未完成计算或接受尚未结算时，不能发布或迁移会话。

会话 ID 在进程内唯一，reset/replacement 更换 incarnation；append、重放和截断更新内容/历史版本。尾块的有效长度属于版本描述，不能拿完整旧块覆盖一个截断后的尾块。mean-K/Q 仍由现有捕获与接受逻辑管理；这里的 `statistics_version=0`，适配器不宣称公共 `pre_rope_statistics` 或 `state_checkpoint` 能力。GDN checkpoint 仍归引擎。

实际 `NativeBackend` / `NativeTransfer` 使用公共 `MemorySession`，目前启用范围是单 GPU、Flash Attention、packed K/V 保存在 RAM、旧 CPU/NVMe arena 关闭且启用驻留复用。其他组合保留原桥接，尚未宣称完成公共接口迁移。每个 lane 有自己的物理槽池；事务对象仅活在持有引擎安全边界的同步调用内，不会比引擎所有者活得更久。

准备阶段校验槽池容量，预留 packed Host 目的容量与 GPU 搬运/重排 scratch，不释放旧 GPU KV。spill 等待真实复制和 Host 写入完成并检查所有 attention 层覆盖后，协调器才允许回收、恢复与布局。沿用原有 packed 复制和 MTP follower，不改变 RoPE。publish 验证原始位置、多模态位置元数据和完整行数，再公布 view version；pending 期间 attention proof 无效。准备/换出失败保留旧视图；进入可能破坏旧字节的 admission 后发生异常，则排空、清空并使会话失效，要求重新 prefill。

历史保留规则是 **RetainAll**：即使选择策略是 recency，RawKvStore 仍保留已有历史的 packed K/V。普通选择中的预算是有效历史；压力 admission 额外保留未来块槽位，未来行不进入已提交 catalog、Host 有效长度或公开执行视图。部分尾块新增行使用现有槽位，只有新块计入独立 reserve。

真实 GPU 故障测试已覆盖 prepare、spill、restore、publish 四个阶段以及 fresh prefill 恢复，并检查 pending 视图不能逃逸。0.8B 多 ubatch、三组 KV 格式重放和 1/2/4 lane 服务回归已通过。5060 Ti 上的 27B KV/MTP/GDN 矩阵为 31 项通过；WDDM 下无法可靠制造物理显存 OOM 的 10 个检查明确跳过。普通解码和 MTP 服务回归通过，连同小模型共 54 个请求与 P3.1a 的回答及 usage 一致。性能对照详见工作区 P3.1b 报告。旧全序列删除后残留 packed KV 的路径已修正，hybrid clear 也先排空捕获再清除源 KV。

## P3.2：统计附件身份与统一接受提交

`ContentPrefix` 是 `KvContent` 发出的进程内逻辑前缀凭据，包含会话身份、接受位置和撤销状态。引擎在同一序列化边界调用其创建/校验/失效操作；它不是跨进程快照，也不证明全量与稀疏 attention 的 recurrent 状态等价。追加和同历史重放保持旧前缀，截断穿过前缀、reset 和 invalidate 永久撤销。弱引用目录清理已释放的凭据，不记录每一次生成事件。

llama query 附件带捕获范围和前缀凭据，尾块 mean-K 附件带 checkpoint 行号和前缀凭据。导入还核对维度、有限值和有效行数。默认 user-query / 多模态 checkpoint 路径在恢复任何 recurrent/draft 字节前检查该凭据；选择时跳过已失效的候选，无法找到可信状态则从完整输入重新 prefill。输入/media 的逐前缀匹配继续由 driver 负责，统计附件的凭据不能替代该匹配。

MTP 产品路径使用 `begin_speculative_evaluation` / `commit_speculative`。开始时绑定当前会话和执行历史，完成图计算后仅记录 evaluated 范围；接受时检查同一 capture、起点和宽度，先完成 GDN fold（若 recording），再推进 accepted KV 与 mean-K。未接受的草稿不生成 query 统计，不经 prefill mean-K 路径提交；零接受、部分接受、宽度一拒绝、超量/重复提交有真实 GPU 回归。原专用 hooks 保留，不要求 ninfer 使用 llama 的 tensor 或 recurrent 布局。

这一步明确了适配器的统计/状态边界，没有把 Host worker 的完成事件伪装成公共 statistics version。`statistics_version=0`，公共 `pre_rope_statistics` / `state_checkpoint` 能力仍不声明。通用统计搬运接口、旧 server 编排分支和其余 legacy bridge 配置继续在 P3/P4 整理。

P3.2 主机测试 20/20，0.8B GPU 7 项、27B GPU 32 项通过（WDDM 物理显存 OOM 10 项跳过）。54 个既有请求与 P3.1b 的回答及 usage 一致，另有六次客户端断流后恢复通过；查询重放不改变首次 Q 的 sum/count。完整数据与范围见隔离工作区 `p32-report.md`。
