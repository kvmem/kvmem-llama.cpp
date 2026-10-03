# 多模态接入实现计划

日期：2026-09-13。依据：[完整设计](multimodal-integration-design-2026-09-13.md)。

状态：2026-09-14 已完成全部功能实现、实机验收和默认服务启动。以下按实际覆盖范围勾选；原生 CLI 多图独立对照保留为补充验证项。实现与证据见 [实现与验收报告](multimodal-implementation-report-2026-09-14.md)。

## 交付目标

- IQ3 加载指定 `mmproj-Q8_0.gguf`，视觉头可选择 GPU 或 CPU；主语言模型设备遵循原脚本。
- 支持 OpenAI 格式图片请求、流式回答和带图后的文本 MTP 2。
- 同一驻留会话的前缀未变时，追加图片复用已有主模型 KV、GDN 和有效草稿状态，只处理新增内容及明确计数的必要短尾/query replay。
- KVMem 换出换入、Q8/Q5/Q4/F16 KV 和已有纯文本功能保持正确。
- 图片编码、视觉位置、模型图、attention、采样及多模态前缀比较复用 llama.cpp。

最终交付不能通过重建主上下文、全量重算公共前缀或悄悄关闭 MTP 来满足表面上的“能看图”。历史被修改、模型被替换、进程重启或会话缓存不存在时，按实际有效边界处理，不把这些情况宣称为完整命中。

## 阶段与依赖

| 阶段 | 主要成果 | 依赖 | 完成门槛 |
|---|---|---|---|
| P0 | 原生基准与两个关键接口原型 | 无 | 确认模型兼容、CPU/GPU 编码可用，以及逻辑位置和双路 MTP 输入的可行接入点 |
| P1 | 多模态输入、视觉头加载和统一 prompt 表示 | P0 | 原生媒体分块、图片 ID、前缀比较可直接复用 |
| P2 | KVMem 逻辑行与模型位置解耦 | P0、P1 | 常驻视觉 KV 与原生行为对齐，纯文本回归通过 |
| P3 | 视觉 KV 换出换入及媒体组检索 | P2 | KV 字节、位置、cell 元数据均可恢复 |
| P4 | 同一主上下文中的增量图片 prefill | P1–P3 | 60K 文本追加图片，不重算有效公共前缀 |
| P5 | 视觉 MTP 同步和图片后 MTP 2 | P0、P2、P4 | 草稿无视觉缺口，输出及接受/回滚正确 |
| P6 | 请求提交、取消和失败恢复 | P3–P5 | 失败 suffix 不污染缓存，已提交前缀仍可继续使用 |
| P7 | IQ3 实机验收、性能报告和默认脚本 | P1–P6 | CPU/GPU 看图与完整长会话通过，再更新默认配置 |

P0 先验证高风险接口，再铺开其余改动。每阶段完成后保留对应测试证据；某个门槛失败时解决该层问题，不以功能降级跳过。

## P0：建立对照与验证原生接口

- [x] 记录当前根仓库/llama.cpp 版本、已有改动、编译选项、IQ3 启动参数和模型文件标识，保留已有工作。
- [ ] 使用同一个 IQ3 主模型和指定 projector，建立原生 mtmd 的单图、多图、text→image→text 对照。原生单图 CPU/GPU 与原生内存模式已验证；多图、text→image→text 已在自定义服务验收，未单独补跑原生 CLI 多图对照。
- [x] 分别运行视觉头 GPU/CPU 编码，记录输出宽度、图片行数、位置布局、权重占用、编码时间和采样显存峰值。
- [x] 验证 `server-common` 中 `server_tokens` 与媒体解析的最小链接范围；决定直接链接还是小范围共享组件拆分。
- [x] 做最小 batch 原型：逻辑行映射在 ubatch 分割/重排后保持正确，不更改原生模型位置。
- [x] 做最小 MTP 输入原型：视觉输入嵌入与上一行 hidden state 分开进入已有 `embd`、`h` tensor，跨 batch 不错位。

涉及位置：`CMakeLists.txt`、`backends/llamacpp/tools/mtmd`、`backends/llamacpp/tools/server/server-common.*`、`backends/llamacpp/src/llama-batch.*`、`backends/llamacpp/src/llama-graph.*`、`backends/llamacpp/src/models/qwen35.cpp`。

产物：原生对照数据、接口约定、最小验证记录。必要的小范围 llama.cpp 扩展在本仓库补丁机制中维护；不另写视觉编码或 attention 实现。

## P1：接入视觉头与多模态请求

- [x] 链接 `mtmd`，加入 projector 生命周期管理、加载错误处理和维度校验。
- [x] 支持 `--mmproj`、`--mmproj-offload` / `--no-mmproj-offload` 及原生图像 token 上限参数。
- [x] 接入 `image_url`：base64 data URI 与 HTTP(S) 图片读取，复用原生下载/解码工具和限制。
- [x] 将现有纯 `vector<llama_token>` 的 prompt 缓存表示升级为可持有原生 `server_tokens`/媒体 chunk 的表示。
- [x] 缓存身份包含图片内容 ID、projector 与预处理配置版本；不按 URL 或统一占位符判定图片相同。
- [x] 模板、工具调用、采样、usage 和 SSE 尽量保留现有调用链；无 projector 的图片请求返回明确错误。

涉及位置：`tools/llama-kvmem-server.cpp`、`CMakeLists.txt`，以及 P0 确定的原生共享组件。

验证：同图同位置命中、同尺寸不同图不命中、媒体 chunk 不被截断、错误图片不触发模型状态提交。此阶段先在独立测试入口验收，不更新 IQ3 默认脚本。

## P2：解耦缓存行、模型位置和物理槽

- [x] 明确 `logical_id`、`model_pos`、`slot` 的接口类型和语义，新增随 batch/ubatch 传播的逻辑映射。
- [x] 块定位、预算和 capture 使用逻辑行；原生 RoPE 输入及 KV cell 二维位置继续使用模型位置。
- [x] 增加逻辑游标、逻辑截断接口，审计所有把 token 数直接用作 `seq_pos_max`/`seq_rm` 参数的调用。
- [x] 按 cell 集合处理局部移除，解决多个图片 patch 共享 `t` 时的误删问题。
- [x] 处理文本/图片交界的混合块，以及图片之后逻辑位置与 RoPE 位置不再相等的文本。
- [x] 检查 graph 复用、Q capture 和 batch 位置数组布局，不复用不兼容的输入绑定。
- [x] 保持 GDN 原生状态位置语义，另行记录其对应的逻辑边界。

涉及位置：`src/adapter/llama-kvmem-batch.*`、`llama-kvmem-capture.*`、`llama-memory-kvmem*`、`llama-kvmem-hooks.h`、`kvmem_store.*`，以及必要的原生 batch/cell 接口。

验证：重复 `t` 的每个 patch 有唯一缓存行；图像跨 ubatch 后映射一致；常驻且不发生检索裁剪时，与原生 mtmd 的位置、mask、logits 和输出对照；纯文本路径回归通过。

## P3：换出换入与检索适配

- [x] 扩展块元数据，保存原生位置、cell 扩展字段、媒体组和有效行边界。
- [x] 复用 packed K/V 保存与复制路径，恢复时同时恢复元数据，不重新旋转视觉 K。
- [x] D2D 重排、CPU/NVMe 换出换入覆盖纯文本块、视觉块和混合块。
- [x] 主 KV 与 MTP KV 使用各自实际类型和行跨度，不共享错误的字节布局假设。
- [x] 在现有检索选择器上增加媒体组约束：新图整体保留，历史图按完整媒体组选择。
- [x] 预算按实际行数统计，容量不足明确返回，不静默丢图或清空会话。

涉及位置：`kvmem/include/kvmem/raw_kv_store.hpp`、`kvmem/src/host/raw_kv_store.cpp`、`kvmem_store.*`、`kvmem_runtime.*`、`src/adapter/llama-memory-kvmem*.cpp`。

验证：Q8/Q5/Q4/F16 的实际 K/V 字节与元数据恢复；换到不同 slot 后不变；固定同一个选中块集合，对照换出前后 logits。不要拿裁剪后的稀疏窗口与原生完整窗口要求逐位相同。

## P4：增量 prefill，保留已有文本前缀

- [x] 用原生公共前缀比较确定 `[0,L)`，用原生位置换算确定后续输入的模型坐标。
- [x] 复用原生 helper 分批解码，通过薄回调传递逻辑行映射，不复制其视觉 batch 构造逻辑。
- [x] 主上下文保持驻留，仅对 `[L,end)` 执行新增文本或视觉 prefill。
- [x] 已命中的历史图片不再调用视觉 encoder；需要局部 replay 的投影嵌入采用有容量上限的 CPU 缓存。
- [x] 将必要的未提交尾 token、GDN 边界恢复、query replay 与真正新增行分开计数。
- [x] 在媒体边界按需维护有上限的检查点，不为每个 patch 保存整份 recurrent 状态。
- [x] 增加实际 decode 区间追踪与 context 身份记录，证明有效前缀没有再次执行。

涉及位置：`tools/llama-kvmem-server.cpp` 的 `run_prefill_retrieval`、缓存提交和检查点逻辑，以及原生 helper 回调。

验证：约 60K 文本后追加新图、同图继续追问、再追加第二张图；逐次核对前缀命中、encoder 调用数、实际执行行范围。历史图片被替换时，只复用仍有效且可恢复的图前前缀。

## P5：同步视觉 MTP 并继续 MTP 2

- [x] 将 P0 的双路输入原型接入正式 batch/ubatch、graph 输入填充及图复用判定。
- [x] 扩展原生 MTP `process()` 处理视觉输入，按 `(h[p-1], x[p])` 关系同步草稿 KV。
- [x] 跨 text→image、image→text 和 ubatch 边界维护 `pending_h`，直接使用本次主模型已产出的 hidden rows。
- [x] 更新 `begin`、draft、verify、accept、restore 的双游标逻辑，不再以 prompt 行数代替模型位置。
- [x] MTP 跟随同一套块到 slot 映射，但保存其独立计算得到的 K/V。
- [x] 在 checkpoint 中纳入草稿游标及 carry 状态，生成前验证草稿覆盖到正确的已提交边界。
- [x] 图片后的文本启用现有 MTP 2；同步失败明确报错，不悄悄用普通 decode 完成请求。

涉及位置：`backends/llamacpp/common/speculative.cpp`、`backends/llamacpp/src/llama-batch.*`、`backends/llamacpp/src/llama-graph.*`、`backends/llamacpp/src/models/qwen35.cpp`、`tools/kvmem-spec.*`、`src/adapter/llama-memory-kvmem-mtp.*`。

验证：先做固定输入的普通 decode/MTP greedy 对照，检查近并列 logits 导致的数值差异；再测试全接受、部分接受、全拒绝和回滚。官方随机采样测试记录接受率、吞吐与采样行为，不要求跨不同执行路径的同 seed 输出机械一致。

## P6：一致提交与错误恢复

- [x] 将 prompt、位置映射、主 KV 有效边界、GDN 检查点和 MTP carry 作为同一提交边界管理。
- [x] 对新增 suffix 采用未提交状态，成功后才进入缓存命中表。
- [x] 图片读取失败、编码失败、decode 失败和客户端取消时，回到最后已提交边界。
- [x] 保护跨越最后一个不满块的旧前缀，避免新图片失败时误截断旧行。
- [x] 历史编辑超过已有检查点范围时，记录失效边界并自动重建当前请求缓存，不能宣称原 KV 仍有效；客户端不需要额外的重置字段。
- [x] 复测图后工具调用和工具结果续接，确保模板变化与前缀识别一致。

验证：每种故障后立即追加一个正常问题；确认旧前缀可命中、无残留图片行、GDN/MTP 状态同步且没有常规全量 reset。

## P7：实机验证和默认脚本交付

- [x] 27B 真实图片测试固定在 5060 Ti 或适用的 27B GPU，使用指定 IQ3 与视觉头；不能拿不匹配的 0.8B 模型代替 projector 兼容性验证。
- [x] 低成本索引/字节测试先完成；5050 只用于适配的小模型或独立 encoder 测试。GPU/CPU 视觉模式均完成实际看图测试。
- [x] 跑完整长会话矩阵：60K 前缀、新图、旧图追问、多图、强制换出、MTP 回滚和失败恢复。
- [x] 使用官方 Thinking 采样配置测 encoder、增量 prefill、decode、接受率、显存峰值、CPU 缓存占用；保留 greedy 正确性结果。
- [x] 根据实测确定图像 token 上限，验证现有 IQ3 KV/检索预算下 GPU 模式能否容纳；如不足，报告实测缺口，不擅自缩减预算。
- [x] 验收通过后再修改 `scripts/start-iq3.sh`：默认指定视觉头与 GPU，提供 CPU 选择方式；保留已有模型、采样和预算配置。
- [x] 更新 README、参数帮助、请求示例和测试说明，记录已验证的模式及边界。
- [x] 默认服务切换安排在实际实施交付阶段；记录原进程和配置，避免测试误停正在使用的 IQ4 服务。

复用现有 `scripts/server_smoke.py`、`scripts/chat_sampling_canary.py`、`scripts/mtp_canary.py` 和 `tests/mtp-kv-test.cpp`；新增测试只覆盖多模态特有边界和缺失的增量验收。

## 最终验收清单

- [x] CPU/GPU 两种视觉头模式均能读图，文本模型设备不被连带改变。
- [x] 60K 有效前缀追加图片时，同一个主 context 继续使用，旧前缀没有完整重新 prefill。
- [x] 旧图后续追问时，旧图 encoder 调用数为 0；必要 replay 的原因、区间和耗时可解释。
- [x] 图片后的文本生成实际运行 MTP 2，草稿视觉行无缺口、接受和回滚正确。
- [x] 换出换入后 K/V 与位置元数据正确，支持现有各类 KV。
- [x] 错误/取消后已提交前缀可恢复，纯文本、采样、流式和工具调用回归通过。
- [x] IQ3 默认脚本、文档、可复现请求、逐阶段日志和性能报告齐备。

至少记录这些观测量：`prefix_hit_rows`、`new_text_rows`、`new_image_rows`、`replayed_rows`、`vision_encode_calls`、`logical_cursor`、`model_cursor`、`mtp_synced_rows`、context 身份和 reset 原因。通过实际 decode 行范围验收，不能仅凭一个 cache-hit 计数认定没有重算。
