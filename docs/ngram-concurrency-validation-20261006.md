# KVMem ngram 并发补测（2026-10-06）

本报告的图片 ngram 范围是当时的五种 KV。九种 KV、快速 prefill 和多卡的当前边界见 [多后端运行](multi-backend.md)。

Engine、服务端和统一 Windows 启动器移除了 ngram 的额外单路门禁：需要 MTP，宽度1–15允许 C1–8；启动并发 C2–8 时，配置的16–63自动降至15并提示，C1保留原宽度；多路上限沿用原生 GDN 16列验证工作区。本轮文本准入包括 BF16/INT8/NVFP4/K8V4/RK8V4，resident/CPU 图片放开当时的五种 KV，后续五种 KV 验收见 [全部KV图片ngram补测](vision-ngram-all-kv-validation-20261006.md)。R 必须覆盖 `max(MTP drafts, ngram drafts) + MTP drafts`。默认并发仍为1，统一脚本默认 ngram0，快捷脚本仍是单路/ngram31、视觉1024。

## 参数和真实模型检查

Windows／CUDA13.2／MSVC2022 Release／SM120a，默认8个编译任务。功能测试使用空闲 RTX 5060 Ti，UUID `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`；另一张5050空闲但8GiB显存装不下此 IQ3_S-vision-mtp 模型。没有测量或宣称性能提升。

- 首次放行时，Engine options、服务端 options 和 Windows 启动器通过。覆盖五种文本KV、两种图片驻留、C2–8、ngram1/4/15、固定/自适应MTP，保留16/31/63多路拒绝和未放行图片KV拒绝。
- NVFP4／固定MTP3／ngram15／C2／eager：复制、普通神经草稿请求混跑、恢复、取消与恢复同伴通过；复制接受36个ngram tokens，冷/恢复输出逐token一致。
- INT8／自适应MTP3／ngram15／C8／eager：B768/R128/H4096MiB，七路各256-token Python复制和一路普通请求混跑通过。冷启动峰值2路，恢复/取消/恢复同伴峰值均8路；恢复103轮/208行，七路复制共接受1673个ngram tokens。复制输出严格为各自源文件的前缀，冷/恢复/取消后恢复逐token一致。Main/MTP结束页与活动Host预留为0，全程遵守全局H。
- NVFP4／固定MTP3／ngram15／C2／resident视觉／Graph：完整复制、普通请求混跑、恢复、取消和恢复同伴四阶段通过。复制每轮接受239个ngram tokens，恢复98轮/113行、峰值2；复制冷/恢复/恢复同伴逐token一致。颜色独立控制改用单词回答后，两路红/蓝均正确。
- INT8／自适应MTP3／ngram4／C8／CPU视觉／Graph：完整验收通过，六路图片、两路文本混跑；冷启动峰值3，恢复/取消/恢复同伴均峰值8。恢复111轮/461行，七路复制共接受1428个ngram tokens；复制冷/恢复/取消后恢复逐token一致，八条后续红/蓝图片问答全部正确，Main/MTP结束页与活动Host预留为0。
- 预热修复后，INT8／自适应MTP3／ngram15／C8／Graph 的完整文本复制、普通请求混跑、恢复、取消及恢复同伴通过；峰值8，恢复103轮/208行，复制接受1673个ngram tokens，冷/恢复/取消后恢复逐token一致。接受数及恢复轮/行数与同配置eager一致；没有跨进程导出token序列作逐token比较。

图片测试使用512×512纯色PNG，每图实际256个 merged tokens，配置上限1024；B1536/R128。此前的实际1024-token图片并发测试见 [视觉并发验收](vision-concurrency-validation-20261006.md)，其 ngram 关闭，不混作本次证据。

## 原生 Graph 启动缺陷与修复

IQ3_S 模型／C8／ngram15／Graph 在 batch5 验证80列时首次调用 cuBLAS，句柄创建已经处在图捕获中，导致启动失败。关闭 KVMem 的原生服务同层、同列数复现；固定基线的预热、GGUF路由与句柄创建代码相同。完整来源、触发、证据和影响记录在实验根目录 `legacy-findings.md` 的 L009。

已在 MTP/ngram 各 batch 捕获前运行既有 eager 预热，不改变验证数学、量化、选页或草稿接受规则；原生及 KVMem 共用该修复。修复后的八路ngram15/Graph完整 KVMem 实测通过。该改动属于 ninfer 初始化，未修改原 KVMem/llama.cpp 路径行为。

原生对照服务使用与失败时相同的模型、C8、INT8、MTP3自适应/ngram15/Graph、context512及KV4096，修复后15.1秒 engine ready，HTTP健康检查通过；随后关闭对照进程。

## 测试修正与限制

- 初版B384复制查询本身约600 tokens，目标模型提前停止，冷/恢复仅峰值2；没有算作八路验收。增大到B768并加长既有历史后，各复制生成256 tokens，恢复与取消峰值8。
- 初版颜色问题要求分析“dominant color”，32-token实际回答仅含分析开头；没有把该失败认定为串图。单词回答控制通过，两次日志均保留。
- 没有对其余三种文本KV做本轮八路真实模型验收，没有全排列、200K上下文或36K/16K默认大预算验收，没有更新旧ZIP。
- 证据位于 `backend-p0p1-20261002/results/ngram-concurrency8-20261006/`。本轮验证程序退出后关闭自身GPU进程，原服务保持关闭。

## 启动时自动降至15

随后按用户要求，将多路宽度门禁改为启动时自动调整：合法并发 C2–8、请求宽度16–63时，实际宽度固定为15，启动日志显示原值、实际值和并发上限。直接 Engine API、原生/KVMem 服务端和统一 Windows 启动器均执行该规则。C1 保留宽度；0保持关闭；宽度64及非法并发仍报错。R按实际宽度校验。启动后只有一个活动请求也不临时恢复宽度。

- 使用既有 Release／CUDA13.2／SM120a／MSVC14.44 配置，8个并行编译任务，生成服务端及两组参数测试程序。
- Engine参数、服务端参数和Windows启动器测试均通过：五种文本KV、resident/CPU图片、C2–8、宽度0/1/4/15/16/31/63及固定/自适应MTP；未放行的图片KV、没有MTP、非法宽度等组合继续拒绝。原生Engine另覆盖MTP/DFlash/DFlash2、C1/2/8，检查单次调整警告及64拒绝。
- RTX5060Ti（UUID `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`）／IQ3_S／INT8／MTP3自适应／Graph，C8请求ngram31，B768/R128/H4096MiB：服务启动通过，`/props`与`server_start.engine.ngram_draft_window`均为15，调整提示恰好一次。两条不同Python复制请求都严格匹配各自源文件前缀，接受54和89个ngram tokens，实际输出59和96 tokens。
- HTTP测试最初错误地要求输出恰好达到`max_tokens=96`，其中一条正常遇到模型停止token而输出59；修正为输出上限检查后，用保留的响应和完整运行记录复核通过，没有重跑模型。这是测试断言修正。
- 本轮验证配置调整和真实复制执行；完整八路重叠、恢复、取消与图文正确性复用上面的ngram15/Graph及视觉证据，没有重复全矩阵模型测试或作性能结论。

本轮证据位于 `backend-p0p1-20261002/results/ngram-concurrency-clamp-20261006/`；测试服务已关闭。`ninfer-serve.exe`和`ninfer-serve-concurrency8.exe`已同步，旧ZIP未更新。
