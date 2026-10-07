# ninfer 视觉并发上限调整（2026-10-06）

KVMem 的 Engine、服务端参数检查及 Windows 统一启动入口现已允许 1–8 条活动文本或 resident/CPU 图片请求，可混合图文。并发默认值仍为 1；`-Concurrency 8` 或 `--max-concurrency 8` 选择八路。启动脚本默认每图视觉上限由 256 改为 1024 merged tokens。

本报告的多路验收关闭 ngram；统一启动入口的 `-NgramDrafts` 默认是 0。其后的 ngram、九种 KV、快速 prefill 和多卡见 [多后端运行](multi-backend.md)。本报告自身不放行冷磁盘、视频、overlay、并发 prefill 或快速图片 prefill。CPU 图片编码沿用原生串行编码器。当时的 ngram 和五种 KV 图片记录见 [ngram并发补测](ngram-concurrency-validation-20261006.md) 和 [全部KV图片ngram补测](vision-ngram-all-kv-validation-20261006.md)。

## 已完成检查

- Windows VS2022、CUDA 13.2、SM120a 既有构建目录，使用 8 个编译任务；Engine、服务端及实图并发回归程序已编译。
- Engine 参数检查通过：五种 KV 格式、resident/CPU、1–8 路及 ordinary/自适应 MTP，另验证 81 个不支持的组合被拒绝。
- 服务端参数、OpenAI 协议、加载报告和资源管理检查通过。服务端参数检查覆盖上述视觉并发组合及超过八路的拒绝。
- Windows 启动参数测试通过，包括两种视觉驻留方式的 1–8 路、默认视觉 1024、ngram 关闭及八路能力描述。
- 集成补丁及 `backends/versions.json` 已同步；`prepare-backends.py --backend ninfer --check` 通过。

原服务关闭后，标准输出 `build-backends/ninfer/apps/ninfer-serve.exe` 也已重新链接；此前的独立输出 `ninfer-serve-concurrency8.exe` 已同步更新。可通过统一启动入口的 `-Worker` 指定任一文件。发布 ZIP 尚未更新。

## 实图验证

用户已授权关闭原服务进行测试。原两实例的参数保存在 `original-services.json`；测试使用 RTX 5060 Ti（`GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`）及既有 Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp 制品。以下均为双 INT8、device profile off、8K 逻辑上下文、prefill128、R128、ngram0 和 FP32 GDN；Host H 是全局预算，B 是每路预算。所有搬运字节检查保持启用。

| 入口与配置 | 每图实际 tokens | 每路 B / 全局 H | 结果 |
| --- | ---: | --- | --- |
| Engine，CPU视觉，2路，ordinary/eager | 256 | 1536 / 1024 MiB | 完整通过，峰值2路 |
| Engine，CPU视觉，8路，自适应MTP3/Graph | 256 | 1536 / 4096 MiB | 完整通过，取消和恢复阶段峰值8路 |
| Engine，resident视觉，8路，ordinary/eager | 1024 | 2560 / 4096 MiB | 完整通过，取消和恢复阶段峰值8路 |
| HTTP，CPU视觉，8路，自适应MTP3/Graph | 1024 | 1536 / 2048 MiB | 冷/缓存两轮共16请求通过，缓存阶段峰值8路 |

Engine 回归检查红/蓝图片与文本混合、不同会话口令、取消一路、恢复其他请求、同名图片替换、紧凑 decode batch、全局 Host 预算、结束资源释放及旧请求较晚结束时的会话发布顺序。八路用例各含六条图片请求及两条文本请求；每条图片请求有历史图与当前图两项。1024-token 用例记录每请求 `vision_tokens=2048`，确认实际两张图片各1024，完整视觉页需要增大 B。

CPU 八路用例的取消阶段有50轮decode、279行，恢复阶段17轮、74行；resident 八路用例分别176轮/900行及38轮/248行，峰值活动数均为8。终止后 Main/MTP resident pages 及活动 Host reservation 均为0，全部请求的口令和颜色通过检查。冷 CPU 图片串行编码，短回复不能保证与下一路形成解码批次；第一轮两路诊断因此触发了过严的批次断言。测试改为在不变的缓存会话阶段强制检查共享批次和配置的峰值活动数，冷图及替换图仍验证内容隔离、预算和资源释放。

HTTP 两轮各含六条图片请求与两条文本请求，每条图片请求使用一张1024×1024图片。冷阶段峰值3路、370轮decode/415行；缓存阶段峰值8路、59轮decode/416行。图片请求实际输入1087 tokens，缓存轮复用1086 tokens；颜色和各路不同口令均通过，Host预算及终止资源释放检查通过。`/props` 的 `total_slots` 和 `max_active_requests` 均为8，视觉启用、ngram为0。CPU的首次图片编码仍串行，不能把活动请求上限理解为八张图片同时编码。

HTTP 补测发现 `--derive-session-keys` 被 KVMem 集成门禁误当成原生缓存容量选项拒绝。修复允许 KVMem 自有历史使用该请求会话键提示，原生 Hybrid 缓存拒绝及关闭复用冲突保持原约束；另检查不兼容 Host 参数在两种参数顺序下仍被拒绝。完整参数回归通过。该问题是适配回归，来源和首个失败日志见实验目录 `legacy-findings.md` 的 L008。

测试结束后已确认所有本轮推理进程退出，两张GPU均显示0 MiB推理显存占用。按本次关闭原服务的指令，原服务保持关闭，配置留存可恢复。

完整默认 36k/16k/200k 配置、其他 KV 格式的八路实图组合及通用视觉质量未在本轮验收。本轮仅做功能正确性验证。

构建及检查日志保存在实验目录 `results/vision-concurrency8-20261006/`。
