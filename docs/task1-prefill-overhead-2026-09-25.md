# 任务一：5060 Ti 单卡与双卡预填充开销（2026-09-25）

## 测试口径

同一 `build-layer/bin/llama-kvmem-server.exe`、Qwen3.8-27B-UD-Q4_K_M、纯文本任务一、`-c 1024 -b 128 -ub 128 -n 64`、Q8_0 KV。启用 KVMem 时使用 budget 384、generation reserve 128、block 32；全部关闭 MTP 和 trace。`CUDA0` 是 RTX 5060 Ti 16 GiB，`CUDA1` 是 RTX 5050 Laptop 8 GiB。双卡均按 4:1 分配。每种配置启动新服务，预热一次，再用 `cache_reset=true` 测三次；全部为 `prompt_n=314`、`cache_n=0`、`predicted_n=64`。数字取服务端 API 的预填充/生成吞吐中位数，括号为三次范围；时间是 `314 / 中位数吞吐`，不包含模型装载和 HTTP 客户端耗时。

| 配置 | 预填充 tok/s | 314 token 预填充 ms | 生成 tok/s |
|---|---:|---:|---:|
| 5060 Ti 单卡，原生 KV | 450.73 (434.97–453.32) | 696.7 | 22.90 |
| 5060 Ti 单卡，KVMem | 407.45 (399.91–422.27) | 770.6 | 22.48 |
| 双卡 layer 4:1，原生 KV | 390.00 (370.59–397.46) | 805.1 | 19.21 |
| 双卡 layer 4:1，KVMem | 322.83 (309.33–334.16) | 972.6 | 19.06 |
| 双卡 tensor 4:1，原生 KV | 299.06 (289.01–300.47) | 1050.0 | 23.31 |
| 双卡 tensor 4:1，KVMem | 258.64 (241.82–260.84) | 1214.0 | 22.96 |

相同 KVMem 配置下，tensor 4:1 比单卡预填充少 **36.5% tok/s**，处理同一 314-token 提示多 **443 ms**。这不表示 443 ms 可归给一个模块；对照显示几种开销同时存在：

- 关闭 KVMem 后，layer 双卡比单卡多 **109 ms**。层从 5060 Ti 移到较慢的 5050，且层边界要传递激活；不能单靠本测试区分这两项。
- 关闭 KVMem 后，tensor 双卡比 layer 双卡再多 **245 ms**。在相同两卡和分配比下，这主要反映 tensor 分片每层跨卡归约/同步和较慢分片拖住整体的代价；精确的算子/通信占比尚未测量。
- 开启 KVMem 相对原生 KV，多 **74 ms（单卡）/ 168 ms（layer）/ 164 ms（tensor）**。双卡 tensor 的 KVMem 增量比单卡多约 **90 ms**。按这些中位数作描述性拆分，单卡 KVMem 到双卡 tensor KVMem 的 443 ms 差距约等于 **109 ms（原生 layer 相对单卡）+ 245 ms（原生 tensor 相对 layer）+ 90 ms（KVMem 的额外多卡增量）**。这是配置间差分，不是逐算子计时，不能据此认定精确的根因占比。

代码路径支持上述解释。Windows 默认采用内部 CUDA AllReduce；`backends/llamacpp/ggml/src/ggml-cuda/allreduce.cu` 明确说明其通过页锁定主机内存交换双卡数据，并为较大的 prefill 归约采用 copy engine 的 D2H/H2D 分块。`backends/llamacpp/ggml/src/ggml-backend-meta.cpp` 的每个子图后都可能触发归约。KVMem 在多卡 `harvest_pending` 中先同步整图，再逐个 tensor 从所属后端读取 KV；单卡则尝试异步 D2D/分批 D2H 路径。这些实现位置说明 **tensor 跨卡归约和多卡 KV 收集是主要候选瓶颈**，但目前没有逐算子 GPU profiler，不能给出两者准确百分比。

另以 `KVMEM_PERF=1` 做一次诊断（不用于吞吐表，因为计数会拖慢运行）：314-token 预填充的适配层 D2H 均为 **143,261,696 bytes = 136.625 MiB**。单卡是 **4 次 D2H + 112 次 D2D**；layer/tensor 双卡均为 **112 次 D2H、0 次 D2D**。因此双卡 KVMem 的问题并非多复制了一份 KV 字节数，而是同样数据量改为更零碎的跨设备收集，且多卡路径显式等待计算图完成。计数仅涵盖 KVMem 适配层的拷贝，不含 Meta AllReduce 内部经主机内存的流量。

单卡同配置开启 MTP snapshots 在模型启动阶段 GPU OOM，报错还差 31.63 MiB 的计算 buffer；这组没有可比速度。`nvidia-smi topo -m` 在本机返回 `Failed to run topology matrix`，所以未据此推断 P2P 能力；对当前程序可直接确认的是 Windows 默认 AllReduce 走内部主机内存路径。

原始响应及服务端日志位于 `build-layer/*-warm*.responses.json`、`build-layer/*-perf-diagnostic.err.log`；复现脚本是 `build-layer/bench-tensor-server-smoke.ps1`。这些文件在本地 build 目录，没有纳入版本控制。样本量每种仅三次，且预热后仍可见逐次加速；小上下文结论应视为当前机器和构建下的近似值。下一步若优化，先为 Meta AllReduce 和 KVMem 多卡 harvest 加独立计时，再试合批/异步跨设备 KV 收集，避免把性能变化误判成分片比例效果。
