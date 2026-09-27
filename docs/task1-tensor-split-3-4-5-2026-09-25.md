# Tensor 3:1 / 4:1 / 5:1 任务一对照（2026-09-25）

沿用 `docs/tensor-parallel-design.md` 的任务一口径：RTX 5060 Ti 16 GiB（CUDA0）+ RTX 5050 Laptop 8 GiB（CUDA1）、Qwen3.8-27B-UD-Q4_K_M、`llama-kvmem-server.exe`、`-c 1024 -b 128 -ub 128`、Q8_0 KV、KVMem budget 384 + reserve 128、block 32、同一文本经过 chat 模板为 314 prompt token、greedy 输出 64 token、未加载视觉头。MTP 使用嵌入式 nextn、宽度 2、snapshots。三种比例均使用同一个 2026-09-25 11:00:59 构建。每个比例和 MTP 状态各启动一个新服务，先跑一次预热，再以 `cache_reset=true` 重复测量三次；每次均核实 `prompt_n=314`、`cache_n=0`、`predicted_n=64`。正式性能运行关闭 `KVMEM_TRACE`。

下表为三次测量的**中位数**，括号内是范围；tok/s 取 API `timings.prompt_per_second` 与 `timings.predicted_per_second`，不是客户端墙钟吞吐。

| Tensor 比例 | 无 MTP 预填充 | 无 MTP 生成 | MTP 预填充 | MTP 生成 | MTP 总计时中位数 |
|---|---:|---:|---:|---:|---:|
| 3:1 | **268.25** (257.40–270.16) | **23.51** (23.24–23.54) | 234.03 (233.37–250.76) | 36.35 (36.24–36.40) | 3,100 ms |
| 4:1 | 258.64 (241.82–260.84) | 22.96 (22.44–22.96) | **252.09** (250.64–253.02) | 40.76 (40.33–41.51) | **2,811 ms** |
| 5:1 | 255.85 (245.83–260.43) | 22.36 (22.28–22.37) | 247.74 (242.76–256.99) | **41.32** (40.96–41.50) | 2,816 ms |

MTP 测量期间整卡最低剩余显存（5060 Ti / 5050）：3:1 为 **4,059 / 3,731 MiB**，4:1 为 **3,155 / 4,635 MiB**，5:1 为 **2,835 / 4,953 MiB**。三组剩余显存之和几乎相同；3:1 相对 5:1 将约 1.2 GiB 占用移到 5050，4:1 约移 0.3 GiB。之前的 5:1 冷启动两次 MTP 结果是预填充 235.94–242.16、生成 41.70–42.13 tok/s；本次为了跨比例比较而采用预热重复测量，不能把冷启动与预热数字直接当成比例效果。

MTP 下 4:1 与 5:1 的完整 314+64-token 请求几乎等时：4:1 预填充略快、5:1 生成略快，差异与单次预填充波动接近。3:1 无 MTP 时基础模型最快，但 MTP 生成比 5:1 慢约 12%，使整次请求慢约 10%。额外的单次 `--kvmem-trace` 诊断显示，3:1 草稿接受率 38/51=74.5%、验证 26 次；4:1 和 5:1 均为 40/48=83.3%、验证 24 次。这支持接受率是 3:1 MTP 变慢的一个原因，但 trace 显著拖慢运行，不能用 trace 的速度数值作性能比较。各比例输出文本存在差异，即使均为 64 token，也不能将速度差异完全归因于设备分片。

可复现脚本为 `build-layer/bench-tensor-server-smoke.ps1`，参数示例：

```powershell
& .\build-layer\bench-tensor-server-smoke.ps1 -Mode snapshots -SplitMode tensor -SplitRatio '4,1' -RunId warm -Warmup 1 -Repeats 3
```

`build-layer/tensor-server-{no-mtp|mtp}-q4-task1-split-{3-1|4-1|5-1}-warm.responses.json` 保留全部三次响应和原始 timings；对应 `.err.log` 保存服务端日志。诊断 trace 文件另以 `-trace.err.log` 命名。脚本仅关闭自己启动的 5098 端口服务。
