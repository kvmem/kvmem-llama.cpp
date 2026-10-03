# MTP KV 控制实验（2026-09-13）

RTX 5060 Ti 上，主模型 Q5 KV 固定，仅切换 MTP KV。每种类型运行两次。本次代码请求中，Q5 的 decode 均值比 F16 低 13.1%，整卡采样峰值反而高 42 MiB。实验后按用户要求，将 IQ4 默认启动配置设为主模型 Q5 KV、MTP F16 KV。

## 固定配置与测量方法

- 模型：`Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf`；文件名中的 Q4_0 是 MTP 权重类型。
- GPU UUID：`GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`（RTX 5060 Ti）。
- context 262144、batch 512、GPU layers 99、retrieval budget 32000、generation reserve 12000（按块对齐为 11904）、block 128；实际 KV pool 43904 cells。
- 主模型 K/V 均为 Q5_0；MTP draft length 2；thinking 开启，reasoning budget 4096，最大生成 16000。
- 复用上一次代码测试的原始请求字节，temperature 0，61159 prompt tokens；每次启动新服务，缓存命中均为 0。
- 顺序：F16 → Q8 → Q5 → Q5 → Q8 → F16。每次确认前一个进程释放显存，再启动下一个进程。
- prefill 使用 `KVMEM_CHAT_PREFILL` 墙钟时间，decode 使用 `KVMEM_GEN_WALL`；生成 token 数包含思考内容。
- 整卡显存来自 NVML v2 used，排除单列的 259 MiB driver reserved。采样循环每次查询后等待 50 ms，实际间隔有抖动，六次测试最长采样间隔为 4.21 秒。因此下列值是采样峰值，不能排除更短的瞬时尖峰；所列峰值均持续到请求结束，且在重复实验中一致。

## 结果

以下速度和接受率为两次运行的算术均值；同一类型的两次显存峰值相同。

| MTP KV | Prefill tok/s | Decode tok/s | Decode 两次范围 | MTP 接受率 | MTP KV MiB | 整卡采样峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 515.09 | 35.16 | 34.93–35.39 | 92.9% | 171.50 | 15645.10 |
| Q8_0 | 509.87 | 31.12 | 30.63–31.60 | 75.9% | 91.11 | 15719.10 |
| Q5_0 | 504.12 | 30.57 | 30.33–30.81 | 75.9% | 58.95 | 15687.10 |

Q5 相对 F16 的 decode 变化：`(30.57 / 35.16 - 1) × 100% = -13.05%`。Q8 比 Q5 均值高约 1.8%，但两次运行范围重叠，样本不足以判断稳定优势。Prefill 差距较小，也不宜据两次运行推广。

| 顺序 | MTP KV | Prefill tok/s | Decode tok/s | 生成 tokens |
|---|---|---:|---:|---:|
| 1 | F16 | 508.58 | 34.93 | 4649 |
| 2 | Q8_0 | 509.79 | 30.63 | 2096 |
| 3 | Q5_0 | 507.22 | 30.33 | 2096 |
| 4 | Q5_0 | 501.01 | 30.81 | 2096 |
| 5 | Q8_0 | 509.94 | 31.60 | 2096 |
| 6 | F16 | 521.60 | 35.39 | 4649 |

## 解释与边界

按 drafted / 2 估算投机轮数，F16 平均每轮生成约 2.86 tokens，Q8/Q5 为 2.52；对应整轮墙钟时间约为 81.27、80.89、82.32 ms。观察到的吞吐差异主要体现为接受率和每轮产出变化，不能据此认定 Q5 attention kernel 本身慢了 13%。各次 decode 的平均 SM 时钟约 1767–1769 MHz、温度约 67°C，未见明显降频差异。

不同 KV 类型产生了不同输出。F16 两次输出及思考内容分别完全一致；Q8/Q5 四次输出及思考内容也分别完全一致。F16 每次生成 4649 tokens，Q8/Q5 为 2096。客户端平均总耗时 F16 为 251.21 秒、Q8 为 187.53 秒、Q5 为 190.12 秒。因此，F16 每秒生成更多 tokens，但完成这次请求所需时间更长。这不是固定输出序列的 kernel 微基准，也不能推导普遍的质量优势。

两个不同的代码输出都未通过 `c++ -std=c++17 -pthread -fsyntax-only`，均误把 key 用作 `list::splice` 的迭代器参数。原始输出保留，未人为修正；本次结果用于性能对比，不能当作编程正确率评测。

F16 的 KV 本体比 Q5 大 112.55 MiB，但整卡采样峰值低 42 MiB。源码中的一个可能原因是额外计算空间：[`fattn.cu`](../backends/llamacpp/ggml/src/ggml-cuda/fattn.cu) 的 `ggml_cuda_flash_attn_ext_get_alloc_size` 为 TILE/MMA_F16 路径请求 F16 K/V；[`fattn-common.cuh`](../backends/llamacpp/ggml/src/ggml-cuda/fattn-common.cuh) 的 `ggml_cuda_flash_attn_ext_get_f16_extra_data` 在源类型不是 F16 时分配转换空间。量化 KV 本体节省不等于整卡峰值下降。本次没有逐项追踪实际分配，不能把全部差额精确归因于该空间。

## 代码与复现记录

新增 CLI/server 参数 `--spec-kv-dtype`，独立设置 MTP K/V；省略时继承主模型类型。默认 IQ4 脚本现在显式使用 `--kv-dtype q5_0 --spec-kv-dtype f16`，其余配置保持上述实验配置。

server、CLI、MTP KV 测试已构建通过。GPU 字节保存/恢复测试覆盖 Q8/Q5/Q4/F16 继承模式，以及主模型 Q5 + MTP F16/Q8/Q5 三种组合，七组通过；参数帮助和非法类型处理已检查。

实验脚本为 [`scripts/mtp_kv_ab.py`](../scripts/mtp_kv_ab.py)。运行时需要默认 IQ4 服务已启动且 5060 Ti 可独占；脚本会临时停止 18200 服务，在 18201 测试，结束后调用默认启动脚本恢复。可显式传入已保存请求：

```bash
python3 scripts/mtp_kv_ab.py --rounds 2 \
  --request logs/mtp_kv_ab_20260913_195610/request.json
```

原始数据目录：[`logs/mtp_kv_ab_20260913_195610`](../logs/mtp_kv_ab_20260913_195610/)。`summary.json` 包含原始六次结果和可执行文件 SHA256；`analysis.json` 包含汇总、请求/配置/输出一致性检查；每轮目录保留服务日志、SSE、代码、思考内容及显存 CSV。`benchmark.py` 是运行时脚本快照，`mixed-kv-byte-test.log` 和 `build.log` 保存验证日志。请求 SHA256 为 `2c3a1bc21901599242ee640505cd2eac98637fc06c6c747b627a6c6c6d847c8c`。
