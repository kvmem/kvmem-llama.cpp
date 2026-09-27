# ROCm README 对齐配置与 5060 Ti 基线对比

以下保留 FangJiangyi 在 PR #33 提供的历史实测记录；版本、设备和配置以正文为准。
当前整合版本的构建和启动方法见 [rocm.md](rocm.md)。

本次复测严格采用 [推荐配置与测试结果](recommended-config-performance.md)
中的 budget / reserve：IQ3 为 `36864 + 16384`，IQ4 为
`32768 + 12288`，并在 Task 1 和 Task 2 中保持不变。测试日期为
2026-09-20，测试代码版本为 `509d5ba`。

测试机器：AMD Radeon RX 7900 XTX 24 GiB、Intel Xeon E5-2603（8 核，
1.80 GHz）、ROCm 7.14。服务使用 `build-rocm/bin/llama-kvmem-server`。
对照数据来自 RTX 5060 Ti 16 GiB、Core Ultra 7 255H 的 README 测量，
因此不是同卡 CUDA/ROCm 后端对比。

## 配置与完成状态

| 模型 | budget | reserve | 主 KV / MTP KV | projector |
|---|---:|---:|---|---|
| IQ3 | 36864 | 16384 | Q8_0 / F16 | Q5_K_MIX / GPU |
| IQ4 | 32768 | 12288 | Q5_0 / F16 | BF16 / CPU，8 线程 |

两个任务都使用相同参数：256K context、batch 512、MTP3、ReplaySSM、
query replay `auto`、query policy `user`。Task 1 使用
`--image-max-tokens 1024`、一次预热和两次正式测量；Task 2 保留
512-token 图片限制并加载 projector，但不发送图片。

| 运行 | 状态 | 最终上下文 | 备注 |
|---|---|---:|---|
| Task 1 IQ3 | 通过 | 两轮代码均为 512 token | 图片识别正确 |
| Task 1 IQ4 | 通过 | 两轮代码均为 512 token | 图片识别正确 |
| Task 2 IQ3 | **长度检查失败** | 261778 / 262144 | prompt 261546，最终生成自然停于 232 token |
| Task 2 IQ4 | 长度检查通过 | **262058 / 262144** | prompt 261546，最终生成 512 token |

Task 2 IQ3 完成了基础请求、32 个工具结果轮次和最终请求，没有 HTTP、缓存、
显存或 swap 错误；benchmark 进程因最终输出没有达到要求的 512 token 而失败。
该输出还生成了再次读取 `final.py` 的字面工具调用，而不是所要求的完整 Python
统计工具，因此在长度和语义上都不是 README 的 262058-token 等价完成。Task 2
IQ4 的“通过”仅表示满足 benchmark 的 512-token 长度断言，不表示语义验收。

## Task 1：约 12K 文本、图片、代码生成

正式两轮复用预热得到的图片 embedding。Prefill 和 aggregate decode 按两轮
共 6 个请求聚合；Image decode 与 Code decode 分别聚合两轮对应请求。

### IQ3

| 指标 | RX 7900 XTX ROCm | RTX 5060 Ti README | 差异 |
|---|---:|---:|---:|
| Prefill — initial computation | 724.21 token/s | 574.49 token/s | +26.06% |
| Prefill — overall | 654.71 token/s | 544.65 token/s | +20.21% |
| First image encode (warmup) | 0.328 s | 0.41 s | -19.91% |
| Aggregate decode | 50.31 token/s | 38.55 token/s | +30.52% |
| Image decode | 43.12 token/s | 38.64 token/s | +11.59% |
| Code decode（每轮 512 token） | 53.71 token/s | 39.88 token/s | +34.67% |
| MTP acceptance | 73.80% | 70.74% | +3.06 pp |
| Runtime host RAM peak | 4372.08 MiB | 4308.21 MiB | +1.48% |
| VRAM peak | 16313 MiB | 15591.10 MiB | +4.63% |
| Minimum free VRAM | 8247 MiB | 460.90 MiB | 总显存不同，不直接比较 |

IQ3 ROCm 使用当前推荐的 Q5_K_MIX GPU projector，而历史 CUDA 基线使用
Q8 projector；图片编码项并非完全相同 projector 的比较。

### IQ4

| 指标 | RX 7900 XTX ROCm | RTX 5060 Ti README | 差异 |
|---|---:|---:|---:|
| Prefill — initial computation | 740.93 token/s | 595.60 token/s | +24.40% |
| Prefill — overall | 670.23 token/s | 505.52 token/s | +32.58% |
| First image encode (warmup) | 99.70 s | 21.79 s | +357.56% |
| Aggregate decode | 43.17 token/s | 44.30 token/s | -2.55% |
| Image decode | 46.23 token/s | 45.83 token/s | +0.87% |
| Code decode（每轮 512 token） | 44.76 token/s | 44.35 token/s | +0.92% |
| MTP acceptance | 61.36% | 80.85% | -19.49 pp |
| Runtime host RAM peak | 5043.06 MiB | 4846.82 MiB | +4.05% |
| VRAM peak | 16212 MiB | 15445.10 MiB | +4.97% |
| Minimum free VRAM | 8348 MiB | 606.90 MiB | 总显存不同，不直接比较 |

IQ4 两边都使用 BF16 CPU projector。本机 Xeon E5-2603 的单核和向量性能
明显低于 Core Ultra 7 255H，因此首次图片编码仍是 CPU 瓶颈。

## Task 2：32 个工具结果轮次

Prefill 覆盖全部 33 个请求和 261545 个新增输入位置。Aggregate tool decode
和 MTP acceptance 聚合 32 个工具轮次；最终代码 decode 单独列出。

### IQ3（最终长度未通过）

| 指标 | RX 7900 XTX ROCm | RTX 5060 Ti README | 差异 |
|---|---:|---:|---:|
| Prefill — initial computation | 384.47 token/s | 437.13 token/s | -12.05% |
| Prefill — overall | 216.81 token/s | 242.06 token/s | -10.43% |
| Aggregate tool decode | 33.77 token/s | 31.74 token/s | +6.39% |
| Final code decode | 39.34 token/s（232 token） | 30.53 token/s（512 token） | 非等长 |
| MTP acceptance | 59.66% | 64.70% | -5.04 pp |
| Runtime host RAM peak | 13067.95 MiB | 13483.52 MiB | -3.08% |
| VRAM peak | 16011 MiB | 15617.10 MiB | +2.52% |
| Minimum free VRAM | 8549 MiB | 434.90 MiB | 总显存不同，不直接比较 |

最终请求的 prompt、7927 个新增输入位置和 query replay 都正确；模型以 stop
结束 232-token 输出。由于生成长度不同，Final code decode 只作为诊断值，
不应解读为与 512-token CUDA 数值严格等价。

### IQ4

| 指标 | RX 7900 XTX ROCm | RTX 5060 Ti README | 差异 |
|---|---:|---:|---:|
| Prefill — initial computation | 419.52 token/s | 463.18 token/s | -9.43% |
| Prefill — overall | 232.00 token/s | 253.41 token/s | -8.45% |
| Aggregate tool decode | 39.31 token/s | 33.31 token/s | +18.02% |
| Final code decode（512 token） | 44.78 token/s | 38.15 token/s | +17.38% |
| MTP acceptance | 60.79% | 67.08% | -6.29 pp |
| Runtime host RAM peak | 10978.13 MiB | 11244.75 MiB | -2.37% |
| VRAM peak | 16164 MiB | 15591.69 MiB | +3.67% |
| Minimum free VRAM | 8396 MiB | 460.31 MiB | 总显存不同，不直接比较 |

四组进程的 swap 峰值均为 0 MiB。RX 7900 XTX 的总显存为 24560 MiB，
因此 Minimum free VRAM 不能与 16 GiB RTX 5060 Ti 的数值直接比较。

## 复测命令

模型路径按本机实际位置替换。IQ4 文件是由
`scripts/quantization/quantize-iq4-mtp.py` 生成的 MTP Q4_0 版本。

```bash
# Task 1, IQ3
python3 scripts/multimodal_canary.py \
  --binary build-rocm/bin/llama-kvmem-server --gpu-api rocm --gpu-index 0 \
  --model /path/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --mmproj /path/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf --device gpu \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q8_0 --draft-kv f16 --budget 36864 --reserve 16384 \
  --ctx 262144 --batch 512 --image-max-tokens 1024 --long-words 12000 \
  --quick --performance --thinking-budget 128 --warmup-runs 1 \
  --performance-runs 2 --startup-timeout 900 \
  --folder logs/rocm-readme-aligned-20260920/task1-iq3

# Task 1, IQ4
python3 scripts/multimodal_canary.py \
  --binary build-rocm/bin/llama-kvmem-server --gpu-api rocm --gpu-index 0 \
  --model /path/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --mmproj /path/mmproj-BF16.gguf --device cpu --threads 8 \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q5_0 --draft-kv f16 --budget 32768 --reserve 12288 \
  --ctx 262144 --batch 512 --image-max-tokens 1024 --long-words 12000 \
  --quick --performance --thinking-budget 128 --warmup-runs 1 \
  --performance-runs 2 --startup-timeout 900 \
  --folder logs/rocm-readme-aligned-20260920/task1-iq4

# Task 2, IQ3
python3 scripts/multimodal_canary.py \
  --binary build-rocm/bin/llama-kvmem-server --gpu-api rocm --gpu-index 0 \
  --model /path/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --mmproj /path/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf --device gpu \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q8_0 --draft-kv f16 --budget 36864 --reserve 16384 \
  --ctx 262144 --batch 512 --image-max-tokens 512 \
  --long-context-benchmark --long-chunk-tokens 8192 --thinking-budget 128 \
  --startup-timeout 900 \
  --folder logs/rocm-readme-aligned-20260920/task2-iq3

# Task 2, IQ4
python3 scripts/multimodal_canary.py \
  --binary build-rocm/bin/llama-kvmem-server --gpu-api rocm --gpu-index 0 \
  --model /path/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --mmproj /path/mmproj-BF16.gguf --device cpu --threads 8 \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q5_0 --draft-kv f16 --budget 32768 --reserve 12288 \
  --ctx 262144 --batch 512 --image-max-tokens 512 \
  --long-context-benchmark --long-chunk-tokens 8192 --thinking-budget 128 \
  --startup-timeout 900 \
  --folder logs/rocm-readme-aligned-20260920/task2-iq4
```

## 日志与聚合

本次本地原始记录位于 `logs/rocm-readme-aligned-20260920/`：

- `task1-iq3/`
- `task1-iq4/`
- `task2-iq3/`
- `task2-iq4/`

每个目录包含实际命令 `argv.json`、全部请求和 trace 的 `summary.json`、聚合
后的 `metrics.json`、`vram.csv`、`rss.csv` 和服务日志。Task 2 IQ3 的进程
退出码为 1，原因是 benchmark 的 512-token 最终输出断言；其 summary 和采样
在断言后仍完整写入。日志由 `/logs/` 忽略规则排除，不随 Git 提交。

```bash
python3 scripts/summarize_canary.py \
  logs/rocm-readme-aligned-20260920/task1-iq3 \
  logs/rocm-readme-aligned-20260920/task1-iq4 \
  logs/rocm-readme-aligned-20260920/task2-iq3 \
  logs/rocm-readme-aligned-20260920/task2-iq4
```
