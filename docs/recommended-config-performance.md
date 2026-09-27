# 推荐配置与测试结果

测试机器：RTX 5060 Ti 16 GiB、Intel Core Ultra 7 255H、物理 RAM 32 GiB；WSL2 Ubuntu 22.04.5，Linux 可见 16 个逻辑 CPU、约 19.53 GiB RAM。测试结果更新于 2026-09-15。

## 推荐配置

| 配置 | IQ3 | IQ4 |
|---|---|---|
| 启动脚本 | [start-iq3.sh](../scripts/start-iq3.sh) | [start-iq4.sh](../scripts/start-iq4.sh) |
| 主权重 | IQ3_S，GSQ-RCO | IQ4_XS，MTP 权重 Q4_0 |
| 主模型设备 | RTX 5060 Ti | RTX 5060 Ti |
| 视觉头 / 设备 | Q8_0 / GPU | BF16 / CPU |
| 主 KV / MTP KV | Q8_0 / F16 | Q5_0 / F16 |
| MTP 草稿长度 / 状态保存 | 3 / ReplaySSM | 3 / ReplaySSM |
| 检索预算 | 36864（36 × 1024） | 32768（32 × 1024） |
| 生成预留 / 默认最大输出 | 16384（16 × 1024） | 12288（12 × 1024） |
| context / batch | 262144 / 512 | 262144 / 512 |
| 脚本默认图片 token 上限 | 512 | 512 |
| query replay / policy | auto / user | auto / user |
| 正式脚本默认思考预算 | 4096 | 4096 |

两个启动脚本默认传入 `--spec-draft-n-max 3 --kvmem-mtp-state replay`。可用 `SPEC_DRAFT_N_MAX` 和 `KVMEM_MTP_STATE` 覆盖。IQ4 使用 [llama-quantize 配方](../scripts/quantization/quantize-iq4-mtp.py)生成的 MTP Q4_0 权重；下载与量化命令见 [README](../README.md)。IQ3 使用发布的 ISTA MTP 模型。

## 任务一：图文测试，MTP3 + ReplaySSM

约 12K 文本背景 → 一张 896×896 三色图形 PNG → 根据图片生成 HTML/SVG。全程 thinking，思考预算 128 token，每请求最多生成 512 token；temperature=1、top_p=.95、top_k=20、min_p=0、presence/frequency penalty=0、repetition penalty=1、seed=42。

每组先预热一轮，再正式测两轮。本次测试图片上限为 1024（区别于启动脚本默认 512），图片对应 784 行视觉输入。正式两轮复用预热留下的图片嵌入缓存，因此图片编码单独列首次预热时的耗时，不混入正式两轮的速度统计。

| 指标 | IQ3 | IQ4 |
|---|---:|---:|
| 全任务聚合 prefill，首遍 | **574.49 token/s** | **595.60 token/s** |
| 全任务聚合 prefill，有效 | **544.65 token/s** | **505.52 token/s** |
| 首次图片编码，预热轮 | **0.41 秒（GPU）** | **21.79 秒（CPU）** |
| 全任务聚合 decode，含思考 | **38.55 token/s** | **44.30 token/s** |
| 图片回答 decode，含思考 | 38.64 token/s | 45.83 token/s |
| 图片回答输出量，每轮含思考 | 94 token | 64 token |
| 写代码 decode，含思考 | **39.88 token/s** | **44.35 token/s** |
| MTP 草稿接受率 | 70.74% | 80.85% |
| 整卡采样峰值显存 | **15591.10 MiB** | **15445.10 MiB** |
| 峰值时可用显存 | **460.90 MiB** | **606.90 MiB** |
| 运行阶段进程 RAM 峰值（RSS，不含加载） | **4308.21 MiB** | **4846.82 MiB** |

prefill 按正式两轮共 6 个请求聚合：新增输入总量除以对应总耗时，两组均为 25834 个新增输入位置，不计缓存历史或前轮生成量。首遍只计模型首次处理新输入；有效速度还包含缓存管理等前置开销。任务一没有历史重算，正式两轮没有再次编码图片。

decode 同样按总生成 token / 总生成耗时聚合，包含思考及 MTP 验证。资源峰值覆盖预热和正式测量阶段；RAM 取进程 VmRSS，排除模型加载，显存取 NVML 整卡采样值。两组均正确识别三种颜色和形状，代码续接追加 46 行文本；代码均达到 512 token 上限，未检查完整页面功能。

### 为什么默认 MTP3

| 设置 | IQ3 聚合 decode | IQ3 峰值显存 | IQ4 聚合 decode | IQ4 峰值显存 |
|---|---:|---:|---:|---:|
| MTP3 | **38.55 token/s** | 15591.10 MiB | **44.30 token/s** | 15445.10 MiB |
| MTP4 | 37.82 token/s | 15595.10 MiB | 43.04 token/s | 15449.10 MiB |
| MTP5 | 34.09 token/s | 15597.10 MiB | 未测 | 未测 |

MTP3 在两种模型的这组图文任务中均最快；IQ3 代码阶段 MTP3/4 接近，IQ4 两轮整体均为 MTP3 更快。更长草稿的显存增量很小，但接受率下降会抵消速度收益。该选择依据当前任务输入，其他任务仍可覆盖 MTP 长度。

## 任务二：256K 多轮工具测试，MTP3 + ReplaySSM

固定工具输入分 32 轮累积，每轮约 8192 个新 token，最后输入 `final.py` 并生成代码。加上基础请求共 33 个请求；两组最终 prompt 均为 261546 token，最终生成 512 token，总计 **262058 / 262144 token**。两组收到相同工具历史，视觉头保持加载但不输入图片，图片上限沿用启动配置的 512。全程 thinking，思考预算 128，每请求最多输出 512 token。

| 指标 | IQ3 | IQ4 |
|---|---:|---:|
| 全任务聚合 prefill，首遍 | **437.13 token/s** | **463.18 token/s** |
| 全任务聚合 prefill，有效 | **242.06 token/s** | **253.41 token/s** |
| 工具轮次聚合 decode，含思考 | **31.74 token/s** | **33.31 token/s** |
| 最终代码 decode，512 token | 30.53 token/s | 38.15 token/s |
| 工具轮次 MTP 草稿接受率 | 64.70% | 67.08% |
| 整卡采样峰值显存 | **15617.10 MiB** | **15591.69 MiB** |
| 峰值时可用显存 | **434.90 MiB** | **460.31 MiB** |
| 运行阶段进程 RAM 峰值（RSS，不含加载） | **13483.52 MiB** | **11244.75 MiB** |

prefill 覆盖全部 33 个请求，以 261545 个新增输入位置除以对应总耗时；有效速度包含历史重算及缓存管理。decode 和接受率聚合 32 个工具轮次，不含基础请求。每种配置完整测一轮；全部请求成功，后续轮次均保留前缀缓存命中。

IQ3 首次运行在第 10 个工具请求发生 CUDA `unknown error`，未计入本表。CUDA 复查通过后，以完全相同配置重新完成 256K；表中 IQ3 为完整重试结果，首次失败日志保留。IQ4 一次完整通过。当前 IQ4 使用原生 llama-quantize MTP 权重。

### 复测任务二

```bash
python3 scripts/multimodal_canary.py \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q8_0 --draft-kv f16 --image-max-tokens 512 \
  --budget 36864 --reserve 16384 --ctx 262144 --batch 512 \
  --long-context-benchmark --long-chunk-tokens 8192 --thinking-budget 128 \
  --startup-timeout 900 --folder logs/task2-iq3

python3 scripts/multimodal_canary.py \
  --model models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --mmproj models/unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf --device cpu \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q5_0 --draft-kv f16 --image-max-tokens 512 \
  --budget 32768 --reserve 12288 --ctx 262144 --batch 512 \
  --long-context-benchmark --long-chunk-tokens 8192 --thinking-budget 128 \
  --startup-timeout 900 --folder logs/task2-iq4

python3 scripts/summarize_canary.py logs/task2-iq3 logs/task2-iq4
```

本地原始记录：`logs/memory-optimization-20260915/iq3-replay-mtp3-task2-retry/` 和 `iq4-replay-mtp3-task2/`。两组使用相同的 `replay-mtp5-bin` 构建，服务二进制 SHA-256：`e813390e7405fbd20033125ff61da7b543d5c0d1c4ba88df3daaee8d5184066d`。

## 复测任务一

脚本支持 NVML 和 ROCm 显存采样；可用 `--gpu-api auto|nvml|rocm` 与
`--gpu-index` 选择后端和设备。运行前应确保该 GPU 和测试端口 18201 可用；
每组完成后脚本会关闭自己启动的服务。ROCm 的 README 对齐配置和复测结果见
[ROCm README 对齐配置与 5060 Ti 基线对比](rocm-recommended-config-performance.md)。

```bash
python3 scripts/multimodal_canary.py \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q8_0 --draft-kv f16 \
  --image-max-tokens 1024 --budget 36864 --reserve 16384 --ctx 262144 --batch 512 \
  --long-words 12000 --quick --performance --thinking-budget 128 \
  --warmup-runs 1 --performance-runs 2 --startup-timeout 900 \
  --folder logs/task1-iq3

python3 scripts/multimodal_canary.py \
  --model models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --mmproj models/unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf --device cpu \
  --query-replay auto --query-policy user --mtp-state replay --mtp 3 \
  --kv q5_0 --draft-kv f16 \
  --image-max-tokens 1024 --budget 32768 --reserve 12288 --ctx 262144 --batch 512 \
  --long-words 12000 --quick --performance --thinking-budget 128 \
  --warmup-runs 1 --performance-runs 2 --startup-timeout 900 \
  --folder logs/task1-iq4

python3 scripts/summarize_canary.py logs/task1-iq3 logs/task1-iq4
```

输入、响应、逐请求 trace、NVML/RSS 采样和汇总保存在指定目录。汇总脚本只读取已有记录，不启动模型。

本地原始记录（不随仓库分发）：`logs/memory-optimization-20260915/iq3-replay-mtp3-task1/`、`iq4-replay-mtp3-task1/`，同目录的 `iq3-replay-mtp-comparison.md` 和 `iq4-replay-mtp-comparison.md` 保存 MTP 对照明细。IQ3 使用归档 `replay-bin`，IQ4 使用增加 MTP5 容量支持的 `replay-mtp5-bin`，均未启用 Q/mean-K GPU 聚合优化。
