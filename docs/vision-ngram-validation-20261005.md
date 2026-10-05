# 图片＋MTP＋ngram 定向补测（2026-10-05）

放行固定/自适应 MTP 与 ngram1–63、单请求 INT8/NVFP4 resident/CPU 图片组合。
三个目标组合中两组完整通过，K8V4 一组出现冷启动/复用换行差异，保留限制。
仅移除 KVMem 对 INT8/NVFP4 图片＋ngram 的准入限制，没有修改 CUDA 数学、复制算法或视觉编码器。
仍要求 MTP1–15、C1、R >= max(MTP drafts, ngram drafts) + MTP drafts；
并发 ngram、并发图片、overlay、视频、图片冷快照及图片 fast prefill 继续拒绝。

## 环境与实际执行证据

RTX 5060 Ti `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`，`Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`；
CUDA13.2/sm120a/VS2022 Release，8 路增量编译，沿用既有构建树和内嵌 UI。
B768/R128、context8192、H512MiB、prefill128、图片上限256 merged tokens；
min-match12、lookup0，逐字节 Main/MTP/GDN 搬运检查开启。
表中接受数取冷启动图片复制请求，replay 次数取实际 `KVMEM_QUERY_REPLAY` 日志。

| # | 配置 | 结果 | 请求数 | 冷复制接受 ngram tokens | 实际 query replay 次数 | 秒 |
|---|---|---|---:|---:|---:|---:|
| 1 | `cpu-int8-graph-adaptive4-ng31` | PASS | 21 | 66 | 6 | 246.4 |
| 2 | `resident-nvfp4-eager-fixed15-ng63` | PASS | 15 | 63 | 4 | 62.4 |
| 3 | `cpu-k8v4-graph-adaptive15-ng63` | 暂不放行，冷启动/复用换行差异 | 2 | 63 | 1 | 51.4 |

成功组合累计 308.8 秒（5.1 分钟），不含构建、诊断重试及服务恢复。
CPU/INT8 默认组合使用完整图片用例：短图、旧图、相同占位符替换图片、复用、取消恢复、
最后用户消息图片 replay、多图和图片超 budget 拒绝。NVFP4组完成query专项边界；K8V4组在复制复用检查处停止。
两组完整通过的配置均执行带图片的复制任务，检查正向宽草稿接受、冷启动/复用逐 token 一致、
取消恢复、输出上限1/2/65、固定 seed 采样复用一致、编辑前缀失效，以及无活动页泄漏和 H 上限。
Engine/serve 参数检查覆盖五种 KV、两种视觉 residency、固定/自适应和 NG15/31/63，
仅放行 INT8/NVFP4 图片ngram，其他三种 KV 的图片ngram 保持拒绝；
Windows 启动器参数映射检查通过。其他 KV 数学和视觉基本能力复用既有验收证据。

## 初次诊断与边界

第一次 CPU/INT8 用例关闭模型默认结束符、强制生成192 tokens，冷启动与复用在正常结束后分歧。
恢复正常结束规则后，两次都在91 tokens结束，完整有效输出逐 token 一致；采样与取消恢复也一致。
保留初次日志，未将正常结束后的强制续写用于放行，也未据此修改原版行为。
这不承诺不同推测窗宽、任意随机采样或忽略结束符的继续生成具有逐 token 等价性。

K8V4＋CPU＋Graph＋自适应15＋NG63 两次均输出91 tokens、接受63个ngram tokens，
颜色和复制正文一致，但 `Red` 后首个换行不同；未放宽逐token断言或改动算法。
Main/MTP/GDN 搬运逐字节检查通过；两次选页及自适应窗宽记录一致。
补两个对照：KVMem同配置NG0（53.4秒）和
原生非KVMem同配置NG63（57.2秒）均逐token一致。
因此尚不能确认是继承缺陷、适配回归或不同prefill形状引起的数值差异。
K8V4图片ngram继续拒绝，BF16/RK8V4图片ngram未做真实组合验收也继续拒绝。

本次是组合功能验收，没有性能对比、200k全默认预算或连续16k生成验收。
30/40系硬件未补测。旧预编译 ZIP 未更新，新组合需要本次重新编译的服务端。
默认源码启动脚本恢复 ngram31，保留 CPU视觉、自适应MTP4、双INT8和原36k/16k/200k预算。

原始命令、配置、逐次日志及服务恢复记录位于实验目录
`backend-p0p1-20261002/results/vision-ngram-20261005/`。
