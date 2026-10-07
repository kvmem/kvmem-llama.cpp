# 自适应 MTP 组合定向补测（2026-10-05）

本报告记录 2026-10-05 的自适应 MTP 定向范围，包括当时的快速 prefill、并发和冷快照限制。当前源码边界见 [多后端运行](multi-backend.md)。

10 个目标组合全部通过；真实模型启动 10 次，失败尝试 0 次。
真实模型累计 515.4 秒（8.6 分钟），不含构建、等待 GPU 和分析。
每个模型进程执行多个已有请求/断言，启动次数不是请求次数。

## 放行范围

自适应 MTP1–15 沿用固定 MTP 的通用预算及功能限制：文本 C1–4，resident/CPU 图片 C1；
可与单文本请求的 ngram（1–63，min-match4–64）和冷快照（device profile off）组合。
快速 prefill 仍只允许 INT8/RK8V4 文本，禁止图片/冷快照。
lookup-ngram、MTP 独立 attention window、并发图片、overlay、视频及其他原有拒绝条件保持。
本次修改准入检查、帮助、启动器和已有测试的参数入口，没有修改 CUDA 数学或控制器算法。
真实测试配置见下表，参数检查覆盖五种 KV 格式；不是全部格式/宽度/开关的全排列验收。
NG31/63 仅用于 ngram 专项，其余实验 NG0；所有实验 lookup0。

## 环境与断言

RTX 5060 Ti `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`；模型 `Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`，CUDA13.2/sm120a/VS2022 Release，8 路编译。
history B384/R128/H2GiB；concurrency 每路B384/R128、H512MiB；vision B768/R128/H512MiB；
disk B384/R128/H512MiB，1GiB磁盘限额；prefill128、device profile off、默认CUDA内存策略。
保留逐字节 Main/MTP/状态搬运检查、Host预算及活动页退休断言。

| # | 配置（自适应上限15） | 结果 | 秒 | 观察到的窗宽 |
|---|---|---|---:|---|
| 1 | `adaptive15-fast-int8-graph` | PASS | 29.7 | 3,4 |
| 2 | `adaptive15-fast-rk8-eager` | PASS | 21.2 | 3,4 |
| 3 | `adaptive15-k8-c2-eager` | PASS | 65.4 | 1,3,4,5,6 |
| 4 | `adaptive15-fast-rk8-c4-graph` | PASS | 118.8 | 1,3,4,5,6,7 |
| 5 | `adaptive15-k8-ngram31-graph` | PASS | 50.6 | 1,3,4,6 |
| 6 | `adaptive15-rk8-ngram63-eager` | PASS | 45.5 | 1,3,4 |
| 7 | `adaptive15-nv-resident-graph-query` | PASS | 38.8 | 3,4 |
| 8 | `adaptive15-k8-cpu-graph-query` | PASS | 111.6 | 3,4 |
| 9 | `adaptive15-k8-disk-cold-graph` | PASS | 19.0 | 3,4 |
| 10 | `adaptive15-k8-disk-hit-graph` | PASS | 14.7 | 3,4 |

history 检查越窗、复用、取消；多路检查会话隔离、单路取消后其他路继续、再次出版和旧历史退休。
ngram 检查复制/代码任务、greedy逐token相等、限长1/2/3/17/65、固定seed采样、编辑前缀和取消恢复。
图片仅补 query slice：旧图片重放、重复复用、多图片、取消/恢复；既有普通视觉及已修复CPU边界证据复用。
磁盘 cold/hit 分别启动进程，验证访问码、命中/写入、Host上限和无活动页泄漏。
各请求检查 adaptive observation，并打印实际窗口分布/切换次数。
上限15不意味着强制每轮使用15个草稿；表中记录实际观察到的窗宽。2/4路用例分别观测到对应活动请求数。

Engine、serve options、原生控制器、Windows launcher 四项检查通过。
MTP Graph profiles、INT8/RK8V4 fast prefill独立数值 oracle 复用上一轮通过证据，未重复运行或放宽容差。
没有进行性能比较、连续16K生成或大预算新增验收；30/40系只编译/打包检查，硬件推理待相应GPU验收。

## 本地证据

`backend-p0p1-20261002/results/adaptive-combinations-20261004/` 保存计划、逐次命令、GPU、耗时、日志、
修改前文件及 authored diff。三代构建、嵌入UI、依赖检查和服务恢复单独记录在同目录。
本文及版本化集成补丁保存可随源码提交的结论；模型、二进制和原始日志不加入源码补丁。
