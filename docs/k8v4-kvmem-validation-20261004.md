# K8V4 KVMem 本地源码验收（2026-10-04）

结果：26/26 次真实模型实验及 8/8 项基础检查通过。真实模型进程内累计耗时 19.5 分钟，
不包括源码接入、编译、基础检查和整理。没有真实模型失败重跑；一项 NVFP4 实验专门为磁盘格式隔离生成冷记录。
因此真实实验为 25 个 K8V4 进程加 1 个 NVFP4 种子进程，服务只冷启动一次。

## 环境与实现

- 分支：`feat/multi-backend-framework`；ninfer base 固定为 `6e9189c1089f6968e29fe2e1c51b4596e2346eed`。
- GPU：RTX 5060 Ti 16GB，UUID `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`；驱动 610.62，CUDA 13.2，sm_120a，Release，VS2022，8 个编译任务。
- 制品：`Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`，12,790,639,360 字节；使用本地转换制品，无再次转换。
- ngram 与 lookup 全部为 0。除输入回退和容量服务显式使用 FP16 GDN 外，Engine 用例保留默认 FP32 GDN。
- 复用原生 `Fp8KeyNvfp4Value` 与 CUDA 编解码/Attention；生产改动集中于 Engine/CLI 格式准入、
  `ninfer.k8v4-fp8-nvfp4.page.v1` 布局身份及 Windows 启动器能力。
- D256 每 token/head：K FP8 codes 256 B + FP16 row scale 2 B，V NVFP4 codes 128 B + raw E4M3 group-16 scales 16 B，
  共 402 B；K、V 均沿用原生归一化 Hadamard。页搬运按各 plane 的真实 extent/dtype 原样复制，不重新量化。
  pre-RoPE BF16 mean-K/Q、提交与回滚流程保持原样。

## 26 次真实实验

一次表示一个真实模型进程，进程内包含多个请求/断言；数值子用例不计入这 26 次。
1–12、13、22–23 的 B/R=384/128；并发每路 B/R=384/128；图像 B/R=768/128；
磁盘 B/R=384/128；质量 B/R=2048/512；最后一项 B/R=32768/16384。

| 次数 | 内容 | 结果 | 耗时（秒） |
|---:|---|---|---:|
| 1 | 短请求/复用，MTP0 eager | PASS | 37.1 |
| 2 | 短请求/复用，MTP0 Graph | PASS | 32.2 |
| 3 | 短请求/复用，MTP4 eager | PASS | 25.2 |
| 4 | 短请求/复用，MTP4 Graph | PASS | 25.4 |
| 5 | 短请求/复用，MTP1 Graph | PASS | 26.8 |
| 6 | 短请求/复用，MTP2 Graph | PASS | 25.4 |
| 7 | 短请求/复用，MTP3 Graph | PASS | 24.6 |
| 8 | 预算/生成边界，MTP0 eager | PASS | 19.2 |
| 9 | 预算/生成边界，MTP4 eager | PASS | 17.5 |
| 10 | 预算/生成边界，MTP4 Graph | PASS | 17.5 |
| 11 | 完整 endpoint，MTP4 Graph | PASS | 75.9 |
| 12 | 输入回退/取消恢复，MTP4 Graph，FP16 GDN | PASS | 20.7 |
| 13 | 多会话/淘汰/隔离，MTP4 Graph | PASS | 103.9 |
| 14 | 4 路并发，H512 MiB，MTP4 Graph | PASS | 86.7 |
| 15 | 4 路 Host 压力，H73 MiB，MTP4 Graph | PASS | 140.7 |
| 16 | resident 图片红/蓝/多图/拒绝恢复，MTP4 Graph | PASS | 54.1 |
| 17 | K8V4 冷写入，MTP4 Graph | PASS | 17.2 |
| 18 | K8V4 新进程冷命中，MTP4 Graph | PASS | 8.0 |
| 19 | 冷记录损坏后拒绝/重新计算，MTP4 Graph | PASS | 11.7 |
| 20 | NVFP4 冷记录种子，供下一项格式隔离 | PASS | 11.0 |
| 21 | K8V4 拒绝 NVFP4 冷记录并重新计算 | PASS | 11.1 |
| 22 | 输入回退状态冷写入，FP16 GDN | PASS | 15.6 |
| 23 | 输入回退状态新进程恢复，FP16 GDN | PASS | 16.5 |
| 24 | 8K 早期双口令/复用/同路径种子稳定性 | PASS | 104.3 |
| 25 | 32K 早期双口令与后续追问 | PASS | 65.8 |
| 26 | 单次服务启动 B32768/R16384、越窗/复用/输出1–5/取消恢复 | PASS | 175.9 |

测试入口分别是 `ninfer_p4_history`、`ninfer_memory_sessions`、`ninfer_memory_concurrency`、
`ninfer_memory_vision`、`ninfer_memory_disk`、`ninfer_p4_quality` 与 `ninfer-serve`。
history 命令最后显式传 `0`，避免 endpoint/fallback fixture 默认启用 ngram。
Host 压力使用实际 73 MiB（96 MiB INT8 × 402/528 向下取整），没有把 73 当等价 token 预算。
磁盘恢复在不同进程执行，损坏注入在独立副本完成；格式不匹配保持零命中、产生错误统计并重新计算。

8K 采样检查分别比较同种子同 cold 路径、同 cached 路径；两种路径间的 token 一致性仅记录，
没有要求不同原生分块/执行路径逐 token 相等。质量检查属于合成早期口令检索，不是通用模型质量评测。

## 基础检查

8 项：Engine options、serve options、Windows launcher、独立 payload 布局/字节搬运、MTP graph 路由，
以及 native K8V4 codec、native K8V4 Attention、稀疏分页 Attention。
payload 独立定义三层 Main＋一层 MTP，并覆盖 PageMajor/HeadMajor 和不同 K/V scale 类型。
稀疏检查覆盖完整/不连续/重排页、offset 0/63、width 1–5/13/53/63/64/65 和 CUDA Graph，
使用独立 FP64 softmax oracle、原生表示/转换边界及 NaN 末页 scale 注入；保持既有数值判据。

## 容量服务与显存

- `127.0.0.1:18201`，模型 ID `iq3s-k8v4`，MTP4 Graph，B32768/R16384，H6144 MiB，prefill256，
  C1，logical context131072，FP16 GDN，默认 CUDA 分配，graph allowance72 MiB。
- 越窗请求 55,538 tokens，重复请求 55,538 tokens；
  核对换出/恢复与 history hits 增长、早期 pulsars/galaxies 答案、输出 1–5 tokens、流式断开后的取消与恢复，MTP 实际产生草稿。
- 1 秒采样的物理显存 `memory.used` 峰值 13,372 MiB（卡总量 16,311 MiB）；
  同步样本中 total−used 最小 2,939 MiB，该差值含驱动 reserved，不等于可分配余额。
  全程 5 秒采样的 `memory.free` 最小 2,677 MiB。
  Windows 进程 Shared Usage 峰值 338.0 MiB。原始 occupancy 同时保留显卡 UUID。
- 通过后服务保留在线，PID 36588。16K 是配置的 gen reserve；本轮没有连续生成 16K tokens，
  边界正确性由小 R 的历史用例及实际小输出请求覆盖。

编译第一次链接因旧 Bonsai 进程占用 exe 而失败；记录配置后临时停止该空闲进程，原配置已恢复。
这是 Windows 文件占用问题，没有修改 CUDA 内核或放宽正确性判据。

## 证据与范围

本地完整证据目录：`backend-p0p1-20261002/results/k8v4-integration-20261004/`。
包含 `plan.json`、`runs.json`（命令/UUID/PID/耗时）、逐项日志、请求/响应/props、显存采样，
`tested-source-binaries.json`、`before.json`、`authored.diff` 与 `changes.json`。
生产补丁和 manifest 固定于主仓库；未提交模型、二进制、冷缓存或运行数据。

本轮没有完整组合全排列、性能对比、8GB K8V4 容量验收、64K/128K 质量或发布包重验。
只对这里记录的本地源码、IQ3S 制品、设备和配置做验收结论。
