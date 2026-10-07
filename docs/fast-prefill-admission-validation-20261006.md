# 快速 prefill 准入补测（2026-10-06）

用户要求先测试能否简单放行，复杂修改另行确认。本轮只扩展 KVMem Engine、
服务参数和 Windows 启动器的快速 prefill 准入清单，复用已有原生 prompt
Attention 内核，没有修改生产量化算法、Attention 算法或调度架构。
本报告的 GPU 验收范围是单张物理 GPU。多卡 layer pipeline 的验收见
[多卡验收](ninfer-multigpu-validation-20261006.md)，不能用本报告推断多卡结果。

## 准入范围

`int8`、`rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8` 可组合快速 prefill、文本或
resident/CPU 图片、C1–8、ordinary 或固定/自适应 MTP1–15、ngram。
ngram 非零要求 MTP；C2–8 将有效配置的 16–63 自动降到 15，C1 保留原宽度。
快速 prefill 默认关闭，启动器用 `-FastPrefill`，原生服务用
`--fast-prefill-kernel` 显式开启。

此开关只影响符合原生路由条件的 prompt Attention。KVMem 的并行 prefill
及冷快照仍拒绝，CPU 图片编码仍串行。`bf16`、`fp8`、`nvfp4`、`k8v4` 在
KVMem 下不准入这个快速内核开关；它们各自的普通路径不受本轮影响。

## 环境和数值检查

- 固定基线 `6e9189c1089f6968e29fe2e1c51b4596e2346eed`，既有 Release 构建目录，
  CUDA 13.2、sm_120a、MSVC 14.44.35207；编译采用 8 个并行任务。
- 数值和参数检查使用空闲的 RTX 5050 Laptop，UUID
  `GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29`。
- 真实模型检查使用能容纳模型的 RTX 5060 Ti，UUID
  `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`，device profile 为 off。
- 模型 `converted-models/Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`；图片为
  `p9-inputs/red.png` 和 `blue.png`，512×512、每图实际 256 merged tokens，
  配置上限 1024。

五种格式各通过 36 项快速 Attention 数值检查，共 180 项：D256/Q24/KV4，
完整、稀疏和乱序逻辑页，页偏移 0/63，T65/127/128/129，eager 和 CUDA Graph。
夹具显式断言路由为 Prompt 且原生 fast selector 返回 true，防止只通过参数检查。
输出继续对比独立 FP64 represented-value masked-attention oracle，沿用既有
数值判据。缓存写入、保护区及 E8 fused/standalone append 精确字节检查通过。

## 真实模型

新增三种 RK 格式各通过 18 次 Graph 文本历史请求，共 54 次，开启 fast，
context16384、B384/R128、prefill128、H2GiB、逐字节搬运验证。RK4V4、RK4V4-E8
使用固定 MTP3；RK2V4-E8 使用自适应 MTP3 和 ngram31。覆盖冷输入、重复恢复、
下一轮/重算对照、关闭复用、编辑前缀、取消 decode/prefill 及恢复、修改 query、
跨 reserve 的 129-token 输出、截断为 1–5 tokens 后的恢复。相同 checkpoint 的
greedy token 及取消后的恢复 token 精确一致，Main/MTP 页与 Host 预算检查通过。

INT8/resident/C1/eager/ordinary 的快速图片身份回归通过，日志记录 11 次完成
请求，另检查预期的图片 token 预算拒绝及之后的恢复。覆盖重复图片、换图失效、
编辑文本失效、取消恢复、query 图片和多图识别。

图片＋ngram 使用 Graph、context8192、B1536/R128、prefill128、H512MiB×C、
图片上限1024及逐字节搬运验证。每个组合检查冷输入、缓存恢复、取消一路、恢复
全部四轮请求，再检查每路红/蓝图片。复制请求须与源文一致，cold/restored/recovered
token 精确一致并有有效 ngram 接受；自由生成一路检查非退化输出。三轮恢复阶段
须实际达到配置路数并执行多路 compact decode，同时检查取消隔离和资源退役。

| KV | C | 图片 | MTP | ngram 配置→生效 | 结果 |
|---|---:|---|---|---|---|
| INT8 | 2 | CPU | 自适应 3 | 31→15 | PASS |
| RK8V4 | 2 | resident | 固定 3 | 31→15 | PASS |
| RK4V4 | 2 | resident | 固定 3 | 31→15 | PASS |
| RK4V4-E8 | 2 | CPU | 固定 15 | 31→15 | PASS |
| RK2V4-E8 | 8 | CPU | 自适应 3 | 63→15 | PASS |

RK2V4-E8 的冷阶段峰值为 3，三轮恢复阶段均为 8，八路图片颜色均正确。
INT8/CPU/C2 冷阶段 rounds=rows=114，说明该阶段没有多路 decode 重叠；三轮恢复阶段均 peak=2 且 rows>rounds。RK4V4-E8/CPU/C2 的三轮恢复也均 peak=2 且 rows>rounds。五项图片/ngram 组合共 80 次请求，加上文本历史 54 次及图片身份 11 次，共 145 次验收请求（不含首次失败尝试和预期的预算拒绝）。

## 测试夹具和构建记录

首次留存测试程序时漏带 FFmpeg 运行库，真实模型程序以 `0xC0000135` 在启动
阶段结束，未执行推理。补入相同构建目录的运行库后重跑；失败记录保留在
`first-snapshot-missing-dll/`，程序和运行库哈希记于 `tested-binaries.json`。

INT8/CPU/C2 和 RK4V4-E8/CPU/C2 的首次推理通过两个冷请求的复制及生成检查，
但未满足夹具的“冷阶段必须重叠”断言。冷 prefill 本来串行，快速复制一路可能
在下一路 CPU 图片进入 decode 前结束。夹具改为记录冷阶段 peak/rounds/rows，
三轮恢复阶段继续严格要求满路数重叠。逐 token、逐字节、取消和预算断言均未
放宽；新夹具另存程序副本。首次断言失败日志保留在 `first-cold-overlap-assertion/`。

增量构建曾遇到 Ninja 依赖日志重整时的权限/重命名中断，随后空依赖日志引发
构建工具断言。停止该构建，保留完整重整日志和部分新日志后恢复完整日志，
测试目标增量构建成功。没有重新配置构建或修改生产源文件来规避此问题。

## 入口和适用范围

Engine options、服务参数、Windows 启动器测试通过：覆盖五种格式、C1–8、
文本/resident/CPU、MTP 和 ngram 保留/降宽，同时保留四种非 fast 格式、无效
参数及冷快照拒绝。真实服务程序的 15 个 fast/text/resident/CPU/C8 组合均通过
参数准入并到达预期的缺失模型文件错误，四种非 fast 格式拒绝，显式
`--kvmem-disk-mib 0` 仍报冷快照暂时禁用。此项只验证 CLI 入口。

这是定向功能/数值验收，未进行性能比较、通用质量评测、完整组合全排列、
每图实际 1024 tokens 或 HTTP 并发矩阵补测。此前 K8V4 CPU/C1/自适应15/ngram63
的换行差异仍未处理，见 [原始记录](vision-ngram-validation-20261005.md)。

工作区原始日志、命令、退出码、程序哈希和修改前备份保存在
`results/fast-prefill-admission-20261006/`。

`build-backends/ninfer/apps/ninfer-serve.exe` 与 `ninfer-serve-concurrency8.exe`
已同步并与本轮 CLI 检查的程序哈希一致。旧发布 ZIP 和历史 qualified 程序没有更新。

`ninfer-kvmem.patch` 和 `versions.json` 已按当前完整工作树同步，保留同目录多卡改动；
`prepare-backends.py --backend ninfer --check` 通过，真实暂存区未改变。
