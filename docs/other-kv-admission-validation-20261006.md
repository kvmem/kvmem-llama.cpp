# 新增 KV 格式准入补测（2026-10-06）

本报告记录九种 KV 的准入补测。写本报告时，快速 prefill 仍只允许 INT8/RK8V4 文本。该限制已由 [快速 prefill 准入补测](fast-prefill-admission-validation-20261006.md) 放宽。当前边界见 [多后端运行](multi-backend.md)。

用户要求先测试能否简单放行，复杂修改另行确认。本轮补入 `fp8`、`rk4v4`、
`rk4v4-e8`、`rk2v4-e8` 的 KVMem 格式身份，并扩展 Engine、服务端和 Windows
启动器的准入清单。沿用已有原生量化、Attention、Main/MTP 页池和字节搬运实现，
没有修改生产量化算法或执行架构。

当前可选格式共九种：`bf16`、`int8`、`fp8`、`nvfp4`、`k8v4`、`rk8v4`、
`rk4v4`、`rk4v4-e8`、`rk2v4-e8`。文本及 resident/CPU 图片准入 C1–8，
ordinary 或固定/自适应 MTP1–15。ngram 非零要求 MTP；C2–8 将有效配置的
16–63 自动降到 15，C1 保留原宽度。冷快照保持暂时禁用。本轮结束时，快速 prefill
仍仅允许 INT8/RK8V4 文本，CPU 图片编码仍串行。

## 环境与范围

- 原生固定基线：`6e9189c1089f6968e29fe2e1c51b4596e2346eed`，既有 Release
  构建目录、CUDA 13.2、sm_120a、MSVC 14.44.35207；编译使用 8 个并行任务。
- 参数、页搬运和数值检查使用空闲的 RTX 5050 Laptop，UUID
  `GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29`。
- 真实模型检查使用能容纳模型的 RTX 5060 Ti，UUID
  `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`。
- 模型：`converted-models/Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`，
  位于本实验目录；图片为 `p9-inputs/red.png` 和 `blue.png`，512×512、实际
  每图 256 merged tokens，配置上限 1024。模型测试的 device profile 为 off。
- 本轮为定向功能/数值验收，没有性能比较、全排列验收或通用模型质量结论。
  完整默认大预算、每图实际 1024 tokens、HTTP 并发矩阵未在新四种格式上重测。

## 数值和搬运

四种格式各通过 90 项稀疏 Attention 检查，共 360 项：D256/Q24/KV4，完整、
稀疏和乱序逻辑页，页偏移 0/63，T1/2/3/4/5/13/53/63/64/65，以及 eager/Graph。
输出对比已有独立 FP64 masked-attention oracle，输入从独立量化编码及存储 scales
解释，不改变已有数值判据。E8 格式另检查 fused/standalone append 的精确字节一致性。

页搬运测试独立指定三层 Main 加一层 MTP 的 codes/scales，检查 page-major 和
head-major、碎片物理映射、分组搬运、分平面恢复、过期 generation 和保护区。
四种新格式全部通过。D256 每 token 每 KV head 的独立原始 payload 为：

| 格式 | K codes/scales | V codes/scales | 合计 |
|---|---|---|---|
| FP8 | 256 B + 2 B | 256 B + 2 B | 516 B |
| RK4V4 | 128 B + 8 B | 128 B + 16 B | 280 B |
| RK4V4-E8 | 128 B + 8 B | 128 B + 16 B | 280 B |
| RK2V4-E8 | 64 B + 8 B | 128 B + 16 B | 216 B |

首次 RK4V4 稀疏测试以退出码 `0xC0000005` 结束：本次测试调用了仅适用另一组
格式的 `make_cache(storage)` 夹具，其 packed-key 缓冲没有初始化。改用原生测试
已维护的 `kPlanRk4v4` 专用夹具后，90 项全部通过。修正限于测试调用，没有修改
生产 codec，也没有放宽数值或复制一致性检查。首次失败日志保留。

## 真实模型

四种格式分别通过 ordinary/eager 的 C1 历史回归：context16384、B384/R128、
prefill128、H2GiB、逐字节搬运验证。每种 10 次请求，覆盖冷输入、重复恢复、
下一轮/重算对照、关闭复用、编辑前缀、取消 decode/prefill 及恢复。
相同 checkpoint 的 greedy token、取消后恢复 token 精确一致；Main/MTP 页和
Host 使用量满足退役及预算断言。

图片＋ngram 采用 Graph、context8192、B1536/R128、prefill128、H512MiB×C，
逐字节搬运验证。每个组合检查四轮并发：冷输入、缓存恢复、取消一路、恢复全部；
复制请求要求源文一致、cold/restored/recovered token 精确一致及有效 ngram 接受，
另有自由生成请求和每路红/蓝图片识别，并检查满配恢复并发、Host 预算、页退役。

| KV | C | 图片 | MTP | ngram 配置→生效 | 结果 |
|---|---:|---|---|---|---|
| FP8 | 2 | resident | 固定 3 | 31→15 | PASS |
| RK4V4 | 8 | CPU | 自适应 3 | 31→15 | PASS |
| RK4V4-E8 | 2 | resident | 固定 15 | 31→15 | PASS |
| RK2V4-E8 | 8 | CPU | 自适应 3 | 63→15 | PASS |

八路冷输入的首次 prefill 串行，不要求冷阶段每轮均达到八路；恢复阶段要求实际
峰值八路。RK4V4 与 RK2V4-E8 冷阶段峰值均为 3，后三轮峰值均 8，八路颜色均正确。
MTP15 的 E8 组合及 FP8 两路颜色、复制、恢复也全部通过。
四个历史实验加四个图片/ngram 实验共 140 次模型请求（40 次历史、100 次并发/颜色）。

此前 K8V4 CPU 图片/C1/自适应15/ngram63 的换行差异仍未解决，见
[原始记录](vision-ngram-validation-20261005.md)。本轮没有改动该路径。

## 程序和记录

Engine options、服务参数、Windows 启动器测试通过，覆盖新增格式及 C1–8
准入，同时保留未知格式、无效预算、快速图片和冷快照拒绝。最终服务端参数测试
再次通过；真实服务程序的四种图片/C8/ngram31 组合均通过准入并到达预期的缺失
模型文件错误，显式 `--kvmem-disk-mib 0` 仍报冷快照暂时禁用。此检查仅验证
CLI 入口，真实推理由上面的 Engine 实验验证。

`build-backends/ninfer/apps/ninfer-serve.exe` 和 `ninfer-serve-concurrency8.exe`
已同步，两个文件哈希一致；可用统一启动器的 `-Worker` 指定。旧发布 ZIP 和
历史 qualified 程序没有更新。`ninfer-kvmem.patch` 及 `versions.json` 已同步，
`prepare-backends.py --backend ninfer --check` 通过，原有暂存区未改变。

实验原始记录在工作区 `results/other-kv-admission-20261006/`，包含各项日志、
命令/退出码 JSON、首次 RK4V4 失败和重跑结果、最终程序信息及本轮修改前备份。
