# ninfer KVMem 参数定向补测（2026-10-04）

## 结论与范围

12 个目标组合完成定向验收；15 次配置运行加 3 次原生诊断，共 18 次模型启动。
保留 4 次失败尝试，只有受影响范围重跑；累计真实模型实验
1247.3 秒（20.8 分钟），不含构建和分析。没有重跑 NVFP4 的 229 项或 K8V4 的
26 项基线，也没有进行性能比较。每个模型进程包含多个已有断言，进程数不是请求数。

- 固定 MTP 从 1–4 开放到 1–15；原生数学及 CUDA 内核未改。
- 自适应 MTP：单路文本，ngram/lookup/冷快照关闭；Graph 实验实际观察到 3→4 宽度切换。
- RK8V4：文本 C1–4、resident/CPU 图片 C1，采用既有 Main/MTP 与 Host 页搬运。
- 快速 prefill：INT8/RK8V4 文本，冷快照关闭；独立数值检查通过，不宣称吞吐收益。
- CPU Vision：图片 C1，复用原生 CPU 编码与 Host bridge，补齐 KVMem query 重放衔接。

BF16/INT8/NVFP4/K8V4/RK8V4 的准入和拒绝条件有 Engine/CLI 参数检查覆盖。实际模型组合见
下表；不是所有格式、草稿宽度、并发数及开关的全排列。更大预算、连续 16K 输出、其他设备和
模型，以及自适应多路/图片、快速图片/冷快照，均未新增验收。

## 环境和配置

- RTX 5060 Ti：`GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`；本轮全部 GPU 实验使用此 UUID，每次启动前检查占用。
- 模型：`Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`，已有 IQ3S＋Vision＋MTP 制品。
- 已有 Release、CUDA 13.2、sm_120a、VS 2022 构建树，8 并行编译，无重新配置。
- history：B384/R128/H2GiB、prefill128、C1、逻辑上限16K。
- concurrency：B384/R128（每路）、H512MiB、prefill128、逻辑上限8K、C2/C4。
- vision：B768/R128/H512MiB、prefill128、C1、每图最多256 merged tokens。
- device profile off，默认 CUDA 内存策略，ngram/lookup0，逐字节 KV/state transfer 检查。

## 真实模型实验

| # | 组合与主要检查 | 最终结果 | 成功运行秒数 |
|---|---|---|---:|
| 1 | K8V4 MTP5 Graph，越窗/复用/取消 | PASS | 19.9 |
| 2 | K8V4 MTP15 eager，越窗/复用/取消 | PASS | 19.0 |
| 3 | K8V4 MTP15 Graph，越窗/复用/取消 | PASS | 19.6 |
| 4 | NVFP4 MTP15 Graph，raw/思考/工具续写端点 | PASS | 87.9 |
| 5 | INT8 MTP15 Graph，快速 prefill | PASS | 23.3 |
| 6 | K8V4 自适应 MTP15 Graph | PASS | 25.7 |
| 7 | NVFP4 自适应 MTP15 eager | PASS | 18.6 |
| 8 | RK8V4 MTP15 C4 Graph，快速 prefill/会话隔离/取消 | PASS | 96.6 |
| 9 | RK8V4 MTP5 C2 eager，会话隔离/取消 | PASS | 60.0 |
| 10 | RK8V4 resident Vision MTP15 Graph | PASS | 63.6 |
| 11 | NVFP4 CPU Vision MTP0 eager | PASS | 228.4 |
| 12 | K8V4 CPU Vision MTP15 Graph（分段验收） | PASS | 131.0 |

第 12 行耗时是修复后的 query-only 重跑；失败前七个已通过请求保存在初次运行日志。
原生普通位置诊断 151.9 秒通过，对齐边界诊断 36.9 秒失败，修复后同一边界 31.5 秒通过。

## 基础检查

Engine options、serve options、原生自适应控制器、MTP Graph profiles、INT8 fast prompt 独立
oracle、RK8V4 fast prompt 独立 oracle、Windows launcher，共 7 项通过。Graph profiles 原有
范围覆盖各 KV 格式、两种 D256 head geometry 和 draft widths1–16。RK8V4 fast oracle 通过
测试进程内的 `NINFER_PROMPT_FAST=1` 选择生产路径；真实模型通过正式参数选择 fast 路径。
没有改变 oracle 容差或 CUDA 算法。

## 四次失败与处理

1. **端点测试夹具的 EOS 假设**：旧 fixture 强制生成160 tokens并关闭模型停止，NVFP4 MTP15
   在第110个 token已输出 `<|im_end|>`，后续包括 `<|endoftext|>`/新角色 token。默认展示过滤
   special tokens，用可见 `content` 重建 assistant 后不是相同原始前缀。KVMem安全回退输入
   checkpoint，拒绝错误的完整端点命中。保存 tokenizer/EOS 和逐 token证据；夹具改为尊重
   自然停止、按实际生成长度检查端点。重跑 generated110，工具续写 reused3435，通过。
   生产复用、匹配、停止行为未改，其余 history 的严格输出长度断言保留。
2. **CPU Vision 的 KVMem 适配缺口**：原 `rewind_resident` 是此前 KVMem 集成添加的 resident
   重放入口，CPU编码第一张图片成功，旧图片跨窗口query触发重放时被该入口拒绝并使Engine
   不可用。不是原生 CPU Vision 数学缺陷，也不是需保留的继承行为。入口改为允许resident/CPU，
   rewind前join已预取CPU任务并清除旧cursor关联，保留已有payload检查、handoff退休与重编码。
   CPU MTP0 eager 的图片替换、同placeholder不同内容、query重放、多图、取消恢复、
   必留页超预算拒绝及随后恢复均通过。CPU MTP15 Graph 的早期七个请求也通过，
   随后发现下面的原生边界问题。Overlay继续拒绝。
3. **继承的 Host Vision＋MTP 边界缺陷（两次失败）**：CPU MTP15 的 query 图片首次触发
   `scatter: expected src [D,V], indices [V], dst [D,T]`。普通位置的原生五请求诊断通过；
   将第一视觉列对齐到单元边界的原生单请求同样失败。原始源码只在 Main scatter 列非零时
   分配和复制 Host staging，Main 为零时 shifted MTP 仍需要下一列。来源、触发和影响已记
   本地 `backend-p0p1-20261002/legacy-findings.md` 的 L004。用户批准后改为独立分配 staging、无 Main 列也复制下一列，
   未改 CUDA 数学，最坏容量仍是 chunk+1。原生对齐边界复测通过；KVMem 只补跑 query重放、
   重复复用、多图、query取消/恢复五个请求，全通过。与失败前七个已通过请求组合构成
   CPU MTP15 的定向证据；本轮没有重跑该配置的超预算拒绝尾段，CPU MTP0 保留该覆盖。

## 工件与限制

完整本地证据目录为 `backend-p0p1-20261002/results/parameter-admission-20261004/`。
其中 `runs.json` 保存原始命令、设备、耗时及各次失败/成功；计划、日志、端点 token/EOS证据、
修改前文件和 authored diff也在该目录。本文保存可随源码提交的配置、结论和失败处理摘要；
原始运行数据保留在本机。

5060 Ti 的 18201 服务已恢复为原 IQ3S／K8V4／固定 MTP4／Graph 配置，B32768/R16384/H6144MiB，
ngram/lookup0，健康检查与 `/props` 预算核对通过。PID35736，使用现有构建树另名输出的
`build-backends/ninfer/apps/ninfer-serve-qualified.exe`；编译参数未变，8 并行。5050 的 Bonsai
18202／PID3540 正在处理请求，保留原服务与二进制，未重启。18181 的 tester llama 服务按
用户明确决定停止，未恢复。启动命令、可执行文件哈希和健康证据保存在本地证据目录的
`restored-services.json`。

源码、版本化 ninfer 集成补丁、启动器和本文随此次提交保存。未提交模型、二进制、缓存或
原始运行数据，也未重打冻结运行包。
