# 双卡 tensor 并行设计（基于当前固定版 llama.cpp）

## 结论和边界

优先复用 llama.cpp 的原生 `LLAMA_SPLIT_MODE_TENSOR`，不实现 KV 镜像。它将两张物理 GPU 包装成一个 Meta 设备；Q/K/V 投影及注意力输出按维度分片，`cache_k_l*`、`cache_v_l*` 按注意力头/通道分片，Qwen3.5 的 `cache_r_l*`、`cache_s_l*` 也按 GDN 通道分片。其他少量张量可能是镜像或部分和。GGML Meta buffer 的 `tensor_get/set` 已负责将一个完整逻辑 KV 行拼接到主机内存，或从主机行分发到物理分片。因此 KVMem 的 block ID、slot、主机 RawKvStore 格式和检索策略原则上可保持不变。

这不是放开 `--split-mode tensor` 开关即可完成的功能。KVMem 原先将 `model.n_devices() > 1` 当作多卡判据，而 tensor 模式下 `model.n_devices() == 1`（一个 Meta 设备）。如果不改，归档与回填会错误进入直接 CUDA 指针/单卡 stream 路径。首版限定同一 CUDA 后端的两张卡、`--gpu-layers all`、Flash Attention、单请求序列；视觉头设备选择独立于此方案。

## 当前实现与验证（2026-09-25）

已接通双 CUDA 卡的原生 tensor 模式、KVMem 同步 Meta KV 传输和按物理卡规划池容量。普通推理及嵌入式 nextn MTP 的 `snapshots` 模式均可运行；`ReplaySSM` 在入口明确拒绝，因为现有 CUDA fold 需要物理 shard 指针。Meta KV 保持原生分片，**没有复制整份历史 KV**。KVMem 自建 KV cache 在 tensor 模式下必须使用 `cache_k_l*` / `cache_v_l*` 标准名称，Meta 的模型回调才会按注意力头分片；带 `kvmem` 标签的名称会被识别为镜像，在 MTP 的 `SET_ROWS` 图上触发分片断言。逐卡预算通过新增的 Meta 设备枚举接口取得真实物理设备；同时修正上游 Meta 的显示名称为 `Meta(CUDA0,CUDA1)`。

RTX 5060 Ti 16 GiB + RTX 5050 Laptop 8 GiB、Qwen3.8-27B-UD-Q4_K_M、`--tensor-split 8,1`、Q8_0 KV，按仓库原任务 1 的文本做单次测试：

| 路径 | 预填充 | 生成 | 说明 |
| --- | ---: | ---: | --- |
| 原生 tensor、无 MTP CLI | 331.35 tok/s | 22.58 tok/s | 302 输入 token / 64 输出 token |
| KVMem tensor、无 MTP CLI | 335.71 tok/s | 21.80 tok/s | budget 384、reserve 128、query replay 64 |
| 原生 tensor + MTP CLI | 688.05 tok/s | 35.19 tok/s | 预填充指标不含 MTP 准备工作 |
| KVMem tensor + MTP snapshots CLI | 569.40 tok/s | 32.81 tok/s | budget 384、reserve 128、query replay 63；64 个输出 token 与原生 MTP 全同 |
| KVMem tensor server、无 MTP | 239.56 tok/s | 21.11 tok/s | 同一任务文本作为 chat 请求，314 输入 token / 64 输出 token |
| KVMem tensor server + MTP snapshots | 221.95 tok/s | 39.83 tok/s | 相同 chat 请求，单次新进程测量 |
| 原生 tensor server + MTP | 255.34 tok/s | 39.99 tok/s | 相同 chat 请求；返回文本与 KVMem MTP 完全一致 |

CLI 与 server 的模板/统计口径不同，表中的预填充数不能跨行直接排名；以上均为单次样本，不是稳定吞吐分布。改用 Meta 物理设备枚举接口并重建后，server MTP snapshots 再跑一次为 39.43 tok/s。MTP CLI 在 `--kvmem-query-last 0` 时为 34.40 tok/s，64 个 token 也与原生 MTP 全同。小预算 128 + reserve 128 的 32-token MTP 归档/回放能完成，但输出从第 8 个 token 起与无归档原生路径不同；检索式历史选择本身会改变上下文，不能据此判为分片错误。`8,1` 的实际 KV 字节分片受头数取整影响：无 MTP 估算每 token 约 CUDA0 30,464 B、CUDA1 4,352 B；加入 MTP 保守预留后为 36,608 B 与 6,400 B。启动时还按卡预留 GDN snapshot 与计算余量。

完整 K/V 行经 Meta `tensor_get/set` 的冷块回填对比已得到 K/V cosine=1、RMSE=0、K packed bytes mismatch=0；强制选择非连续历史块后的 CLI query replay 仍会触发已有的 position-gap 错误，layer 模式同样复现，因此不能把该特殊用例列为已通过。普通任务 1 的服务端请求成功。P2P 和异步分片传输尚未实施；实测这台机器上 tensor 预填充比 layer 慢，优化前不应把 tensor 当作默认分卡模式。

服务端启用示例（`<model.gguf>` 换成实际路径）：

```powershell
llama-kvmem-server.exe -m <model.gguf> --device CUDA0,CUDA1 --split-mode tensor --tensor-split 8,1 -ngl all --kvmem --kvmem-budget 384 --kvmem-gen-reserve 128 --kvmem-block-tokens 32 --spec-type draft-mtp --spec-draft-n-max 2 --kvmem-mtp-state snapshots
```

不使用 MTP 时去掉最后三个 MTP 参数；tensor + KVMem 的 `replay` 状态当前会在加载模型前报错。`--tensor-split` 是权重/算子比例，KV 最终比例还受注意力头取整约束。此示例的预算只适用于短任务，长上下文应按两卡剩余显存重新设定。

## 同条件 layer / tensor 对照（2026-09-25）

同一 `llama-kvmem-server` 构建、Qwen3.8-27B-UD-Q4_K_M、任务 1 文本（chat 模板后 314 输入 token，greedy 输出 64 token）、`-c 1024 -b 128 -ub 128`、Q8_0 KV、KVMem budget 384 + gen reserve 128、block 32、未加载视觉头、`KVMEM_TRACE` 关闭。每行用新进程测量；显存是在请求期间每约 150 ms 采样的最低剩余 MiB，顺序为 5060 Ti / 5050 Laptop。MTP 使用嵌入式 nextn、宽度 2 和 snapshots。MTP 行做了两次，表内为范围；无 MTP 行是单次。

| 分卡 | MTP | 预填充 tok/s | 生成 tok/s | 最低剩余显存 MiB：5060 Ti / 5050 |
| --- | --- | ---: | ---: | ---: |
| layer 5:1 | 关 | 302.00 | 19.87 | 4,213 / 4,489 |
| tensor 5:1 | 关 | 242.17 | 22.20 | 3,405 / 5,157 |
| layer 8:1 | 关 | 284.58 | 20.44 | 3,225 / 5,471 |
| tensor 8:1 | 关 | 232.99 | 20.89 | 2,371 / 6,189 |
| layer 5:1 | snapshots | 297.13–306.79 | 34.25–35.95 | 3,951 / 4,131 |
| tensor 5:1 | snapshots | 235.94–242.16 | 41.70–42.13 | 2,853 / 4,971 |
| layer 8:1 | snapshots | 313.20–313.94 | 38.56–39.60 | 2,943 / 5,145 |
| tensor 8:1 | snapshots | 227.66–232.28 | 38.97–39.28 | 1,773 / 6,055 |

相同数值的 `--tensor-split` 在两种模式下并不产生相同的物理占用：tensor 在 5060 Ti 上持续多占约 0.8–1.2 GiB，在 5050 上少占相近的量。按两卡最低剩余显存之和估算，tensor 的总占用无 MTP 多约 0.14 GiB、有 MTP 多约 0.26 GiB。5:1 的 tensor MTP 解码较快，但预填充较慢；两个模式处理整次 314+64 token 请求都约 2.8–2.9 秒。8:1 的 MTP 解码速度接近，layer 整次请求约 2.6 秒，tensor 约 3.0 秒。对以更大模型或更高精度为首要目标的这组显卡，当前 layer 5:1 留下最均衡的显存余量；layer 8:1 是该短任务的更快折中。以上是短请求和小 KV 池测量，不能直接推断长上下文吞吐；不同分卡模式的生成文本有细小差异，MTP 接受率也可能不同。

tensor 3:1、4:1、5:1 在同一任务一上经过预热和重复测量的后续数据见[分片比例对照](task1-tensor-split-3-4-5-2026-09-25.md)。

## 数据流

```
原生 Meta 模型:  GPU0 的权重/KV/GDN 分片  <->  Meta 图  <->  GPU1 的权重/KV/GDN 分片
                                | ggml_backend_tensor_get/set
                                v
                   KVMem 完整逻辑 KV 行与 RawKvStore
                                |
                                v
                      block/slot 检索和归档逻辑
```

KVMem 通过现有 `kvmem_tensor_get/set` 操作逻辑 KV tensor，GGML Meta buffer 在底层分片/拼接。这里的传输是同步的正确性基线，后续再考虑异步分片传输与 P2P。注意调用必须按完整逻辑行对齐：当前 K/V tensor 是 `[embedding, cells, 1]`，block 读写的 offset/size 正好是 `cell * ggml_row_size(...)`。需要对每种实际 KV dtype、最后一个不足整块的 block 做断言和往返测试。图上捕获的 Q/K 临时 tensor 也应先经调度器同步，再走 Meta `tensor_get`，禁止把 Meta tensor 的 `t->data` 当成某张 CUDA 卡的指针。

## 需要改动的模块

1. **模式与设备识别。** 将 `tools/kvmem-server-devices.h` 的拒绝条件改为明确支持 `layer` 与 `tensor` 两种 CUDA 双卡模式；保留禁止混合 CUDA/Vulkan 用于语言模型、GPU 全层加载、tensor split 数量校验。适配器使用显式 split mode/设备类别，而不是用 `n_devices()>1` 推断物理卡数。`layer` 路径的“每层 KV buffer 归属该层 GPU”断言不能用于 Meta；tensor 路径应验证 K/V buffer 是 Meta 类型、维度和行跨度满足预期。
2. **同步 KV 传输。** 将现有双卡 layer 的同步 `ggml_backend_tensor_get/set` 路径用于 Meta KV 与捕获 tensor。统一禁用 KVMem 的单卡 D2D 布局、异步 D2H/H2D stream、CUDA 指针 staging、直接 mean-K CUDA kernel 等快路径；保留主机实现。对原生 Meta buffer 做一个小型完整行/部分 block 的 round-trip 测试，证明 host 的数据次序与原模型一致。
3. **逐物理卡池容量。** 原 `kvmem_compute_pool` 按 `model.dev_layer(il)` 汇总 layer KV；tensor 模式的所有层都指向一个 Meta 设备，不能沿用。以 `model.params.devices` 的物理 CUDA 列表和上游 `llama_meta_device_get_split_state` 对每层 K/V（以及 MTP 跟随层）求出各卡每 token 的真实分片字节数。加载权重后分别读取 GPU free/total，扣除图缓存与 GDN/MTP 余量；共同的 KV slot 数取各卡允许值的最小值。对借用的 hybrid attention cache，要在实际分配前做同一预估，避免事后 OOM。启动日志输出每卡权重后空闲、KV row、状态余量、计算出的最大 cells 和最终 cells。特别检查 5060 Ti/5050 的不对称 split：头数/量化块粒度的取整可能使小卡分到 0 个 KV 头，`--tensor-split 8,1` 不能假定自动得到期望的 8:1 分片。
4. **GDN 状态。** Qwen3.5 的 GDN `r/s` 由上游 Meta backend 分片。无 MTP 先使用原生 recurrent memory，并确认回滚/清理可通过 Meta buffer 完成。当前 `use_gdn_replay` 只接受 CUDA 物理层设备，且 ReplaySSM fold 直接使用 `t->data`、按 CUDA device 分组；首版在 tensor 模式下必须关闭/明确报错，不能默默走此路径。后续可先支持原生 snapshots，再扩展 ReplaySSM：获取每卡 GDN 状态和 record 的真实物理 shard 以及设备指针，按 shard 构造 fold descriptors，并按两卡完成后统一提交。
5. **MTP follower。** 当前 follower 对 nextn KV 要求单张层 GPU、D2D 布局也使用直接 CUDA 指针。主模型无 MTP 正确后，再让其通过 Meta KV `get/set` 使用主机布局回退；保证接收/拒绝草稿后的 KV 和 GDN 回滚一致。最后才优化 D2D。未完成适配前，tensor+MTP 启动应给出明确错误，避免隐藏的错误结果。
6. **性能观测。** 分开记录预填充、生成、Meta collective/同步、逻辑 KV D2H/H2D、检索耗时和每卡峰值显存。P2P 是性能优化，不是功能前提；必须先测这台机器的设备拓扑和实际吞吐，不能从 layer 的速度推断 tensor 一定更快。

## 落地顺序与验收

| 阶段 | 交付 | 通过条件 |
| --- | --- | --- |
| 0 原生可行性 | 同版 `llama-cli` 双卡 tensor，任务 1，小模型后 Q4_K_M | 模型可加载、输出正确；记录两卡显存与 prefill/decode。 |
| 1 KVMem 基线 | 放开模式、Meta 检测、同步 KV/capture、MTP 禁用 | 任务 1 与无 KVMem 原生 tensor 结果相符；归档、逐出、回填、检索、取消请求后再请求正确。 |
| 2 容量和稳定性 | 逐卡预算及不对称比例预检 | Q4_K_M 在 5060 Ti + 5050 上不过量分配；改变 budget/reserve 有可解释的两卡显存变化。 |
| 3 MTP | snapshots follower，之后 ReplaySSM | 与主模型 token、接受率、GDN 状态一致；失败能显式降级/报错。 |
| 4 优化 | 异步分片传输/P2P（仅数据证明确有收益时） | 相同任务 1 和相同参数，吞吐提升且无正确性回退。 |

阶段 0 不需要改上游算子。阶段 1–2 主要是 KVMem 设备/传输/显存规划适配，预计约 **1–2 周工程时间**，具体取决于原生 Qwen3.5 tensor 能否在这两张卡上跑通。阶段 3 的 snapshots 约 **数天到一周**；ReplaySSM 涉及物理 shard 指针和 CUDA fold，约 **再 1–2 周**。这些是设计级估计，阶段 0 的实际运行结果可显著改变范围。如果改成完整 KV 镜像，则必须改上游 Meta 分片规则及 attention 计算/通信，并增加每卡 KV 显存，工作量和维护成本都明显更高，不应作为首版。

## 关键代码依据

- `llama.cpp/src/llama.cpp`：tensor 模式创建一个 Meta device，包装所选物理设备。
- `llama.cpp/src/llama-model.cpp`：K/V cache、Qwen3.5 GDN state 的分片规则与粒度。
- `llama.cpp/ggml/src/ggml-backend-meta.cpp`：Meta buffer 的 `get_tensor`/`set_tensor` 对 axis-0 分片执行拼接/分发。
- `src/adapter/llama-memory-kvmem.cpp`：当前的多卡判据、每层归属断言、池容量计算和同步传输。
- `src/adapter/llama-memory-kvmem-hybrid.cpp`、`src/adapter/llama-memory-kvmem-mtp.cpp`：ReplaySSM 与 MTP follower 的物理 CUDA 指针假设。
