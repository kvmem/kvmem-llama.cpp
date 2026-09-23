# KVMem × llama.cpp 修改计划

当前状态以 [rc3 变更说明](milestones/v0.16.0-rc3.md)和[用户反馈待办](user-feedback-triage.md)为准。下方 2026-09-15 的阶段说明保留作历史记录，其中默认值不代表当前版本。

**状态（2026-09-15）：** P0–P3 **`v0.3.0`**。P4–P5 **`v0.4.0`**。P7 **`v0.5.0`**。量化 KV spill **`v0.6.0`**。GPU stage-in **`v0.7.0`**。32 MiB slab **`v0.8.0`**。packed GPU K/V memcpy + mean-K + 块写满异步 D2H **`v0.9.0`**。decode mean-K + 进程内 prefix reuse + chat tools T1–T5 **`v0.10.0`**。query clamp / thinking budget / stream heartbeat **`v0.11.0`**。same-query skip / recency suffix / stream usage **`v0.12.0`**。compact `drop_reuse` / role-block query / `prompt_cache_*` **`v0.12.1`**。GDN gen-start ckpt / `prefill_tail_offload` **`v0.12.2`**。MTP KV dtype / `FA_ALL_QUANTS` / Qwen3.8 sampling **`v0.12.3`**。OpenAI 图片 / mmproj / 视觉 KV **`v0.13.0`**。query replay skip / IQ3·IQ4 可移植启动器 **`v0.14.0`**。FP32 GDN ReplaySSM / 默认 MTP3 / IQ3·IQ4 图文与 256K 复测 **`v0.15.0`**。host mean-K 单累加器 **`v0.15.1`**（tag `v0.15.1`）。产品默认 `--kvmem` = retrieval + query-last 64，`--kvmem-query-max-tokens` 512，GPU KV **q8_0**；MTP 可选 `--spec-type draft-mtp`（默认 **none**），`--spec-kv-dtype` 可单独设 MTP KV。IQ3/IQ4 脚本默认 `--kvmem-query-replay auto --kvmem-query-policy user`、`--mmproj` + MTP n_max=3、`--kvmem-mtp-state replay`。阶段测试：[milestones/v0.15.1.md](milestones/v0.15.1.md)。P6 未开始。阶段 D `state_write` 与 T6 json_schema 未进此 tag。

本文是落地文档，不是再写一遍可行性分析。架构结论见技术方案；这里规定：**改什么、不改什么、按什么顺序合入、每一步怎样算过关。**

上游对照：

- KVMem 实现与文档：[kvmem/kvmem-qw3](https://github.com/kvmem/kvmem-qw3)
- 推理引擎：llama.cpp submodule（钉 tag，薄补丁）

---

## 1. 目标

把 KVMem 从 qw3 自研 CUDA 引擎里剥出来，变成：

1. 独立库 `kvmem/`：块选择、分层存储、检索、窗口装配。
2. llama.cpp 继续做唯一推理引擎：模型图、量化、Flash Attention、GDN、采样。
3. 两者通过 `llama_memory_i` + 少量可 rebase hook 对接。
4. 后续 llama.cpp 升级以「bump submodule + 重放 `patches/`」为主，而不是跟一套 diverged fork。

产品能力（v1）：任意 llama.cpp 能跑的**纯注意力模型**，可以把超长上下文变成「有界 GPU 工作集 + 可召回历史块」。

产品能力（v1.5）：Qwen 3.5 / 3.6 hybrid（注意力半边 KVMem，DeltaNet 半边原样）。

---

## 2. 非目标（本计划明确不做）

| 不做 | 原因 |
|---|---|
| 改 `ggml_flash_attn_ext` 或任何 backend FA kernel 去读 page-index | 锁死硬件、升级面爆炸 |
| 每次 reselect 把窗口 pack 成 `[0..W)` 再整段拷贝 | resident 块必须留在原槽 |
| GPU 上保留全量历史、只用 mask 藏 token | 省不了显存，decode 仍 O(全量) |
| 追上 qw3-native 在 Blackwell + Qwen3.6 的 tok/s | 那是 FlashInfer / MMQ / NVFP4，不是 KVMem |
| KVMem + continuous batching / 多 slot | qw3 也未完成；和 unified KV 冲突 |
| 自研 MTP 投机状态机 / 方案 A（MTP KV 跟 `-c` 涨） | 投机用 llama.cpp `draft-mtp`；显存必须方案 B。见 [kvmem-mtp-plan.md](kvmem-mtp-plan.md) |
| 把整套 KVMem 合进 ggml-org/llama.cpp | 先 submodule；最多上游一个 hook PR |
| 把 qw3 的 NVFP4 / MMQ / FlashInfer 搬过来 | 超出本项目 |

---

## 3. 已拍板的设计约束

实施时把这些当硬约束。违反其中任何一条，视为方案回退，停下来复盘。

1. **有界块槽位池，不是 paged FA。** GPU attention cache 容量 = `select_budget + gen_reserve`。每个槽 `block_tokens` 个 cell。槽号 ≠ 窗口 RoPE 坐标。
2. **Reselect 是 `KvMemPlan` diff。** `skip` 的块零拷贝；resident 且坐标变了的块只做 in-place re-RoPE；只有 `stage_in` / `stage_out` 走 H2D/D2H。禁止整窗口 pack。
3. **FA 扫 budget，不扫全历史。** llama.cpp 对这段有界 `n_kv` 做稠密 FA + mask。槽填满时 token 数与 qw3 选中窗口同阶。这是可接受的一阶以下差异，不为它改 FA。
4. **只稀疏标准注意力层。** DeltaNet / recurrent 必须看见每一个 token，走 `llama_memory_recurrent`。
5. **选择全局共享。** 所有注意力层同一组 block ID。
6. **immutable raw-K 为 K 的权威**（P2 起）。不要从 q8 cache 反推 content key。
7. **`kvmem/` 零 `#include` llama.cpp 头文件。** 所有 `llama-*.h` 只允许出现在 `src/adapter/`。
8. **llama.cpp 补丁只允许三处：** memory 工厂、`build_attn` 的 Q/K capture、CLI/server 元数据。核心补丁超过约 300 行则 P1 fail。
9. **v1 单序列 / 单 slot。** 关掉 SWA shift、unified 多 seq。MTP 在 v1 关闭；**P7 在同一单 slot 上打开**，draft KV 走槽位池（方案 B），仍禁止 CB + KVMem + MTP。
10. **成功标准是「llama.cpp 能跑 KVMem」，不是打败 qw3-native 的 decode 速度。**

已知 v1 限制（后续单独立项，不挡当前 retrieval）：

- **生成不能无限长（v1）。** retrieval pin 之后 decode 只吃 `gen_reserve` 空槽；槽满且当前块已写满则 `no free GPU slot` / `llama_decode(gen) failed`，**不会**为了续写去挤掉召回块。今天的办法是加大 `--kvmem-gen-reserve`，并在 agent prompt 里限制单轮（含思考）长度。
- **后续：只在 `gen_reserve` 里做 ring。** 不新开第三块池、不加 VRAM、不从 pin 住的 select 里抠槽。`alloc_slot` 失败时只 evict **一块**已写满的生成块（粒度 = `--kvmem-block-tokens`，16 GiB 配方 128），KV 下到 host store；GPU 上仍是完整召回窗口 + 本轮最近约 `gen_reserve`（IQ3 16K / IQ4 12K）。正在写的块不动。MTP 跟同一 slot 下标。详见 `docs/architecture.md`「Known v1 limit」。

---

## 4. 目标仓库布局

本仓库 `kvmem_llamacpp` 从空目录建起：

```text
kvmem_llamacpp/
  docs/
    modification-plan.md          ← 本文件
    architecture.md               ← 从技术方案摘一份短架构（P0 写）
  kvmem/                          ← 独立库，无 llama.cpp 依赖
    include/kvmem/
      kvmem_store.hpp
      kvmem_plan.hpp
      kvmem_config.hpp
      pinned_kv_tier.hpp
      nvme_kv_tier.hpp
      kvmem_runtime.hpp           ← engine-agnostic 装配接口
      kvmem_backend.hpp           ← 存储/拷贝抽象（CPU 先，GPU 后接）
    src/host/
      kvmem_store.cpp
      pinned_kv_tier.cpp
      nvme_kv_tier.cpp
      kvmem_runtime.cpp
    src/ggml/                     ← P2 才加：rope / mean-k / score 图
    tests/
      kv_block_store_test.cpp
      pinned_kv_tier_test.cpp
      nvme_kv_tier_test.cpp
      reselect_diff_test.cpp
    CMakeLists.txt
  llama.cpp/                      ← git submodule，钉 tag
  patches/                        ← 对该 tag 的 git am 队列
    0001-memory-factory-hook.patch
    0002-attn-qk-capture-hook.patch      ← P2
    0003-cli-kvmem-flags.patch
  src/adapter/                    ← 唯一允许 include llama.cpp 的地方
    llama-memory-kvmem.h
    llama-memory-kvmem.cpp
    llama-memory-kvmem-hybrid.h   ← P4
    llama-memory-kvmem-mtp.h      ← P7，nextn 槽位条带
    llama-kvmem-hooks.h
    llama-kvmem-batch.cpp         ← 窗口坐标 vs 真实位置
  tools/
    llama-kvmem-cli.cpp           ← P1 自有 CLI
    llama-kvmem-server.cpp        ← P5，可选
  scripts/
    apply-patches.sh
    rebase-llama.sh
    identity_canary.py
    needle_recall.py
    reselect_overlap.py
    long_ctx_vram.py
  CMakeLists.txt
  README.md
```

升级 llama.cpp 的固定动作：

```text
1. git -C llama.cpp fetch && git -C llama.cpp checkout <new-tag>
2. ./scripts/apply-patches.sh          # git am patches/*
3. 冲突只允许出现在 hook 那三个文件
4. ctest + identity/needle/overlap 回归
```

---

## 5. 从 kvmem-qw3 迁什么

源仓库路径均相对 <https://github.com/kvmem/kvmem-qw3>。

### 5.1 原样迁（P0，可单测）

| 源 | 目标 | 说明 |
|---|---|---|
| `include/qw3/kvmem_store.hpp` | `kvmem/include/kvmem/kvmem_store.hpp` | 命名空间 `qw3` → `kvmem` |
| `src/kvmem_store.cpp` | `kvmem/src/host/kvmem_store.cpp` | host-only，不改语义 |
| `include/qw3/pinned_kv_tier.hpp` | `kvmem/include/kvmem/pinned_kv_tier.hpp` | |
| `include/qw3/nvme_kv_tier.hpp` | `kvmem/include/kvmem/nvme_kv_tier.hpp` | |
| `tests/kv_block_store_test.cpp` 等 | `kvmem/tests/` | 一并迁 |
| `EngineOptions` 里 KVMem 字段 | `kvmem_config.hpp` | 只迁 KVMem 相关，不迁 MTP/serving |

迁的时候允许做的改动：命名空间、include 路径、去掉对 `QwenExecutor` / `DeviceBackend` 的前向依赖。不允许改选择语义、`KvMemPlan` 字段含义、sink/recent 规则。

### 5.2 重写，不搬 CUDA（P1–P3）

| qw3 现状 | 本仓库做法 |
|---|---|
| `QwenExecutor::kvmem_assemble` page-index 重排 | 槽位分配器 + cell pos 写成窗口坐标 |
| `rope_block_remap_paged_*` | P1 可先不做 re-RoPE（recency 窗口坐标=写入坐标）；P2 用 `ggml_rope` in-place |
| `block_kmean_*` / softmax-over-pages CUDA | P2 用 ggml 图 |
| `DeviceBackend` 的 D2H/H2D | `kvmem_backend.hpp` 抽象；adapter 里用 `ggml_backend_tensor_get/set` |
| `qw3 serve` / Anthropic adapter | P5 再做，不进 P1 |

### 5.3 不迁

- `qwen_executor.cpp`、`kernels_cuda.cu`、MMQ、FlashInfer adapter、NVFP4
- continuous batching、MTP、prefix cache 服务端
- `llama_cli_backend.cpp`（那是对照，不是集成）

qw3-native **不在本计划里强制改成依赖公共库**。P0 结束可以开一个可选回接任务，避免策略双轨；不阻塞 P1。

---

## 6. llama.cpp 补丁契约

钉 tag 的原则：选一个**已验证能跑目标模型**的 release（P1 用纯 Transformer 如 Qwen3 / Llama 3；P4 再要求 Qwen3.6-27B）。不要追当日 master。

### Hook 1 — memory 工厂（P1，必做）

- 文件（上游，以当时树为准）：`src/llama-model.cpp` 里 `create_memory()`，或等价开关。
- 行为：`--kvmem` 时创建 `llama_memory_kvmem`，否则原路径。
- 我们的实现放在树外 `src/adapter/llama-memory-kvmem.cpp`，通过补丁把工厂接到这个类型。若编译系统不便链外部 .cpp，允许把 adapter 以 `llama.cpp/src/llama-memory-kvmem.cpp` 形式打进补丁，但逻辑仍视为 adapter，禁止在此文件里写选择策略。

### Hook 2 — Q/K capture（P2）

- 文件：`src/llama-graph.cpp` 的 `build_attn`。
- 行为：K 做 RoPE **之前**、Q 形成之后，若 hook 非空，插入 `ggml_cpy` 到预分配 buffer，或调用 `kvmem_hooks->on_q / on_k_prerope`。
- buffer 形状按 `n_ubatch × n_head × head_dim` **预分配固定**，避免打爆图缓存。
- 不要按 tensor 名字搜图。

### Hook 3 — 标志与元数据（P1 最小 CLI，P5 再扩展）

- P1：`llama-kvmem-cli` 自己的 flags，尽量不改 `llama-cli`。
- P5：要么独立 server，要么给 `llama-server` 加 opt-in JSON 字段 `kvmem.query_begin/end`。

### 禁止出现在 patches/ 里的东西

- `ggml-cuda/fattn*.cu`、`ggml-metal` FA、`ggml-vulkan` FA
- GDN / MTP 内部
- 量化、matmul

---

## 7. 分阶段修改计划（按 PR）

每个 PR 必须独立可审、可合、有退出标准。依赖只允许指向更早的 PR。

```text
P0-1 ─┐
P0-2 ─┼─► P0-3 ─► P1-1 ─► P1-2 ─► P1-3 ─► P1-4 ─┬─► P2-1 ─► P2-2 ─► P2-3
      ┘                                          │
                                                 ├─► P3-1 ─► P3-2     （可与 P2 后期并行）
                                                 │
                                                 └─► P4-1 ─► P4-2     （P2 完成后）
                                                          ├─► P5
                                                          ├─► P6
                                                          └─► P7-0 ─► P7-1 ─► P7-2 ─► P7-3
                                                               KVMem + MTP 方案 B（P4-2 后）
```

---

### PR P0-1 — 仓库骨架与 CMake

**标题：** `build: scaffold kvmem library and cmake`

**依赖：** 无

**改动：**

- 根 `CMakeLists.txt`、`kvmem/CMakeLists.txt`
- `README.md`（怎么编、怎么测、和 qw3 的关系）
- `docs/architecture.md`（从技术方案压缩到 2–3 页）
- 空的 `src/adapter/`、`patches/`、`scripts/apply-patches.sh`

**不做：** 还不加 llama.cpp submodule（P1-1 再加），避免一上来就锁 tag。

**退出：** `cmake -S . -B build && cmake --build build` 能编出一个空 `libkvmem.a`。

---

### PR P0-2 — 迁 KvMemStore + 单测

**标题：** `kvmem: import host block store from kvmem-qw3`

**依赖：** P0-1

**改动：**

- 迁 `kvmem_store.hpp/.cpp`、`kvmem_config.hpp`、`kvmem_plan.hpp`
- 迁并改 namespace 的 `kv_block_store_test.cpp`
- 补 `reselect_diff_test.cpp`：构造两轮 selection，断言 `skip` / `stage_in` / `stage_out` 数量

**硬约束：** `set_selection` 语义与 qw3 一致（窗口按 block_id 升序 pack 坐标；物理槽位是 P1 的事，Plan 里仍是窗口坐标 `from_base/to_base`）。

**退出：** `ctest --test-dir build --output-on-failure` 中所有 `kvmem/tests/*` 绿。

---

### PR P0-3 — CPU / NVMe tier 与 Runtime 接口

**标题：** `kvmem: import CPU/NVMe tiers and KvMemRuntime interface`

**依赖：** P0-2

**改动：**

- 迁 `pinned_kv_tier`、`nvme_kv_tier` 及测试
- 新增 `kvmem_backend.hpp`：`copy_block_to_host/from_host`、`alloc_slot`、`free_slot` 的虚接口；P0 只给 CPU stub
- 新增 `kvmem_runtime.hpp/.cpp`：
  - `configure` / `register_append` / `prepare_reselect` / `finish_reselect`
  - `finish_reselect` 调用顺序强制 **stage_out → stage_in → resident re-RoPE**
  - P0 的 `finish_reselect` 只更新元数据 + CPU tier，不碰 GPU

**退出：** tier 单测绿；Runtime 在无 backend 时也能跑 recency 选择并产出 plan。

---

### PR P1-1 — 钉 llama.cpp submodule

**标题：** `vendor: pin llama.cpp tag for kvmem integration`

**依赖：** P0-3

**改动：**

- `git submodule add` 钉分支（P1 已钉 `kvmem/prism` — PrismML `prism` `9a9394a` 基线；纯 Transformer + CUDA FA，Qwen3-0.6B Q8_0）
- `scripts/apply-patches.sh`、`scripts/rebase-llama.sh`
- 文档写明 tag、验证过的模型、如何 bump

**退出：** 不打补丁时，上游 `llama-cli` 对 **Qwen3-0.6B Q8_0** greedy 能跑。记录一条 baseline token 序列，供 P1-4 identity 用。

---

### PR P1-2 — Hook 1：memory 工厂 + `llama_memory_kvmem` 骨架

**标题：** `adapter: llama_memory_kvmem slot-pool skeleton`

**依赖：** P1-1

**改动：**

- `patches/0001-memory-factory-hook.patch`
- `src/adapter/llama-memory-kvmem.{h,cpp}`
  - 实现 `llama_memory_i` 最小子集：`init_batch` / `init_full` / `clear` / `seq_*` 先透传到内部一块 **容量 = budget + gen_reserve** 的 `llama_kv_cache`
  - 块槽位分配器：`slot_of_block[block_id]`，空槽 freelist
  - `--kvmem` 关闭时工厂走上游默认，行为与 baseline bit-identical
- `patches/0003-cli-kvmem-flags.patch` 或仅 `tools/llama-kvmem-cli.cpp` 读 flags
- 标志：`--kvmem --kvmem-block-tokens --kvmem-budget --kvmem-sink-tokens --kvmem-recent-tokens`

**禁止：** pack-into-`[0..W)`；改 FA；Hook 2。

**退出：**

- `--kvmem` 关闭：与 P1-1 baseline greedy 文本一致
- `--kvmem` 打开且 budget ≥ 全上下文：先允许「尚未稀疏」的透传路径，文本仍一致（identity 的弱形式；强形式在 P1-3 窗口坐标落地后）
- `git am patches/0001*` 可干净打上；统计补丁行数，核心 llama.cpp 改动 ≪ 300 行

---

### PR P1-3 — recency 窗口 + 位置模型

**标题：** `adapter: recency slot-pool with window positions`

**依赖：** P1-2

**改动：**

- `src/adapter/llama-kvmem-batch.cpp`
  - 注意力半边：`llama_batch.pos` 写成**窗口坐标**
  - 真实位置只写在 `KvMemStore`
  - decode 新 token 追加到「当前尾块」槽的尾部，或新开尾槽
- prefill 超过 budget：`pick_prefill_pressure_blocks`（sink + 最新 tail），`stage_out` 中间块到 CPU tier
- 仍选中的 sink/recent **不得换槽**
- cell 的 pos 设为窗口坐标，让 FA mask 走 llama.cpp 已有的基于 pos 的因果
- 诊断：`KVMEM_TRACE=1` 打 `stage_in/out/skip/gpu_reused_blocks`

**P1 不做 re-RoPE：** recency 压力窗口里，写入时就按当时窗口坐标 bake；被淘汰块丢掉 GPU 副本；复活（P2 才有）再从 raw-K bake。P1 没有 retrieval 复活，中间块被淘汰后不回来。

**退出：**

- identity budget：greedy 文本 = 无 KVMem baseline
- 紧 budget：GPU KV 字节不随 prompt 线性涨（`nvidia-smi` 或 `memory_breakdown`）
- 紧 budget：埋在中间的 needle **召不回**，输出仍连贯（不是乱码）
- 连续两轮高 overlap reselect：`stage_in ≈ 0`，无整窗口 D2D
- 乱序 cell + mask：短序列数值与「物理顺序恰好等于窗口顺序」的对照差在 fp16 噪声内

**Go / No-Go：** 若 FA 忽略 mask、按 cell 下标做因果，只能靠 pack-`[0..W)` 才能正确 → **停**，不要进入 P2。

---

### PR P1-4 — CLI 金丝雀脚本

**标题：** `tools: llama-kvmem-cli and P1 canaries`

**依赖：** P1-3

**改动：**

- `tools/llama-kvmem-cli.cpp`：load / prefill / decode，接 `--kvmem*`
- `scripts/identity_canary.py`
- `scripts/needle_recall.py`（P1 期望 fail-to-recall）
- `scripts/reselect_overlap.py`
- `scripts/long_ctx_vram.py`

**退出：** 三条脚本在文档记载的模型和长度上稳定可复现。P1 作为里程碑打 tag `p1-go`。

---

### PR P2-1 — Hook 2：pre-RoPE K 与 query Q 捕获

**标题：** `patches: attention Q/K capture hook`

**依赖：** P1-4

**改动：**

- `patches/0002-attn-qk-capture-hook.patch`
- `llama-kvmem-hooks.h`：`on_k_prerope(il, t0, t1, ptr)` / `on_q(il, t0, t1, ptr)`
- 固定尺寸 capture buffer，挂进图
- CLI：`--kvmem-query-begin/end`（token 坐标，P5 再从 chat message 推）

**退出：** 小模型上 dump 若干层 pre-RoPE K 与 Q，和手动从权重算的形状/范数对得上；关 hook 时图与 baseline 同构（无多余 cpy 节点）。

---

### PR P2-2 — immutable raw-K + ggml_rope 装配

**标题：** `kvmem: immutable raw-K and ggml_rope assemble`

**依赖：** P2-1

**改动：**

- CPU raw-K 权威（按层、真实 token 位置分块）
- `stage_in`：raw-K → 空槽 + `ggml_rope(to_base)`
- resident `!skip`：同一槽 in-place de-RoPE/re-RoPE
- `skip`：不动
- raw refresh 阈值沿用 qw3 默认（remap 次数 / 累积位移）
- `src/ggml/` 里装配图，backend 与模型共用

**退出：**

- identity 仍成立
- 单测：resident skip 不读 raw-K；cold stage-in 只写目标槽
- 强制多次 remap 后 refresh 路径被触发（trace）

---

### PR P2-3 — mean-K retrieval + query replay（纯注意力）

**标题：** `kvmem: mean-k retrieval and query replay`

**依赖：** P2-2

**改动：**

- ggml 图：content-frame mean-K 索引（从 raw-K 均）
- ggml 图：softmax-over-pages（先一层代表，再多层平均）
- `set_retrieval_scores` → `pick_topk_blocks`
- 纯注意力模型的 query replay：checkpoint KV 在 query 边界 → 按 plan 装配 → 用窗口坐标再 decode query 后缀
- flags：`--kvmem-method retrieval --kvmem-query-conditioned`

**退出：**

- identity budget 仍成立（检索跑着但不改变全选窗口）
- 紧 budget + retrieval：P1 召不回的中间 needle **复活**
- 高 overlap reselect：`gpu_reused_blocks` 主导，`stage_in` 仅新复活块

---

### PR P3-1 — 有界 GPU 池与 prefill 压力淘汰

**标题：** `kvmem: bounded GPU pool and in-prefill offload`

**依赖：** P1-4（可与 P2 后期并行；retrieval 不是前提）

**改动：**

- `maybe_offload_during_prefill`：下一 chunk 放不下就 sink+tail 淘汰
- evict-before-stage-in
- `gpu_memory_ratio` / watermark 配置

**退出：** 小模型上 prompt 32K → 128K，GPU 注意力 KV 字节近似常数（随 budget，不随 T）。更长阶梯只作成熟碑抽检。

---

### PR P3-2 — NVMe 与异步 stage

**标题：** `kvmem: NVMe tier and async stage-in`

**依赖：** P3-1

**改动：**

- 接上已迁的 `NvmeKvTier`
- prepare 发 prefetch、finish 等待（CPU 先同步也对；再加后台 read）
- `--kvmem-cpu-gb --kvmem-nvme-dir --kvmem-nvme-gb`

**退出：** CPU 满后 NVMe 有 `stage_out`/`stage_in` trace；512K+ 不 OOM。

---

### PR P4-1 — hybrid memory 组合

**标题：** `adapter: wrap llama_memory_hybrid with kvmem attn cache`

**依赖：** P2-3、P3-1

**改动：**

- attn 半边 = KVMem 槽位 cache；recr 半边 = 原样 `llama_memory_recurrent`
- 注意力用窗口坐标，GDN 用真实顺序
- 关掉 MTP
- 压力 prefill 不得回滚 GDN 状态

**退出：** **Qwen3.5-0.8B** 短上下文 identity（budget 全覆盖）与裸 llama.cpp 一致。27B 不作为本 PR 门禁。

---

### PR P4-2 — hybrid query replay

**标题：** `adapter: hybrid query-boundary checkpoint and replay`

**依赖：** P4-1

**改动：**

- query 边界 `llama_state_seq_*` checkpoint recurrent
- replay 只重跑 query 后缀，不重放历史 GDN
- 与 qw3 `kvmem_recompute_query` 同构

**退出：** **Qwen3.5-0.8B** 上 needle 的 block 集合 + query replay 正确（GDN 只从 query 边界续）。Qwen3.5-4B 作里程碑抽检。Qwen3.8-27B 仅手动发布抽样。这是 v1.5 里程碑。

---

### PR P5 — serving

**标题：** `tools: llama-kvmem-server (OpenAI-compatible, single slot)`

**依赖：** P2-3；hybrid 模型需要 P4-2

**改动：**

- **默认独立进程**，不改 llama-server 核心（更容易保住 qw3 的 harness 字段）
- 从 chat 最后一条 user message 推 query span
- 支持 pin / mandatory span（harness 层，不进 ggml）
- 单 slot，无连续 batching

**退出：** `/v1/chat/completions` greedy + streaming 冒烟；query-conditioned 请求能打出 retrieval trace。

---

### PR P6 — 多硬件冒烟

**标题：** `ci: metal/vulkan identity and needle smoke`

**依赖：** P2-3

**改动：**

- 同一套脚本在 Metal 或 Vulkan 上跑 identity + needle
- 检索走 ggml，不走 CUDA kernel

**退出：** 至少一种非 CUDA backend 通过 identity；retrieval 若无 GPU 图则允许 CPU ggml（慢但正确）。

---

### PR P7 — KVMem + MTP 槽位条带（方案 B）

详细阶段、工厂行为、lockstep 约束、退出标准见 **[kvmem-mtp-plan.md](kvmem-mtp-plan.md)**。此处只留总表。

| PR | 标题 | 依赖 | 退出一句话 |
|---|---|---|---|
| P7-0 | 接 `draft-mtp`；verify 时 snapshot GDN | P4-2、P5 | 短 ctx 无 retrieval，0.8B `n_max=2` 能投机且 reject 可回滚 |
| P7-1 | `llama_memory_kvmem_mtp` lockstep 池 | P7-0 | MTP `n_kv` == 主池；27B MTP KV 为池子量级 |
| P7-2 | retrieval / query replay 同步 MTP 条带 | P7-1 | 0.8B retrieval + MTP 关思考后召回 BLUEBIRD-42 |
| P7-3 | 长 ctx 两池都不随 T 涨 | P7-2 | 27B 固定 budget，主 KV + MTP KV 随 budget 不随 T |

不搬 qw3 的 `generate_mtp`。独立 `MTP/mtp-*.gguf` 可选，主 Unsloth 27B GGUF 已焊 nextn。

---

## 8. 测试策略与模型

**大部分测试不需要真实模型。** 需要模型时，日常 CI 只用小 GGUF（≤2B，文件几百 MB 到 2GB）。27B 只做可选发布门，不进每次 PR。

### 8.1 三层测试

| 层 | 要不要权重 | 跑什么 | 频率 |
|---|---|---|---|
| **A. Host 单测** | 否 | `KvMemStore`、plan diff、tier 槽位、sink/recent 规则 | 每个 PR、每次 commit |
| **B. 小模型金丝雀** | 要，小 GGUF | identity / needle 机制 / overlap / 乱序 mask / 有界显存 | 每个涉及 adapter 的 PR |
| **C. 发布抽样** | 要，中大模型 | 同一协议在 4B/9B 或 27B 上再跑一遍 | 里程碑 tag，不进 PR CI |

A 就能拦住选择语义和「skip 块被搬走」。B 才需要 llama.cpp forward：证明补丁没把图弄坏、FA mask 对、装配后模型真的看不见被淘汰块。C 只确认小模型上成立的机制在大模型上没有规模 bug。

小模型做不好「开放问答质量」，**够做机制断言**：

- identity：greedy 文本是否与裸 llama.cpp 一致（与参数量无关）
- recency needle：看 **选中的 block_id 集合** 是否不含中间针；生成是否含 codeword 只作辅证（0.8B 可能胡写）
- retrieval：看 needle 所在 block 是否进入 `pick_topk_blocks`；生成作辅证
- VRAM：看注意力 KV 字节是否随 budget 而不是随 T（小模型绝对值小，**斜率**一样）

### 8.2 下载策略（硬性）

1. **只用 Unsloth 量化的 GGUF**（含普通 `Q8_0` / `Q4_K_M` 和 `UD-*` Dynamic）。禁止 bartowski、ggml-org、自量化、ik_llama 专用类型。
2. **只从 ModelScope 的 `unsloth/...` 仓下**，国内带宽。禁止默认走 Hugging Face。
3. 只拉需要的单个文件，不要把整个 GGUF 仓 clone 下来。
4. 落到仓库外的 `models/`（gitignore），脚本用 `KVMEM_TEST_MODEL` 指向绝对路径。
5. **GPU 绑定（本机实测）：**
   - GPU 0 `NVIDIA GeForce RTX 5050 Laptop GPU` 8151 MiB → **所有 < 27B 的测试**
   - GPU 1 `NVIDIA GeForce RTX 5090 Laptop GPU` 24463 MiB → **仅 27B**
   - 脚本：`source scripts/gpu.sh small` 或 `source scripts/gpu.sh 27b`（设置 `CUDA_VISIBLE_DEVICES`）。禁止 27B 出现在 GPU 0。

日常 identity 优先 **Unsloth `Q8_0`**（量化噪声小）。里程碑 / 显存紧用 **Unsloth `Q4_K_M`**；若该仓只有 `UD-Q4_K_M` / `UD-Q4_K_XL`，用 UD 4-bit，不要换成别家。

下载（`pip install modelscope`）：

```bash
# 示例：P1 日常
modelscope download \
  --model unsloth/Qwen3-0.6B-GGUF \
  --include 'Qwen3-0.6B-Q8_0.gguf' \
  --local_dir models/unsloth/Qwen3-0.6B-GGUF
```

Python 等价：

```python
from modelscope import snapshot_download
snapshot_download(
    "unsloth/Qwen3-0.6B-GGUF",
    allow_patterns=["Qwen3-0.6B-Q8_0.gguf"],
    local_dir="models/unsloth/Qwen3-0.6B-GGUF",
)
```

P0 应附 `scripts/download-test-models.sh`，按阶段拉下表文件。ModelScope 上 Unsloth 组织：<https://modelscope.cn/organization/unsloth>。某新模型尚未同步时，等 Unsloth 推到 ModelScope，**不要改下 Hugging Face**。

### 8.3 分阶段固定模型

| 阶段 | 角色 | ModelScope 仓 | 文件 | 约大小 | GPU | 进 CI？ |
|---|---|---|---|---|---|---|
| **P0** | 无引擎 | — | — | — | — | 是 |
| **P1 / P2 / P3 日常** | 纯 Transformer | [`unsloth/Qwen3-0.6B-GGUF`](https://www.modelscope.cn/unsloth/Qwen3-0.6B-GGUF) | `Qwen3-0.6B-Q8_0.gguf` | ~0.64 GB | **5050 / GPU 0** | 是 |
| **P1 备用** | 0.6B 不够用 | [`unsloth/Qwen3-1.7B-GGUF`](https://modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF) | `Qwen3-1.7B-Q4_K_M.gguf` | ~1.2 GB | **5050 / GPU 0** | 否 |
| **P2 质量抽检** | 生成召回辅证 | [`unsloth/Qwen3-4B-GGUF`](https://modelscope.cn/models/unsloth/Qwen3-4B-GGUF) | `Qwen3-4B-Q4_K_M.gguf` | ~2.5 GB | **5050 / GPU 0** | 里程碑 |
| **P4 日常** | hybrid / GDN | [`unsloth/Qwen3.5-0.8B-GGUF`](https://modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF) | `Qwen3.5-0.8B-Q8_0.gguf` | ~0.81 GB | **5050 / GPU 0** | 是 |
| **P4 质量抽检** | hybrid 稍稳 | [`unsloth/Qwen3.5-4B-GGUF`](https://modelscope.cn/models/unsloth/Qwen3.5-4B-GGUF) | `Qwen3.5-4B-Q4_K_M.gguf` | ~3 GB | **5050 / GPU 0** | 里程碑 |
| **P4 发布可选** | 产品同款 | [`unsloth/Qwen3.8-27B-GGUF`](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | **`Qwen3.8-27B-UD-Q4_K_M.gguf`** | ~16.5 GB | **5090 / GPU 1** | 手动 |
| **P5 / P6** | serving / 非 CUDA | 与 P4 / P1 日常相同 | 同上 | 小 | 5050 / GPU 0 | 是 |

协议仍是：P1 全文 2K–8K、紧 budget 256–512、块 64 或 128；P3 合成 32K→128K。Qwen3-0.6B 非 hybrid；Qwen3.5-0.8B 已是 `qwen35`。

**明确不下载：** bartowski / ggml-org GGUF、mmproj、MTP 独立 draft、Flash-Next、MoE 35B、ik_llama 专用 quant、整仓 BF16。

### 8.4 金丝雀怎么写（小模型友好）

所有 B 层脚本默认读环境变量 `KVMEM_TEST_MODEL`，CI 指到 0.6B/0.8B 路径。

1. **identity_canary.py**  
   greedy、`temp=0`、短 prompt（几十 token）、`n=32`。KVMem 关 vs 开且 budget≥全文。比的是**解码 token id**，不是「回答聪不聪明」。

2. **needle_blocks.py（主断言）**  
   合成上下文：`sink 文本 + 填充 + 「口令 ZEPHYR-7 只出现一次」+ 填充 + 问题`。  
   紧 budget 只够 sink+recent。  
   **硬断言：** recency 下 needle 的 `block_id` ∉ selected；retrieval 下 ∈ selected。  
   软断言：生成里是否出现 `ZEPHYR-7`（0.6B/0.8B 失败只记 warning）。

3. **reselect_overlap.py**  
   两轮几乎相同的选择。硬断言 `stage_in==0` 且无整窗口拷贝。与模型智商无关。

4. **vram_slope.py**  
   同一小模型，prompt 8K vs 32K vs 64K，budget 固定。硬断言 GPU 注意力 KV 字节变化 ≪ 按全量 ctx 线性估算。

### 8.5 验收矩阵

| 用例 | 阶段 | 模型 | 期望 |
|---|---|---|---|
| host `KvMemStore` 选择 / diff | P0 | 无 | 与 qw3 单测一致 |
| KVMem 关闭 | P1-2 | Qwen3-0.6B | 与上游 greedy token 一致 |
| identity budget | P1-3 起 | Qwen3-0.6B | 同上 |
| 紧 budget + recency | P1-3 | Qwen3-0.6B | needle **block 不在** selected；输出非崩溃 |
| 高 overlap reselect | P1-3 | Qwen3-0.6B | `stage_in≈0`，无整窗口 D2D |
| 乱序 cell + mask | P1-3 | Qwen3-0.6B | 与顺序槽对照只差 fp16 噪声 |
| GPU KV vs T | P1-3 / P3 | Qwen3-0.6B | 随 budget，不随 T |
| retrieval 复活 | P2-3 | Qwen3-0.6B | needle **block 在** selected |
| raw-K refresh | P2-2 | Qwen3-0.6B | trace 可见，不换槽 |
| hybrid identity | P4-1 | **Qwen3.5-0.8B** | 与裸 llama.cpp greedy 一致 |
| hybrid replay + needle | P4-2 | Qwen3.5-0.8B | block 集合正确；GDN 未重放整段历史 |
| 27B 抽样 | 里程碑 | Qwen3.8-27B Q4_K_M | 短 identity + 一条 needle，手动 |
| MTP 短 ctx 投机 | P7-0 | Qwen3.5-0.8B | `draft-mtp` greedy；GDN reject restore |
| MTP `n_kv` == 主池 | P7-1 | 0.8B；27B 抽检 | 随 budget，不随 `-c` |
| retrieval + MTP needle | P7-2 | Qwen3.5-0.8B | 关思考后 BLUEBIRD-42 |
| 长 ctx 两池封顶 | P7-3 | Qwen3.8-27B | 主 KV + MTP KV 随 budget |
| bump llama.cpp tag | 每次升级 | 日常小模型 | `git am` + 金丝雀绿 |

性能对照（记录，不当 P1 门禁）：

- 小模型 decode tok/s vs 同卡 llama.cpp `ctx=budget`
- 不把 qw3-native Blackwell 27B 峰值当 CI 门禁

---

## 9. 默认决议（原开放问题）

实施按下面默认走，避免 P0 卡着：

| 问题 | 默认 |
|---|---|
| v1 是否必须 Qwen3.6 hybrid | 否。P1/P2 纯 Transformer；hybrid 是 P4 / v1.5 |
| serving | 独立进程（P5），先不改 llama-server |
| 是否立刻回接 qw3-native | 否。公共库主树在本仓库；qw3 回接另开任务 |
| llama.cpp tag | P1 钉能跑 **Qwen3-0.6B** 的稳定 b 号；P4 再确认同一 tag（或一次 bump）能跑 **Qwen3.5-0.8B**。不追 master，不为 27B 单独锁 CI tag |
| KVMem + MTP | **方案 B**（MTP 进同一槽位池）。投机用 llama.cpp `draft-mtp`，不自研。见 [kvmem-mtp-plan.md](kvmem-mtp-plan.md) |
| 测试权重从哪下 | **只用 Unsloth GGUF，只从 ModelScope `unsloth/...` 拉**。日常 `Q8_0`，抽检 `Q4_K_M`/`UD-Q4_K_*`。禁止 Hugging Face 默认源 |
| 用哪张卡 | **<27B → RTX 5050（CUDA 0）**；**27B → RTX 5090（CUDA 1）**。`source scripts/gpu.sh small\|27b` |

---

## 10. 时间（粗估，单人全职）

| 阶段 | 周期 | 里程碑 |
|---|---|---|
| P0 | 1–2 周 | 可测的独立库 |
| P1 | 2–4 周 | **go/no-go**：槽位池 + recency 在 llama.cpp 上成立 |
| P2 | 3–6 周 | retrieval + immutable-K |
| P3 | 2–4 周 | 长上下文显存封顶 |
| P4 | 3–6 周 | Qwen3.6 hybrid v1.5 |
| P5 | 2–3 周 | 可对 harness 的单 slot 服务 |
| P6 | 1–2 周 | 非 CUDA 冒烟 |
| P7 | 3–5 周 | KVMem + MTP 方案 B（槽位条带） |

P1 失败则停止扩检索，先修位置/mask 模型。不要用 P2 掩盖 P1 的接口问题。

---

## 11. 风险与对应修改策略

| 风险 | 计划内对策 |
|---|---|
| `llama_memory_i` 不是公共 API | adapter 薄；补丁队列；冲突只修 hook |
| `batch.pos` 撒谎打到 SWA/shift | v1 禁用这些模型；hybrid 两套坐标 |
| FA 不能用 mask 表达乱序因果 | P1 no-go；不偷偷改成全量 pack 当主路径 |
| capture 破坏图复用 | 固定 shape buffer |
| q8 KV 无法 de-RoPE | raw-K 权威，P2 起独立存 |
| qw3 与公共库分叉 | 主树在 `kvmem/`；qw3 回接另开 |
| 补丁膨胀 | P1 核心 llama.cpp 改动硬顶 ~300 行 |

---

## 12. 实施时的代码评审检查单

每个涉及装配的 PR 必须能回答：

- [ ] 本轮 `skip` 的块有没有任何 D2D/H2D？
- [ ] 新块是否只进空槽，有没有挤压邻居？
- [ ] `n_kv` 是否仍是 budget 量级，而不是逻辑 ctx？
- [ ] 有没有改 FA kernel？
- [ ] `kvmem/` 里有没有 `#include "llama-*.h"`？
- [ ] identity 金丝雀过了没有？
- [ ] MTP 若开启：`n_kv` 是否仍是 budget 量级？主干与 MTP 是否同一 slot / 同一 pos？

---

## 13. PR 总表

| PR | 标题 | 依赖 | 关键路径 |
|---|---|---|---|
| P0-1 | scaffold kvmem library and cmake | — | 构建 |
| P0-2 | import host block store | P0-1 | `kvmem/src/host/kvmem_store.cpp` |
| P0-3 | CPU/NVMe tiers + KvMemRuntime | P0-2 | `kvmem_runtime.cpp` |
| P1-1 | pin llama.cpp tag | P0-3 | `llama.cpp/` submodule |
| P1-2 | memory factory + slot-pool skeleton | P1-1 | `patches/0001`, `llama-memory-kvmem.cpp` |
| P1-3 | recency window positions | P1-2 | `llama-kvmem-batch.cpp` |
| P1-4 | llama-kvmem-cli + canaries | P1-3 | `tools/`, `scripts/` |
| P2-1 | Q/K capture hook | P1-4 | `patches/0002` |
| P2-2 | immutable raw-K + ggml_rope | P2-1 | `kvmem/src/ggml/` |
| P2-3 | mean-k retrieval + query replay | P2-2 | runtime + ggml score graph |
| P3-1 | bounded GPU pool + in-prefill offload | P1-4 | runtime |
| P3-2 | NVMe + async stage | P3-1 | `nvme_kv_tier` wiring |
| P4-1 | hybrid attn=kvmem, recr=stock | P2-3, P3-1 | adapter |
| P4-2 | hybrid query replay | P4-1 | adapter + state_seq |
| P5 | independent OpenAI server | P2-3（hybrid 需 P4-2） | `tools/llama-kvmem-server.cpp` |
| P6 | Metal/Vulkan smoke | P2-3 | scripts / CI |
| P7-0 | wire draft-mtp + GDN verify snapshot | P4-2, P5 | tools + hybrid snapshot |
| P7-1 | MTP lockstep slot-pool | P7-0 | `llama-memory-kvmem-mtp.cpp` |
| P7-2 | MTP follows retrieval/replay | P7-1 | adapter |
| P7-3 | long-ctx MTP VRAM canary | P7-2 | scripts |

第一刀从 **P0-1** 开始。P1-4 打上 `p1-go` 之前，不把 retrieval、hybrid、server 混进主路径。P7 在 P4-2 之后单独开，不回头改 FA。
