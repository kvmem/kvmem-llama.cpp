# Query replay：执行策略与验证

KVMem 在检索改变可见历史时，需要恢复 recurrent/MTP 状态并重新处理受影响的输入。当前实现会检查历史视图是否保持有效，在满足条件时省去第二遍计算；同一用户问题下的工具续接还可以复用已捕获的查询特征。

实现位于 [host runtime](../kvmem/src/host/kvmem_runtime.cpp)、[内存适配层](../src/adapter/llama-memory-kvmem.cpp) 和[多模态请求执行器](../tools/llama-kvmem-driver.cpp)，继续复用 llama.cpp 的 decode、MTP、聊天模板和 mtmd 能力。

## 已实现的执行路径

| 日志中的 path | 条件与行为 |
|---|---|
| `all_resident` | 全历史 KV 完整驻留，预算、布局和请求生成空间均满足要求；必要后缀只算一次，提交驻留选择元数据 |
| `unchanged_selection` | 探测期间旧历史没有变化，选中块与真实可见视图一致；保留第一遍结果 |
| `keep_selected` | 同一真实用户问题、相同因果前缀和媒体身份，且空间够；导入并冻结 Q，追加工具后缀 |
| `cached_q_reselect` | Q 可复用但需要释放空间；先重选、再单遍追加，并保护 Q 捕获之后已经读入的工具后缀 |
| `query_replay` | 选择或历史视图变化，或无法证明可以跳过；恢复匹配检查点，重算完整依赖后缀 |
| `legacy` | 兼容入口或没有需要检索的 query 后缀；普通解码仍保留刷新 logits 必需的计算 |

Q 特征、mandatory 和 replay 起点已经解耦。512 token 限制只约束特征，不能用它截短需要失效和重算的 KV 后缀。主模型和 MTP 使用各自实际的 Q8/Q5/Q4/F16 KV 字节类型。

adapter 核对每个块的全部有效 cell，包括逻辑行、时序/空间位置、主/草稿覆盖、部分尾块以及多余的未截断 cell；不仅检查块首或 packed 副本。选择计划记录历史版本和逻辑行数。截断、覆盖、换入换出、移动、状态恢复及草稿历史变动会使旧证明失效；正常追加不会仅因总行数增长就使证明失效。首版的等价跳过要求规范布局，无法证明时继续 replay。

选择预览复用现有评分器和选块器，不改变 slot、tier、working-set/baked 元数据或启动预取。提交接受确定的块 ID，不再重选一次。快速驻留提交只更新工作集标记，不伪造 KV 的 baked 位置。

## 检查点与 Q 生命周期

检查点的 recurrent、MTP carry、尾块 mean-K 合并为不可变共享 payload。检查点选择、rollback 和 query 检查点只复制句柄；仍只保留至多四个历史条目。live payload 身份和行边界同时匹配才省略序列化/恢复，不能只看行号。每次主模型计算和生成开始都会使 live 标记失效。

Q 缓存包含分层 sum/count、用户问题内容和位置、force 参数、原生媒体 ID 以及因果前缀。原生模板另外渲染截至真实最后用户消息的前缀，与完整请求逐字验证后，用原生消息分段映射到图片展开后的行号；工具结果的 user 风格包装不会被当成真实用户。图片问题优先使用最后图片之后的文字。纯图片、文字只在图片之前、无法精确定位的模板和含媒体的显式区间使用有界后缀特征，并标为 `bootstrap_suffix`，不冒充可复用的用户 Q。

Q 导入前校验层/头尺寸、非空且一致的有效层计数和有限数值。发布缓存前等待 capture 完成，并确认计数等于真实特征长度。工具续接冻结 Q，不向旧 Q 累加工具文本；mean-K、packed KV 和 MTP 同步继续维护。失败、取消和无法验证的回滚使 Q 缓存失效，下一请求走正常回退；缺少 recurrent 检查点仍按 cache miss 处理，保留 dsh 修复。

`user` 模式下，无视觉头的文本请求也使用同一套已支持文本/媒体的缓存执行器。旧文本执行器保留为 `legacy` 兼容入口，避免两套缓存状态在启用视觉头的会话中来回切换。

**稀疏质量约束：** 实验发现，只凭旧问题的 Q 重选会丢掉后来工具文件中的事实，原版能通过的两文件召回曾出现回归。因此缓存 Q 重选必须保留 `[user_query_end, restored_base)` 的已读后缀、部分追加尾块以及原有 sink/media/force 约束。若选择器因预算裁剪了这些必保块，或者所选历史加追加/生成空间放不下，快速重选被拒绝，改走新的后缀探测及完整依赖 replay。这是基于实测加入的质量保护，没有偷偷增加检索预算或改变 `recent_tokens=0` 的含义。

## 验证范围

- Host 测试覆盖选块预览无副作用、拒绝不完整驻留提交及 pending plan。
- [mtp-kv-test.cpp](../tests/mtp-kv-test.cpp) 覆盖 KV 字节、部分块、中间缺洞、过期计划和 GPU logits 对照；需要带 MTP 的 GGUF 和 CUDA，未提供模型时 CTest 按返回码 77 跳过。
- 多模态回归覆盖 CPU/GPU 视觉头、普通/MTP 解码、纯图片、文字在图片前、重复请求及缓存续接。
- 稀疏工具测试检查已读文件事实的保留，以及空间不足时回退到完整依赖重算。

2026-09-14 的固定样本验收中，七种主/草稿 KV 组合通过；Q8/Q5 单遍与 replay 的完整 logits RMSE/maxabs 均为 0。稀疏策略另外使用文件事实和图片回答验证，不能用单个 logits 样本推断所有请求的质量。

```bash
ctest --test-dir build --output-on-failure
build/bin/kvmem-mtp-kv-test /path/model-mtp.gguf
```

历史对照使用约 12K 前缀和三轮工具结果，两轮实验中工具续接 prefill 合计中位数从 14.896 秒降至 6.466 秒，五个响应逐字一致。这组数据使用当时的 40000/16000 预算，当前推荐配置与完整任务结果见[性能文档](recommended-config-performance.md)。原始记录本地保存在 `logs/query_replay_20260914/`，不随仓库分发。

## 使用与回退

IQ3/IQ4 默认启动脚本选择 `--kvmem-query-replay auto --kvmem-query-policy user`，模型、KV 类型、MTP 2、采样、预算及 CPU/GPU 视觉头选项保留。直接运行服务二进制时默认仍是 `auto + legacy policy`，给其他模板保留兼容入口。无 projector 且使用 legacy policy 时，保留旧文本执行器及其原有 warm-skip 规则；新的执行决策用于媒体执行器或显式 user policy。

```bash
scripts/start-iq3.sh
scripts/start-iq4.sh

# 只保留经证明的 replay 跳过，使用先前 query 策略：
KVMEM_QUERY_POLICY=legacy scripts/start-iq3.sh

# 兼容执行策略；共享检查点仍启用：
KVMEM_QUERY_REPLAY=legacy KVMEM_QUERY_POLICY=legacy scripts/start-iq3.sh
```

替换正在运行的服务时按原脚本用 `--restart`。`query-replay=legacy` 只控制探测后是否跳过；若同时选择 `query-policy=user`，同问题 Q 续接仍可单遍执行。要回到完整旧策略必须同时选两个 `legacy`。

复现实机序列：

```bash
python3 scripts/multimodal_canary.py --query-policy user --draft-kv f16 \
  --image-max-tokens 512 --folder logs/query-replay-full
python3 scripts/multimodal_canary.py --query-policy user --draft-kv f16 \
  --budget 2048 --reserve 2048 --ctx 16384 --long-words 6000 \
  --query-quality --folder logs/query-replay-quality
```

## 性能诊断

默认日志 `KVMEM_PREFILL_PERF` 分别报告首遍、历史重算、检索和检查点时间；各阶段聚合时避免重复计入嵌套操作。`KVMEM_CHECKPOINT_MEMORY` 区分按引用计费的字节和共享 payload 的实际字节数。

`KVMEM_PERF=1` 启用主模型/MTP 分时及适配层复制统计，会增加同步完成点。复制统计不包含 llama.cpp 内部的模型权重、hidden 和视觉编码传输，不能视为全进程 PCIe 流量。

当前实现未启用探测遍延后 MTP 或 Q 的 GPU 归约，也未增加生成中途任意切换窗口的能力；生成预留约束继续生效。
