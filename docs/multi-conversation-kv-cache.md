# Multi-conversation KV cache

`--kvmem-conversations N` lets the server hold N conversations' KV in host RAM
at the same time and switch between them. With the flag absent, or set to `1`,
the server behaves exactly as before: a request that does not continue the
stored conversation discards it. Host RAM is the abundant resource and GPU VRAM
is the scarce one, so the design is one `llama_context`, one GPU working set,
N host stores, time-multiplexed. Requests are still served one at a time.

There is no client API change. A conversation is identified by the token
sequence the client sends, so clients that send nothing extra work unchanged.
`{"kvmem": {"conversation_id": "..."}}` in the request body is honored as an
optimization when a client does send it, and is ignored when `N` is 1. It can
never fail a request: the key was unrecognized before this feature existed, so
a value the server cannot use as a key (not a string, empty, over 128 bytes, or
carrying a byte outside printable ASCII) is dropped and the request is matched
by its token prefix instead.

## Decision table

Every request maps to one conversation before prefill. The policy is a pure
function in [`tools/kvmem-conversation-store.h`](../tools/kvmem-conversation-store.h);
the reason it reports appears in a `KVMEM_TRACE store_select` line.

| Reason | Condition | Behavior |
|---|---|---|
| `extend` | The prompt continues a stored conversation, and a recurrent checkpoint exists at or before the match point | Attach that store and resume from the checkpoint |
| `client_id` | Same as `extend`, reached through a `kvmem.conversation_id` the client supplied | Attach that store; the id only skips the prefix comparisons |
| `new_conversation` | No store is live | Fresh store |
| `no_recurrent_checkpoint` | The prefix matches, but no recurrent checkpoint sits at or before the match point | Fresh store. This is a hard constraint of the hybrid model, not a policy choice |
| `fork_zero_keep` | A checkpoint exists but only at row 0, so there is nothing to resume | Fresh store, leaving the matched store intact |
| `fork_shared_prefix` | The match covers only a small part of the matched store, for example a shared system prompt or chat template | Fresh store, so the matched store's longer tail is not truncated |

With `N` = 1 none of this runs. `conversation_begin_request` returns before the
policy, so the request reaches the same prefill path, with the same checkpoint
selection, that it reached at v0.16.0-rc3: default behavior is identical by
construction rather than because two expressions agree. In particular the
continuation rule is a multi-store rule. The single-store prefill path has no
such gate, and applying it there would change default behavior in exactly the
`fork_shared_prefix` configuration, where today's server attaches at a low
checkpoint and truncates the stored tail.

## The three formulas

| Formula | Source |
|---|---|
| `keep_cap = min({lcp, live_row, eval_end - (spec_ok ? 0 : 1)})` | [`tools/kvmem-multimodal-server.h`](../tools/kvmem-multimodal-server.h), `keep` in `run_prefill_multimodal` |
| `found, keep` = the largest checkpoint row `<= keep_cap` | [`tools/kvmem-multimodal-server.h`](../tools/kvmem-multimodal-server.h), the `st.mm_checkpoints` scan that follows it |
| `continuation = (rows - lcp) <= max(0, last_n_gen) + 64` | [`tools/llama-kvmem-server.cpp`](../tools/llama-kvmem-server.cpp), `suffix_slack` in `run_prefill_retrieval` |

`lcp` is the longest common prefix of the incoming prompt against a stored
conversation, computed with the media-aware `kvmem_prompt::common_prefix`, so
an image counts as the rows it expands to. Candidates are ranked by `keep`, not
by `lcp`: `keep` is the prefix the hybrid model can actually restore, and `lcp`
is only aspiration. Ties break toward the more recently used store.

The continuation bound is absolute rather than proportional, and it is the
existing continuation gate rather than a new tunable. A prompt matching 90
percent of a 100k-token store is a branch of that conversation, not its
continuation, and truncating the remaining 10k tokens is the destruction this
feature exists to prevent. The bound encodes the two legitimate ways a
continuation's tail shrinks: the client dropped the last assistant turn
(bounded by `last_n_gen`) and re-templating moved a few prompt-side tokens
(the 64).

## Host bytes, and what the cap does

`--kvmem-conversations-gb GB` caps the accounted host bytes summed over the
live stores. It only caps stores `--kvmem-conversations N` creates, so passing
it without `N > 1` is rejected at startup rather than silently ignored. Per
store the number is `RawKvStore::bytes_k() + bytes_v()`
([`kvmem/src/host/raw_kv_store.cpp`](../kvmem/src/host/raw_kv_store.cpp):1045
and :1073) plus the server-side recurrent checkpoints and the cached token
vector. `bytes_k()` sums, for every block and layer, the raw K rows, the packed
GPU-format K, the F32 mean sums and the NVMe tail; `bytes_v()` does the same
for V. So it is accounted store bytes, not pure anonymous host RAM, and it does
not include the pinned `cpu_arena_` that `--kvmem-cpu-gb` allocates per store.

Two properties to read literally:

- The cap evicts the least recently used **inactive** store. Neither the
  conversation the current request maps to nor the one still attached to the
  GPU working set is evicted while the request is being planned, so a single
  conversation larger than the cap still runs and the server warns once. That
  reproduces today's uncapped behavior instead of failing a request the server
  would otherwise serve. The flag is therefore a bound on the sum of the
  inactive stores, not an OOM guard. Eviction runs after the reply, when the
  outgoing conversation is parked and evictable again, so the cap converges at
  the next turn that commits rather than inside the request that first exceeded
  it. `conversation_commit` runs only from `commit_cached`, so a turn that
  fails after the mapping does not re-measure at all and the overage stays
  until some later turn completes.
- `0` means no byte cap; the count cap is then the only bound.

Both functions take the store mutex and walk blocks times layers, so each
conversation's figure is measured once, at that conversation's commit, and
cached in the store table. Selection reads the cached figure, never the walk:
only the attached conversation's footprint can have moved since its own commit,
and that is the one the commit refreshes.

## Quiesce ordering across a swap

A switch drains the whole GPU working set to host and rebuilds the incoming
store's, reusing the drain-and-restage machinery that already runs inside a
conversation (`layout_gpu_slots_by_orig_pos`'s host fallback and
`write_block_to_gpu` in
[`src/adapter/llama-memory-kvmem.cpp`](../src/adapter/llama-memory-kvmem.cpp)).
The detach order is not guessable from the call sites, so it is stated here as
well as in the code:

1. `kvmem_stagein_flush_sync` — the stage-in slab is a process global holding
   H2D packed K/V aimed at cells the outgoing store still owns.
2. `decode_mean_reset` — flush, then discard; the flush writes the partial
   block's running mean into the outgoing store's host mirror, and the reset
   also forces the incoming conversation to zero the process-global mean-K
   accumulator.
3. `harvest_flush` — waits for the harvest worker queue and both
   `CaptureD2hPipe` slots. That wait is also the release/acquire edge for the
   worker's last write to the host mirror and the query accumulator, so it must
   happen even when the pipe looks idle. The pipe and the worker are drained,
   never freed or stopped: their buffers are engine-sized and identical for
   every conversation.
4. `harvest_gpu_v_commit` — drains the stage-out slab into the host mirror.
5. `RawKvStore::wait_writes` — `harvest_gpu_v_commit` does not wait for the
   store's own writes. The three pre-existing call sites get away with that
   only because each ends in a host-mirror mutation that waits first; a detach
   mutates nothing, so it waits explicitly.
6. Drain every GPU-resident block to host, release its slot, empty the
   attention cells, rebuild the free-slot list and bump the attention epoch so
   server-held selection proofs cannot validate across the swap.

The MTP follower's host mirror moves in lockstep with the trunk's, one
statement apart, the way `truncate_cached` already drives the follower. Its
draft cells are emptied by the detach but never zeroed, so each draft slot
counts as holding the previous conversation's packed draft K until a full
block of this one's has been written over it. The per-slot marks are sized
from the target's own slot count (`llama_memory_kvmem::n_slots()`, which is
`ceil(kv_size / block_tokens)`) and allocated once when the follower is built,
so every slot the trunk can hand out is covered and no swap allocates. What
makes the highest index safe is that rounding, not the copied `kv_size`:
sizing the marks with a truncating division left the top slot unmarked
whenever the pool is not a whole number of blocks. A block the mirror cannot
cover is then left out of the follower's window rather than labelled with this
conversation's positions, for the whole of the request and not only for the
attach: the draft has no context there, which costs accept rate and nothing
else, because the target model verifies every drafted token.
The recurrent (GDN) half is not switched by the adapter: it is a server-side
byte snapshot restored per request, and `mm_live_checkpoint` is cleared after
every switch so the live short-circuit in `multimodal_restore` cannot skip that
restore and decode one conversation against another's SSM state.

A swap is refused, with a warning and a fallback to clearing the outgoing
store, when the outgoing store cannot be drained safely: a query replay in
flight, a GDN recording in progress, a prepared-but-unapplied runtime plan, or
a host mirror the full-`seq_rm` divergence below left stale. The outgoing
conversation then loses its KV, and the bool the switch returns describes the
incoming store only, so the server does not trust it: after every switch it
cross-checks `llama_kvmem_store_rows` on the handle it just parked and drops
that conversation's payload if the store no longer holds its rows. Otherwise
the payload would name a checkpoint the next request resumes against KV that
was never restored. For the same reason attaching a store that holds no rows
reports false, whether it was empty all along or had to be reset, which is the
one meaning the bool carries on every path.

Exchanging the two bundles is all-or-nothing, and the handover has exactly one
window. `swap_conv` can throw only before it: out of the drain, or out of the
clear a store that cannot be drained safely gets. Both leave the outgoing store
on the memory object and the incoming bundle in the caller's handle, so
`llama_kvmem_store_switch`'s catch is right to leave both handles and its own
active id where they were. Past the handover nothing throws: the attach's
repair path is `noexcept`, so a failed attach leaves the incoming store
attached and reset to empty and reports that as false, which the server reads
as an ordinary cache miss. That matters more than it sounds: if the repair
could throw, the catch would skip a handover the memory object had already
made, and a later switch would attach one conversation's KV under the other's
payload. Nothing propagates across the `LLAMA_API` boundary either, because
`httplib` would swallow it and leave the server decoding against a store its
own bookkeeping no longer describes.

The feature also
refuses to arm without flash attention, because V is only mirrored to host when
flash attention is on and a drained conversation could not otherwise be
restaged. That cannot be checked at parse time, since the flash-attention mode
is resolved inside `llama_init_from_model`; the server asks the adapter after
the context exists and falls back to one store with a warning.

## Non-goals

- No concurrent decoding. `total_slots` stays 1 and `-np` / `--parallel` still
  accepts only 1. Sequence parallelism and conversation multiplexing are
  different axes: `-np` asks for concurrent decoding across `llama_seq_id`s,
  `--kvmem-conversations` asks for serialized time-sharing of one sequence slot
  across N host stores.
- No change to `n_seq_max` or to `llama_seq_id` semantics.
- No cross-store block sharing. A fork does not inherit the shared prefix, so
  the first turn of each conversation is a full prefill by design, and N
  conversations sharing a long system prompt cost N copies of its KV in host
  RAM. Interleaving many short conversations that share only a long system
  prompt is therefore slower than today on each first turn, not faster.
- No persistence across restarts. `state_write` / `state_read` still ignore the
  KVMem host store, so stores live and die with the process.

## Known limits

- **Orphaned host blocks after a full `seq_rm`.** The block table is zeroed
  while the host mirror keeps its blocks, and `truncate_cached`'s early return
  (`n_past >= total_tokens`) can no longer reach them. The re-prefilled block 0
  then keeps the previous content's packed K, because both `harvest_gpu_v` and
  `harvest_full_blocks_async` skip a layer whose packed K is already present.
  This predates the feature and is left untouched so the default path stays
  byte-identical. Multi-store cannot amplify it: the store records the
  divergence, a swap declines to carry it and clears it instead, and the server
  then drops that conversation's payload so the next request to select it is an
  ordinary cache miss rather than a resume against KV that is gone. Until the
  underlying bug is fixed, treat `--kvmem-conversations-gb` as advisory and the
  count cap as the real bound.
- **A drain failure the repair cannot finish leaves a store that reads full.**
  When the drain throws, `swap_conv` empties the outgoing store and rethrows,
  and the switch is reported as a refusal with both handles untouched. That
  repair cannot throw, but it can fail. It quiesces the store, empties the
  attention cells, rebuilds the free-slot list and marks the host mirror stale
  before it touches the block table, then calls `reset_policy` up to twice,
  because the one failure it knows about (`KvMemRuntime::wait_prefetch`
  rethrowing a stored NVMe read error) clears its futures before it rethrows
  and so cannot fail the same way twice. If both attempts fail anyway, the
  store stays attached with its attention cells gone and its block table still
  reading full, so `llama_kvmem_store_rows` on that handle reads healthy and
  the server's post-switch cross-check does not drop the payload. The request
  that failed can then prefill against a prefix whose cells are empty and
  answer from them. What bounds it is the stale mark: the next switch reads
  `host_mirror_stale`, refuses to drain that store and clears it instead of
  carrying it into another conversation, and the server drops that
  conversation's payload on the row cross-check at that point. So the exposure
  is the remainder of the request that was already failing, never a later
  request served against another conversation's KV.
- **Switch cost.** Two clients alternating every turn pay a full drain and
  restage twice per exchange: D2H of the resident K and V into host RAM, then
  H2D of the incoming store's blocks. Within-conversation turns pay nothing
  today. The first request that forks to a new conversation pays more than a
  switch: creating the host store allocates and zero-fills that store's pinned
  CPU arena synchronously, under the inference lock, before the prefill runs.
  Any benchmark must report switch cost separately from prefill cost, and the
  first-fork request separately from later switches.
- **One idle store after arming.** The store the context was built with starts
  empty, so the first request forks away from it and it sits idle in the table
  until eviction reclaims it. It costs one slot out of `N` until the first time
  the cap bites.
- **`--kvmem-raw-k-nvme` and multi-store are exclusive.** Each host store would
  open its own NVMe arena, sized for one conversation and named by a fixed
  file, so the adapter declines to arm in that mode.
- **`--kvmem-cpu-gb` and `--kvmem-nvme-gb` are per store, not per process.**
  Every host store builds its own `KvMemRuntime`, which allocates one pinned
  CPU arena of `--kvmem-cpu-gb` and one NVMe tier of `--kvmem-nvme-gb`, so
  `--kvmem-conversations 8 --kvmem-nvme-gb 64` means eight tiers each sized for
  64 GiB of slots in the same directory. The two tiers differ in what they
  actually commit. An NVMe tier is sparse and ephemeral: it neither
  preallocates nor keeps its file (`preallocate` and `durable` both default
  false and the server sets neither), so eight tiers of 64 GiB overcommit the
  configured budget rather than consume it. The pinned CPU arena is the
  opposite: `KvMemRuntime`'s constructor does `cpu_arena_.assign(slot_count *
  slot_bytes, 0)`, a fully resident zero-filled allocation, so
  `--kvmem-conversations 8 --kvmem-cpu-gb 8` commits up to 64 GiB of resident
  pinned RAM. The arena is allocated as stores are created, not at arming, and
  a store is created on the request that first forks to a new conversation, so
  that request pays the zero-fill under the inference lock. Default
  `--kvmem-cpu-gb` is 0, so a server that does not pass it commits nothing
  here. The server warns once at arming and does not divide the numbers: they
  stay per store, and the caps that bound the whole process are
  `--kvmem-conversations` and `--kvmem-conversations-gb`.

## Validation

```bash
# Policy, LRU and byte accounting. No CUDA toolchain and no llama.cpp checkout.
c++ -std=c++17 -I tools -o /tmp/cst tests/conversation-store-test.cpp && /tmp/cst

# The same test plus the flag parsing and the host-store byte definition.
ctest --test-dir build --output-on-failure

# CLI validation, no model.
python scripts/test_server_compat.py --server build/bin/llama-kvmem-server

# Interleaved conversations. Needs a GPU and a real GGUF.
python scripts/test_server_conversations.py --server build/bin/llama-kvmem-server \
    --model /path/model.gguf --output out/conversations
```

`kvmem-conversation-store-test` includes an exhaustive sweep of
`kvmem_store_keep_cap` and `kvmem_store_checkpoint`, the two helpers the policy
shares with the single-store prefill path, against a local transcription of
today's checkpoint selection, and asserts that the policy never extends on a
row that transcription would not select. Default identity itself is not
evidenced there but by construction: with the flag absent the policy is never
called, so no store is evicted and neither `store_select` nor `store_evict` is
emitted. The same test pins the eviction planner's refusal to ever name the
attached conversation, which would otherwise leave a host store alive with
nothing able to reach or release it.

`scripts/test_server_conversations.py` runs four servers. The baseline case
asserts today's behavior, that an interleaved conversation comes back as a full
miss. The second case, with `--kvmem-conversations 4`, asserts that the
returning conversation resumes its own prefix and that each conversation's
answer carries its own needle and not the other's. The third case, with
`--kvmem-conversations 2`, asserts that a third conversation evicts the least
recently used store, that the survivor still hits, that the store count stays
within the cap, and that the adapter's own store count agrees with the server's
table. That last one is what an eviction dropping a table row without
releasing the host store would break, and it is repeated on the `--bytes-case`
server, which is the path where that used to happen. The fourth case asserts
that a request answered with 400 leaves every conversation counter where it
was, selects no conversation, and leaves the least recently used conversation
resident, because the store switch runs below every validation of the request.
The 400 it raises is the `n_ctx` one, which needs no projector: it sends a
prompt hundreds of KB long at a 16K context. The cap is two with both
conversations resident, so attaching the conversation that prompt describes
would have had to evict the older one; that conversation answering its next
turn out of cache is what says no victim was destroyed. The three validations
below the `n_ctx` check are out of reach on this server: capacity validation
needs an image group, the reasoning-budget rejection needs a chat template
with no thinking markers, and the MTP sampler probe needs a grammar the draft
model cannot take. So the case pins the ordering with the last rejection it
can reach rather than with the image-sized request that originally broke it.

`scripts/test_server_compat.py` covers the default path: with `--model` it
asserts that a request carrying an unusable `kvmem.conversation_id` is still
served on a server with no `--kvmem-conversations` flag. Without a model it
asserts both new startup cross-checks, that `--kvmem-conversations-gb` without
`N > 1` and `--no-kvmem --kvmem-conversations 2` are rejected the same way
every other bad argument is.

## 中文摘要

`--kvmem-conversations N` 允许服务器同时在主机内存中保留 N 个会话的 KV 并在它们之间
切换；不带该参数或设为 `1` 时行为与此前完全一致，即换一个会话就会丢弃上一个。设计是
一个 `llama_context`、一个 GPU 工作集、N 个主机存储，按时间复用，请求仍然串行处理。

不需要客户端改动：会话身份就是客户端发送的 token 序列。请求体里的
`kvmem.conversation_id` 只作为可选优化，参数为 1 时被忽略；取值不可用（非字符串、
为空、超过 128 字节或含非可打印 ASCII 字节）时直接丢弃该字段，绝不会让请求失败。
请求继续某个已存会话时
延用该存储；只共享系统提示词或聊天模板时新建存储，避免截断原会话更长的尾部。匹配点
之前必须存在 recurrent 检查点，否则按 cache miss 处理，这是混合模型的硬约束。

`--kvmem-conversations-gb` 统计的是各存储 `bytes_k() + bytes_v()` 与服务端检查点、
token 历史之和，按最近最少使用淘汰**非当前**会话，因此它约束的是非活动存储之和，不是
显存或内存的硬上限；`--kvmem-cpu-gb` 与 `--kvmem-nvme-gb` 则是按存储计的，N 个会话会
各分配一份，启动时会警告一次。切换会把 GPU 工作集整体落盘再重建，复用会话内已有的换出/换入
流程；两个会话交替对话时每轮要付两次这个代价。本特性不引入并发解码，不改变
`n_seq_max` 或 `llama_seq_id`，不在存储之间共享块，也不支持跨进程持久化。
