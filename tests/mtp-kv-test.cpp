// Usage: CUDA_VISIBLE_DEVICES=... build/bin/kvmem-mtp-kv-test model-mtp.gguf
#include "kvmem-spec.h"
#include "llama-kvmem-hooks.h"
#include "llama-memory-kvmem-mtp.h"
#include "llama-memory-kvmem-hybrid.h"
#include "llama-kvmem-stagein.h"
#include "llama-memory-recurrent.h"

#include "ggml-backend.h"

#ifdef KVMEM_TEST_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

struct test_spec_session : kvmem_spec_session {
    ~test_spec_session() { kvmem_spec_stop(*this); }
};

struct kvmem_transfer_test_access {
    static inline kvmem::MemoryPhase fail_at = kvmem::MemoryPhase::Idle;
    static void probe(llama_memory_kvmem & mem, kvmem::MemoryPhase phase) {
        if (phase != kvmem::MemoryPhase::Prepared && mem.attention_view(false).valid)
            throw std::logic_error("half-completed native view escaped");
        if (phase == fail_at) throw std::runtime_error("injected native transfer failure");
    }
    static void inject(llama_memory_kvmem & mem, kvmem::MemoryPhase phase) {
        fail_at = phase;
        mem.native_probe_ = probe;
    }
    static kvmem::MemoryStamp stamp(const llama_memory_kvmem & mem) { return mem.content_->stamp(); }
    static uint64_t evaluated(const llama_memory_kvmem & mem) { return mem.content_->evaluated(); }
    static uint32_t mean_tokens(const llama_memory_kvmem & mem) { return mem.decode_mean_stats_.n_tok; }
    static bool complete(const llama_memory_kvmem & mem, uint32_t id, const llama_kv_cache * cache) {
        return mem.gpu_kv_complete(id, cache);
    }
    static void save(llama_memory_kvmem & mem, uint32_t id) { mem.harvest_gpu_v(id); }
    static void flush(llama_memory_kvmem & mem) { mem.harvest_gpu_v_commit(); }
    static void restore(llama_memory_kvmem & mem, uint32_t id) { mem.write_block_to_gpu(id); }
    static bool layout(llama_memory_kvmem & mem) { return mem.layout_gpu_slots_by_orig_pos(); }
    static void complete_manual_writes(llama_memory_kvmem & mem) {
        // This test writes synchronous sentinel bytes instead of running a graph.
        // Cell metadata alone must not make its earlier reservation complete.
        if (mem.content_->stamp().frontier != 0) throw std::runtime_error("reservation already committed");
        mem.content_->complete(0, mem.store_n_tokens());
        mem.evaluation_cursor_ = mem.evaluation_queue_.size();
    }
};

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static std::vector<uint8_t> pattern(size_t size, unsigned seed) {
    std::vector<uint8_t> bytes(size);
    for (auto & byte : bytes) {
        seed = seed * 1664525u + 1013904223u;
        byte = uint8_t(seed >> 24);
    }
    return bytes;
}

static void compare(ggml_tensor * tensor, const std::vector<uint8_t> & expected) {
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    require(actual == expected, "MTP KV bytes differ (including untouched rows)");
}

static void check_transfers(llama_memory_kvmem_mtp & mtp, ggml_type type_k, ggml_type type_v) {
    auto * kv = mtp.get_kv();
    require(kv->get_layer_ids().size() == 1, "test requires one MTP layer");
    const int il = kv->get_layer_ids().front();
    ggml_tensor * tensors[] = {kv->get_k_storage(il), kv->get_v_storage(il)};
    const uint32_t block = mtp.target()->block_tokens();
    auto & store = mtp.target()->runtime().store();
    const uint32_t nt = 2 * block + 13;
    std::vector<llama_pos> positions(4*nt, 0), logical(nt);
    for (uint32_t i = 0; i < nt; ++i) {
        logical[i] = i;
        positions[i] = 17;
        positions[nt+i] = 17 + i/8;
        positions[2*nt+i] = 17 + i%8;
    }
    llama_ubatch ub{};
    ub.n_tokens = nt;
    ub.n_pos = 4;
    ub.pos = positions.data();
    ub.logical_pos = logical.data();
    llama_kv_cache::slot_info_vec_t slots;
    require(mtp.target()->prepare_ubatches({ub}, nt, slots), "visual row slot preparation failed");
    for (uint32_t id = 0; id < 3; ++id) mtp.occupy_block(id);
    require(kvmem_transfer_test_access::complete(*mtp.target(), 0, kv), "complete visual draft block was rejected");
    require(mtp.remove_logical(11, 12), "cannot create a middle-row coverage gap");
    require(!kvmem_transfer_test_access::complete(*mtp.target(), 0, kv), "middle-row draft gap was missed");
    mtp.occupy_block(0);
    require(kvmem_transfer_test_access::complete(*mtp.target(), 2, kv), "partial visual block was rejected");

    std::vector<uint8_t> original[2];
    for (int i = 0; i < 2; ++i) {
        auto * t = tensors[i];
        require(t && t->type == (i == 0 ? type_k : type_v), "draft did not inherit requested KV type");
        require(!ggml_backend_buffer_is_host(t->buffer), "test requires GPU KV");
        original[i] = pattern(ggml_nbytes(t), 42 + i);
        ggml_backend_tensor_set(t, original[i].data(), 0, original[i].size());
    }

    // Save at nonzero offsets, destroy GPU contents, then restore to new slots.
    for (uint32_t id = 0; id < 3; ++id) {
        mtp.on_stage_out(id);
        if (id + 1 < 3) require(mtp.slot_holds(id + 1, (id + 1)*block), "stage-out removed patches sharing t in another block");
        store.set_block_gpu_slot(id, 3 + 2 * id);
    }
    mtp.harvest_flush();
    std::vector<uint8_t> expected[2];
    for (int i = 0; i < 2; ++i) {
        expected[i].assign(original[i].size(), 0xa5);
        ggml_backend_tensor_set(tensors[i], expected[i].data(), 0, expected[i].size());
        const size_t row = ggml_row_size(tensors[i]->type, tensors[i]->ne[0]);
        for (uint32_t id = 0; id < 3; ++id) {
            const size_t size = store.blocks()[id].n_tokens * row;
            std::copy_n(original[i].data() + id * block * row, size,
                        expected[i].data() + (3 + 2 * id) * block * row);
        }
    }
    mtp.follow_retrieval();
    for (int i = 0; i < 2; ++i) {
        compare(tensors[i], expected[i]);
    }
    for (const auto & b : store.blocks()) {
        require(mtp.slot_holds(b.gpu_slot, b.orig_pos_start), "restored slot position differs");
        const auto & cells = kv->get_cells(0);
        for (uint32_t i = 0; i < b.n_tokens; ++i) {
            const uint32_t cell = b.gpu_slot * block + i;
            const uint32_t row = b.orig_pos_start + i;
            require(cells.pos_get(cell) == positions[row], "restored temporal position differs");
            const auto & ext = cells.ext_get(cell);
            require(ext.y == positions[nt+row] && ext.x == positions[2*nt+row], "restored spatial position differs");
            require(ext.logical_pos == (llama_pos) row, "restored logical row differs");
        }
    }

    // A cycle must preserve unread sources and the tail of the partial block.
    const llama_memory_kvmem_mtp::LayoutMove moves[] = {
        {3, 5, block}, {5, 7, block}, {7, 3, 13},
    };
#ifdef KVMEM_TEST_CUDA
    // A failed optional scratch allocation must leave KV and CUDA state intact.
    bool test_physical_limit = true;
#ifdef _WIN32
    int device = 0, tcc = 0;
    require(cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, device) == cudaSuccess,
            "CUDA driver mode query failed");
    // WDDM can accept a cudaMalloc larger than physical VRAM. Such a call
    // would execute the test moves instead of exercising allocation failure.
    test_physical_limit = tcc != 0;
#endif
    if (test_physical_limit) {
        size_t free_bytes = 0, total_bytes = 0;
        require(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess, "CUDA memory query failed");
        const size_t stride = block * (ggml_row_size(type_k, tensors[0]->ne[0]) +
                                       ggml_row_size(type_v, tensors[1]->ne[0]));
        std::vector<llama_memory_kvmem_mtp::LayoutMove> oversized(total_bytes / stride + 1, moves[0]);
        require(cudaGetLastError() == cudaSuccess, "CUDA error before layout OOM test");
        require(!mtp.layout_d2d(oversized.data(), oversized.size()), "oversized layout allocation succeeded");
        require(cudaGetLastError() == cudaSuccess, "layout OOM leaked into the next CUDA operation");
        for (int i = 0; i < 2; ++i) {
            compare(tensors[i], expected[i]);
        }
    } else {
        std::printf("SKIP physical-VRAM OOM injection on WDDM (not a reliable allocation limit)\n");
    }
#endif
    for (int i = 0; i < 2; ++i) {
        const auto before = expected[i];
        const size_t row = ggml_row_size(tensors[i]->type, tensors[i]->ne[0]);
        for (const auto & move : moves) {
            std::copy_n(before.data() + move.src_slot * block * row, move.n_tokens * row,
                        expected[i].data() + move.dst_slot * block * row);
        }
    }
    require(mtp.layout_d2d(moves, 3), "MTP CUDA layout failed");
    for (int i = 0; i < 2; ++i) {
        compare(tensors[i], expected[i]);
    }
}

static void check_target_transfers(llama_memory_kvmem & mem, ggml_type type_k, ggml_type type_v) {
    auto * kv = mem.get_kv();
    auto & store = mem.runtime().store();
    const uint32_t block = mem.block_tokens();
    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<uint8_t>> original, expected;
    for (int il : kv->get_layer_ids()) {
        tensors.push_back(kv->get_k_storage(il));
        tensors.push_back(kv->get_v_storage(il));
    }
    for (uint32_t id = 0; id < 3; ++id) mem.occupy_in(kv, id);
    require(kvmem_transfer_test_access::complete(mem, 0, kv), "complete target block was rejected");
    require(kv->seq_rm_logical(0, 11, 12), "cannot create target coverage gap");
    require(!kvmem_transfer_test_access::complete(mem, 0, kv), "middle-row target gap was missed");
    mem.occupy_in(kv, 0);
    llama_kvmem_turn_spans spans;
    spans.query = {{10, 20}, {30, 32}};
    spans.mandatory = {{0, 64}};
    spans.replay_begin = 0;
    mem.set_turn_spans(spans);
    require(mem.query_contains(10) && mem.query_contains(31) && !mem.query_contains(25), "query spans were conflated with mandatory rows");
    mem.freeze_query(true);
    require(!mem.query_contains(10), "frozen Q can still accumulate");
    mem.freeze_query(false);
    mem.set_query_span(-1, -1);
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        require(tensor->type == (i % 2 == 0 ? type_k : type_v), "target KV type differs");
        original.push_back(pattern(ggml_nbytes(tensor), 123 + tensors.size()));
        ggml_backend_tensor_set(tensor, original.back().data(), 0, original.back().size());
    }
    require(!mem.attention_view(false).valid, "reserved but unwritten rows were published");
    kvmem_transfer_test_access::complete_manual_writes(mem);
    for (uint32_t id = 0; id < 3; ++id) kvmem_transfer_test_access::save(mem, id);
    kvmem_transfer_test_access::flush(mem);
    for (size_t i = 0; i < tensors.size(); ++i) {
        expected.emplace_back(original[i].size(), 0x5a);
        ggml_backend_tensor_set(tensors[i], expected[i].data(), 0, expected[i].size());
        const size_t row = ggml_row_size(tensors[i]->type, tensors[i]->ne[0]);
        for (uint32_t id = 0; id < 3; ++id) {
            std::copy_n(original[i].data() + (3 + 2*id)*block*row, store.blocks()[id].n_tokens*row,
                        expected[i].data() + (2 + 2*id)*block*row);
        }
    }
    for (uint32_t id = 0; id < 3; ++id) {
        require(kv->seq_rm_logical(0, id*block, (id+1)*block), "target logical removal failed");
        if (id < 2) require(!kv->get_cells(0).is_empty((5 + 2*id)*block), "target removed another block sharing t");
        store.set_block_gpu_slot(id, 2 + 2*id);
        kvmem_transfer_test_access::restore(mem, id);
    }
    require(kvmem_stagein_flush(nullptr, nullptr, nullptr, nullptr), "target stage-in flush failed");
    kvmem_stagein_sync();
    for (size_t i = 0; i < tensors.size(); ++i) compare(tensors[i], expected[i]);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto before = expected[i];
        const size_t row = ggml_row_size(tensors[i]->type, tensors[i]->ne[0]);
        for (uint32_t id = 0; id < 3; ++id) {
            std::copy_n(before.data() + (2 + 2*id)*block*row, store.blocks()[id].n_tokens*row,
                        expected[i].data() + id*block*row);
        }
    }
    require(kvmem_transfer_test_access::layout(mem), "target D2D layout failed");
    for (size_t i = 0; i < tensors.size(); ++i) compare(tensors[i], expected[i]);
    for (const auto & b : store.blocks()) for (uint32_t i = 0; i < b.n_tokens; ++i) {
        const auto cell = b.gpu_slot*block + i;
        const auto logical = b.orig_pos_start + i;
        const auto & cells = kv->get_cells(0);
        const auto & ext = cells.ext_get(cell);
        require(cells.pos_get(cell) == 17 && ext.logical_pos == (llama_pos) logical
                && ext.y == 17 + (llama_pos) logical/8 && ext.x == 17 + (llama_pos) logical%8,
                "target restored position metadata differs");
    }
}

static void check_batch_inputs(llama_model * model) {
    constexpr int n = 9, width = 3;
    std::vector<float> input(n*width), hidden(n*width);
    std::vector<llama_pos> pos(4*n, 12), logical(n);
    for (int i = 0; i < n; ++i) {
        logical[i] = 100 + i;
        pos[n+i] = i/3;
        pos[2*n+i] = i%3;
        for (int j = 0; j < width; ++j) {
            input[i*width+j] = i*10+j;
            hidden[i*width+j] = -i*10-j-1;
        }
    }
    llama_batch batch{};
    batch.n_tokens = n;
    batch.embd = input.data();
    batch.embd_nextn = hidden.data();
    batch.pos = pos.data();
    batch.logical_pos = logical.data();
    llama_batch_allocr allocator(4);
    require(allocator.init(batch, *llama_model_get_vocab(model), nullptr, width, 1, true), "visual batch init failed");
    allocator.split_reset();
    int seen = 0;
    while (seen < n) {
        auto ub = allocator.split_simple(4);
        require(ub.n_tokens > 0, "visual batch split failed");
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            int row = ub.logical_pos[i] - 100;
            require(row == seen++, "split lost logical row order");
            for (int j = 0; j < width; ++j) {
                require(ub.embd[i*width+j] == input[row*width+j], "split lost input embedding");
                require(ub.embd_nextn[i*width+j] == hidden[row*width+j], "split lost MTP hidden input");
            }
            for (int j = 0; j < 4; ++j) require(ub.pos[j*ub.n_tokens+i] == pos[j*n+row], "split lost M-RoPE position plane");
        }
    }
    std::vector<int32_t> n_seq(n, 1);
    std::vector<llama_seq_id> seq(n);
    std::vector<llama_seq_id *> ids(n);
    for (int i = 0; i < n; ++i) { seq[i] = i%2; ids[i] = &seq[i]; }
    batch.n_seq_id = n_seq.data();
    batch.seq_id = ids.data();
    llama_batch_allocr reordered(4);
    require(reordered.init(batch, *llama_model_get_vocab(model), nullptr, width, 2, true), "reordered batch init failed");
    reordered.split_reset();
    std::vector<bool> visited(n, false);
    for (int count = 0; count < n;) {
        const auto ub = reordered.split_seq(4);
        require(ub.n_tokens > 0, "reordered batch split failed");
        for (uint32_t i = 0; i < ub.n_tokens; ++i, ++count) {
            const int row = ub.logical_pos[i] - 100;
            require(row >= 0 && row < n && !visited[row], "reorder duplicated or lost logical row");
            visited[row] = true;
            require(ub.seq_id[i][0] == seq[row], "reordered sequence differs");
            for (int j = 0; j < width; ++j) {
                require(ub.embd[i*width+j] == input[row*width+j], "reorder lost visual input");
                require(ub.embd_nextn[i*width+j] == hidden[row*width+j], "reorder lost hidden input");
            }
            for (int j = 0; j < 4; ++j) require(ub.pos[j*ub.n_tokens+i] == pos[j*n+row], "reorder lost M-RoPE plane");
        }
    }
    std::puts("PASS visual batch split: logical rows, M-RoPE, separate hidden input");
}

static void check_replay_logits(llama_model * model, ggml_type type, int mtp_state = 0, ggml_type type_v = GGML_TYPE_COUNT) {
    llama_kvmem_params kp{};
    kp.enabled = true;
    kp.method = 1;
    kp.block_tokens = 32;
    kp.budget = 1024;
    kp.gen_reserve = 256;
    kp.query_begin = kp.query_end = kp.force_pos = -1;
    kp.mtp_state = mtp_state;
    llama_kvmem_set_params(&kp);
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = cp.n_ubatch = 128;
    if (type == GGML_TYPE_Q8_0 && mtp_state == 0 && type_v == GGML_TYPE_COUNT)
        cp.n_ubatch = 32; // one reservation, multiple graph completions
    cp.n_seq_max = 1;
    cp.n_rs_seq = 2;
    cp.type_k = type;
    cp.type_v = type_v == GGML_TYPE_COUNT ? type : type_v;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "logits test context init failed");
    std::string text = "<|im_start|>user\n";
    for (int i = 0; i < 24; ++i) text += "The secret code is 7391. Keep this number in memory.\n";
    text += "What is the secret code? Reply with its four digits.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    auto tokens = common_tokenize(llama_model_get_vocab(model), text, true, true);
    const int end = tokens.size(), query = end - 24;
    require(query > 0 && end < 1024, "unexpected logits fixture length");
    llama_kvmem_set_request_span(query, end, -1);
    llama_kvmem_set_turn_spans({{{query, end}}, {{query, end}}, query});
    auto decode = [&](int begin, int stop) {
        for (int row = begin; row < stop;) {
            const int n = std::min(128, stop - row);
            auto batch = llama_batch_get_one(tokens.data() + row, n);
            std::vector<llama_pos> pos(n);
            for (int i = 0; i < n; ++i) pos[i] = row + i;
            batch.pos = batch.logical_pos = pos.data();
            require(llama_decode(ctx.get(), batch) == 0, "logits fixture decode failed");
            row += n;
        }
        llama_synchronize(ctx.get());
    };
    decode(0, query);
    // Real adapter faults, using a completed model graph and actual device KV.
    // The first two fail before destructive admission; the latter two require
    // an invalid session and fresh prefill, never a metadata-only rollback.
    if (type == GGML_TYPE_Q8_0 && mtp_state == 0 && type_v == GGML_TYPE_COUNT) {
        auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(ctx.get()));
        auto * mem = hybrid ? hybrid->attn_kvmem() : dynamic_cast<llama_memory_kvmem *>(llama_get_memory(ctx.get()));
        require(mem != nullptr, "missing native adapter for fault test");
        for (auto phase : {kvmem::MemoryPhase::Prepared, kvmem::MemoryPhase::Spilling,
                           kvmem::MemoryPhase::Restoring, kvmem::MemoryPhase::Ready}) {
            auto choice = llama_kvmem_preview_retrieval();
            choice.blocks = {choice.blocks.front(), choice.blocks.back()};
            const auto old = kvmem_transfer_test_access::stamp(*mem);
            kvmem_transfer_test_access::inject(*mem, phase);
            bool failed = false;
            try { llama_kvmem_apply_selection(choice); } catch (const std::runtime_error &) { failed = true; }
            kvmem_transfer_test_access::inject(*mem, kvmem::MemoryPhase::Idle);
            require(failed, "native transfer fault was not propagated");
            const auto after = kvmem_transfer_test_access::stamp(*mem);
            if (phase == kvmem::MemoryPhase::Prepared || phase == kvmem::MemoryPhase::Spilling) {
                require(after == old && mem->attention_view(false).valid, "pre-admission failure lost old view");
            } else {
                require(!after.valid && !mem->attention_view(false).valid && mem->store_n_tokens() == 0,
                        "destructive failure did not invalidate and drain the session");
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                llama_kvmem_set_request_span(query, end, -1);
                llama_kvmem_set_turn_spans({{{query, end}}, {{query, end}}, query});
                decode(0, query);
            }
        }
        std::puts("PASS native adapter faults: prepare/spill preserve, restore/publish invalidate, fresh prefill recovers");
    }
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> checkpoint(llama_state_seq_get_size_ext(ctx.get(), 0, flags));
    require(llama_state_seq_get_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
            "logits checkpoint save failed");
    const auto before = llama_kvmem_get_attention_view();
    decode(query, end);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto logits = [&]() {
        const auto * p = llama_get_logits_ith(ctx.get(), -1);
        require(p != nullptr, "missing logits");
        return std::vector<float>(p, p + n_vocab);
    };
    const auto single = logits();
    auto selection = llama_kvmem_preview_retrieval();
    require(llama_kvmem_commit_unchanged(before, selection), "identical attention view rejected");
    llama_kvmem_query_state q;
    require(llama_kvmem_get_query(q) && *std::max_element(q.count.begin(), q.count.end()) == 24,
            "query capture count differs from explicit span");
    llama_kvmem_tail_mean_state tail;
    require(llama_kvmem_get_tail_mean(end, tail) && llama_kvmem_tail_mean_valid(end, tail),
            "accepted mean-K checkpoint missing");
    auto foreign_query = q;
    kvmem::KvContent foreign(32);
    foreign.reserve(end); foreign.complete(0, end);
    foreign_query.prefix = foreign.prefix(end);
    require(!llama_kvmem_set_query(foreign_query), "cross-session Q imported");
    auto foreign_tail = tail;
    foreign_tail.prefix = foreign_query.prefix;
    require(!llama_kvmem_tail_mean_valid(end, foreign_tail) && !llama_kvmem_set_tail_mean(end, foreign_tail),
            "cross-session mean-K checkpoint imported");
    require(!llama_kvmem_tail_mean_valid(end + 1, tail), "wrong checkpoint frontier accepted");
    auto wrong_span = q;
    ++wrong_span.ranges.front().begin;
    require(!llama_kvmem_set_query(wrong_span), "Q from a different query range imported");
    auto missing_required = selection;
    missing_required.blocks.pop_back();
    require(!llama_kvmem_selection_fits(missing_required, end, 0), "trimmed mandatory tail accepted for continuation");
    std::string reason;
    require(!llama_kvmem_can_append(end, cp.n_ctx, true, reason), "oversized generation reserve accepted");
    auto invalid_query = q;
    invalid_query.count.pop_back();
    require(!llama_kvmem_set_query(invalid_query), "invalid Q state dimensions accepted");
    std::vector<std::vector<float>> repeated;
    for (int repeat = 0; repeat < 2; ++repeat) {
        llama_kvmem_apply_selection(selection);
        require(llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
                "logits checkpoint restore failed");
        require(llama_kvmem_remove_logical(ctx.get(), query, -1), "logits suffix removal failed");
        require(!llama_kvmem_commit_unchanged(before, selection), "stale attention view accepted after restore");
        bool rejected = false;
        try { llama_kvmem_apply_selection(selection); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "stale selection accepted after restore");
        llama_kvmem_set_replay(true);
        decode(query, end);
        llama_kvmem_set_replay(false);
        repeated.push_back(logits());
        llama_kvmem_query_state replay_query;
        require(llama_kvmem_get_query(replay_query), "query capture missing after replay");
        require(replay_query.count == q.count && replay_query.sum == q.sum,
                "replay changed first-pass query statistics");
        require(llama_kvmem_tail_mean_valid(end, tail), "execution replay revoked logical prefix");
        std::printf("QUERY_REPLAY repeat=%d rows=%u original_rows=%u\n", repeat + 1,
                *std::max_element(replay_query.count.begin(), replay_query.count.end()),
                *std::max_element(q.count.begin(), q.count.end()));
        selection = llama_kvmem_preview_retrieval();
    }
    auto compare_logits = [&](const std::vector<float> & a, const std::vector<float> & b, const char * label) {
        double sum = 0, maxabs = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite logits");
            const double d = a[i] - b[i];
            sum += d*d;
            maxabs = std::max(maxabs, std::abs(d));
        }
        const double rmse = std::sqrt(sum / a.size());
        const auto top_a = std::max_element(a.begin(), a.end()) - a.begin();
        const auto top_b = std::max_element(b.begin(), b.end()) - b.begin();
        std::printf("LOGITS K=%s V=%s comparison=%s rmse=%.9f maxabs=%.9f top=%td/%td\n",
                    ggml_type_name(type), ggml_type_name(cp.type_v), label, rmse, maxabs, top_a, top_b);
        require(rmse < .0001 && maxabs < .001 && top_a == top_b, "replay logits exceed numerical tolerance");
    };
    compare_logits(repeated[0], repeated[1], "replay_repeat");
    compare_logits(single, repeated[0], "single_vs_replay");
    // A sparse view can also remain unchanged while its partial tail grows.
    // Exercise the proof independently of the scoring policy's chosen budget.
    auto sparse = llama_kvmem_preview_retrieval();
    sparse.blocks = {sparse.blocks.front(), sparse.blocks.back()};
    llama_kvmem_apply_selection(sparse);
    llama_kvmem_begin_cached_turn();
    llama_kvmem_keep_selected();
    const auto sparse_before = llama_kvmem_get_attention_view();
    auto extra = common_tokenize(llama_model_get_vocab(model), " Additional note.", false, true);
    tokens.insert(tokens.end(), extra.begin(), extra.end());
    const int new_end = tokens.size();
    llama_kvmem_set_turn_spans({{{end, new_end}}, {{end, new_end}}, end});
    decode(end, new_end);
    const auto sparse_after = llama_kvmem_get_attention_view();
    auto same_sparse = llama_kvmem_preview_retrieval();
    same_sparse.blocks = sparse_after.blocks;
    require(llama_kvmem_commit_unchanged(sparse_before, same_sparse), "unchanged sparse attention view rejected");
    require(sparse_after.blocks.size() < (uint32_t) (end / 32), "test did not create sparse history");
    if (type == GGML_TYPE_Q8_0 && mtp_state == 0 && type_v == GGML_TYPE_COUNT) {
        auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(ctx.get()));
        auto * mem = hybrid ? hybrid->attn_kvmem() : dynamic_cast<llama_memory_kvmem *>(llama_get_memory(ctx.get()));
        require(mem != nullptr, "missing native adapter for acceptance test");
        llama_kvmem_pin_working_set();
        int next = new_end;
        const int widths[] = {2, 2, 2, 1};
        const int keeps[] = {0, 1, 2, 0};
        for (int step = 0; step < 4; ++step) {
            const int width = widths[step], keep = keeps[step];
            std::vector<llama_token> proposed(width, tokens.back());
            std::vector<llama_pos> positions(width);
            for (int i = 0; i < width; ++i) positions[i] = next + i;
            auto batch = llama_batch_get_one(proposed.data(), width);
            batch.pos = batch.logical_pos = positions.data();
            const auto means_before = kvmem_transfer_test_access::mean_tokens(*mem);
            llama_kvmem_begin_speculative_evaluation();
            require(llama_decode(ctx.get(), batch) == 0, "acceptance fixture evaluation failed");
            llama_synchronize(ctx.get());
            require(kvmem_transfer_test_access::stamp(*mem).frontier == (uint64_t) next &&
                    kvmem_transfer_test_access::evaluated(*mem) == (uint64_t) (next + width) &&
                    !mem->attention_view(false).valid, "unaccepted target KV was published");
            require(kvmem_transfer_test_access::mean_tokens(*mem) == means_before,
                    "unaccepted draft changed mean-K");
            require(!llama_kvmem_commit_speculative(ctx.get(), width + 1), "oversized prefix committed");
            require(llama_kvmem_commit_speculative(ctx.get(), keep), "accepted prefix commit failed");
            require(!llama_kvmem_commit_speculative(ctx.get(), keep), "duplicate prefix committed");
            require(kvmem_transfer_test_access::mean_tokens(*mem) == means_before + keep,
                    "mean-K count differs from accepted prefix");
            require(kvmem_transfer_test_access::stamp(*mem).frontier == (uint64_t) (next + keep),
                    "accepted KV frontier differs");
            if (keep != width) require(llama_kvmem_remove_logical(ctx.get(), next + keep, -1), "rejected tail removal failed");
            llama_kvmem_truncate_cached(next + keep);
            next += keep;
            require(mem->attention_view(false).valid && mem->store_n_tokens() == (uint32_t) next &&
                    kvmem_transfer_test_access::evaluated(*mem) == (uint64_t) next, "rejected KV remained visible");
        }
        std::puts("PASS native target acceptance: zero/partial/full prefix, width-one rejection, no speculative publication");
        require(llama_kvmem_tail_mean_valid(end, tail), "append revoked accepted checkpoint");
        auto * bundle = llama_kvmem_store_bundle_create();
        require(!llama_kvmem_store_bundle_swap(bundle), "fresh conversation unexpectedly had KV");
        require(!llama_kvmem_tail_mean_valid(end, tail), "foreign conversation accepted old checkpoint");
        require(llama_kvmem_store_bundle_swap(bundle), "original conversation restore failed");
        require(llama_kvmem_tail_mean_valid(end, tail), "conversation transfer lost prefix identity");
        require(!llama_kvmem_store_bundle_swap(bundle), "empty conversation acquired old KV");
        llama_kvmem_store_bundle_reset(bundle);
        require(!llama_kvmem_store_bundle_swap(bundle), "reset conversation restored stale KV");
        require(kvmem_transfer_test_access::stamp(*mem).frontier == 0 &&
                !llama_kvmem_tail_mean_valid(end, tail) && !llama_kvmem_set_query(q),
                "bundle reset retained content or statistics identity");
        llama_kvmem_store_bundle_free(bundle);
        std::puts("PASS statistics identity: foreign/range rejection, replay/append preservation, conversation reset revocation");
    }
    std::printf("PASS K=%s V=%s: unchanged view, explicit Q, stale plan rejection, replay logits\n", ggml_type_name(type), ggml_type_name(cp.type_v));
}

struct gdn_model_result {
    std::vector<float> logits;
    std::vector<uint64_t> states;
};

static gdn_model_result run_gdn_transactions(llama_model * model, ggml_type type, bool replay, int drafts, ggml_type type_v) {
    llama_kvmem_params kp{};
    kp.enabled = true;
    kp.block_tokens = 32;
    kp.budget = 512;
    kp.gen_reserve = 128;
    kp.query_begin = kp.query_end = kp.force_pos = -1;
    kp.mtp_state = replay ? 2 : 0;
    llama_kvmem_set_params(&kp);
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = cp.n_ubatch = 128;
    cp.n_seq_max = 1;
    cp.n_rs_seq = drafts;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = drafts + 1;
    cp.type_k = type;
    cp.type_v = type_v == GGML_TYPE_COUNT ? type : type_v;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "GDN transaction context init failed");
    auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(ctx.get()));
    require(hybrid != nullptr, "GDN transaction needs hybrid memory");
    auto * mem = hybrid->get_mem_recr();
    require(llama_kvmem_gdn_replay_enabled() == replay, "GDN strategy differs from requested mode");
    auto tokens = common_tokenize(llama_model_get_vocab(model),
            "Remember the key 7391. Verify the state before accepting a proposed continuation. "
            "Each accepted input advances memory exactly once. Rejected inputs leave no recurrent trace.\n", true, true);
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    require(llama_decode(ctx.get(), batch) == 0, "GDN transaction prefill failed");
    llama_kvmem_end_prefill_capture();
    int pos = tokens.size();
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    gdn_model_result result;
    const int width_max = drafts + 1;
    const int widths[] = {width_max, width_max, width_max, width_max, 1, 2, 3, 1, 2};
    const int keeps[] =  {0, 1, width_max - 1, width_max, 1, 1, 2, 0, 2};
    for (size_t round = 0; round < sizeof(widths)/sizeof(widths[0]); ++round) {
        const int width = widths[round], keep = keeps[round];
        std::vector<uint8_t> checkpoint(llama_state_seq_get_size_ext(ctx.get(), 0, flags));
        require(llama_state_seq_get_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
                "GDN transaction checkpoint save failed");
        std::vector<llama_token> ids(width);
        std::vector<llama_pos> positions(width);
        std::vector<int8_t> outputs(width, 1);
        for (int i = 0; i < width; ++i) {
            ids[i] = tokens[7 + i * 4];
            positions[i] = pos + i;
        }
        batch = llama_batch_get_one(ids.data(), width);
        batch.pos = batch.logical_pos = positions.data();
        batch.logits = outputs.data();
        if (replay) {
            require(llama_kvmem_gdn_replay_begin(pos, width), "GDN transaction begin failed");
            require(!llama_kvmem_gdn_replay_begin(pos, width), "overlapping GDN transaction accepted");
        }
        llama_kvmem_begin_speculative_evaluation();
        require(llama_decode(ctx.get(), batch) == 0, "GDN Record model decode failed");
        for (int row = 0; row < width; ++row) {
            const float * logits = llama_get_logits_ith(ctx.get(), row);
            require(logits != nullptr, "GDN Record logits missing");
            result.logits.insert(result.logits.end(), logits, logits + n_vocab);
        }
        if (replay) {
            require(llama_kvmem_commit_speculative(ctx.get(), keep), "GDN and statistics commit failed");
            require(!llama_kvmem_commit_speculative(ctx.get(), keep), "duplicate GDN commit accepted");
        } else if (keep == 0) {
            require(llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
                    "GDN zero-accept checkpoint restore failed");
        }
        if (!replay) {
            if (keep) require(llama_kvmem_commit_speculative(ctx.get(), keep), "snapshot prefix commit failed");
            else llama_kvmem_decode_mean_discard();
        }
        pos += keep;
        require(llama_kvmem_remove_logical(ctx.get(), pos, -1), "GDN accepted suffix trim failed");
        llama_kvmem_truncate_cached(pos);
        require(mem->seq_pos_max(0) == pos - 1, "GDN frontier advanced by rejected inputs");
        for (size_t il = 0; il < mem->s_l.size(); ++il) {
            if (!mem->s_l[il]) continue;
            for (auto * t : {mem->s_l[il], mem->r_l[il]}) {
                const size_t bytes = ggml_nbytes(t) / (mem->n_rs_seq + 1);
                std::vector<uint8_t> state(bytes);
                ggml_backend_tensor_get(t, state.data(), mem->rs_idx[0] * bytes, bytes);
                uint64_t hash = 14695981039346656037ULL;
                for (uint8_t x : state) hash = (hash ^ x) * 1099511628211ULL;
                result.states.push_back(hash);
            }
        }
    }
    return result;
}

static void check_gdn_transactions(llama_model * model, ggml_type type, int drafts = 2, ggml_type type_v = GGML_TYPE_COUNT) {
    const auto baseline = run_gdn_transactions(model, type, false, drafts, type_v);
    const auto replay = run_gdn_transactions(model, type, true, drafts, type_v);
    require(baseline.logits.size() == replay.logits.size(), "GDN logits dimensions differ");
    double squared = 0, maximum = 0;
    for (size_t i = 0; i < baseline.logits.size(); ++i) {
        require(std::isfinite(baseline.logits[i]) && std::isfinite(replay.logits[i]), "nonfinite GDN logits");
        const double delta = baseline.logits[i] - replay.logits[i];
        squared += delta * delta;
        maximum = std::max(maximum, std::abs(delta));
    }
    const double rmse = std::sqrt(squared / baseline.logits.size());
    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (size_t i = 0; i < baseline.logits.size(); i += n_vocab) {
        const auto a = baseline.logits.begin() + i, b = replay.logits.begin() + i;
        require(std::max_element(a, a + n_vocab) - a == std::max_element(b, b + n_vocab) - b, "GDN top-1 differs");
    }
    std::printf("GDN_LOGITS K=%s V=%s mtp=%d rmse=%.9f maxabs=%.9f exact_states=%d\n", ggml_type_name(type), ggml_type_name(type_v == GGML_TYPE_COUNT ? type : type_v), drafts, rmse, maximum,
            baseline.states == replay.states);
    require(rmse < .0001 && maximum < .001, "GDN model logits exceed numerical tolerance");
    require(baseline.states == replay.states, "GDN model committed states differ");
    std::printf("PASS GDN K=%s V=%s mtp=%d: zero/partial/full commit, all-layer states, restore, duplicate commit\n", ggml_type_name(type), ggml_type_name(type_v == GGML_TYPE_COUNT ? type : type_v), drafts);
}

int main(int argc, char ** argv) {
    const bool target_only = argc == 3 && std::string(argv[2]) == "--target-only";
    if (argc != 2 && !target_only) {
        std::fprintf(stderr, "Usage: %s model.gguf [--target-only] (requires CUDA; full suite requires Qwen 27B MTP weights)\n", argv[0]);
        return argc == 1 ? 77 : 1;
    }
    try {
        llama_backend_init();
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        mp.load_mtp = !target_only;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        check_batch_inputs(model.get());

        llama_kvmem_params kp{};
        kp.enabled = true;
        kp.block_tokens = 32;
        kp.budget = 256;
        kp.gen_reserve = 256;
        kp.query_begin = kp.query_end = kp.force_pos = -1;
        llama_kvmem_set_params(&kp);
        struct cache_case { ggml_type target; ggml_type draft; ggml_type value = GGML_TYPE_COUNT; };
        const cache_case cases[] = {
            {GGML_TYPE_Q8_0, GGML_TYPE_COUNT},
            {GGML_TYPE_Q5_0, GGML_TYPE_COUNT},
            {GGML_TYPE_Q4_0, GGML_TYPE_COUNT},
            {GGML_TYPE_F16,  GGML_TYPE_COUNT},
            {GGML_TYPE_Q5_0, GGML_TYPE_F16},
            {GGML_TYPE_Q5_0, GGML_TYPE_Q8_0},
            {GGML_TYPE_Q5_0, GGML_TYPE_Q5_0},
            {GGML_TYPE_Q8_0, GGML_TYPE_COUNT, GGML_TYPE_Q4_0},
            {GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_Q4_0},
            {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0},
        };
        for (const auto & test : cases) {
            if (target_only) break;
            const auto type = test.target;
            const auto type_v = test.value == GGML_TYPE_COUNT ? type : test.value;
            const auto draft_k = test.draft == GGML_TYPE_COUNT ? type : test.draft;
            const auto draft_v = test.draft == GGML_TYPE_COUNT ? type_v : test.draft;
            auto cp = llama_context_default_params();
            cp.n_ctx = 1024;
            cp.n_batch = cp.n_ubatch = 128;
            cp.n_seq_max = 1;
            cp.n_rs_seq = 2;
            cp.n_outputs_max = cp.n_outputs_max_per_seq = 3;
            cp.type_k = type;
            cp.type_v = type_v;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                    llama_init_from_model(model.get(), cp), llama_free);
            require(bool(ctx), "target context init failed");
            kvmem_spec_opts opts;
            opts.kvmem_enabled = true;
            opts.n_ctx = cp.n_ctx;
            opts.n_batch = opts.n_ubatch = cp.n_batch;
            opts.type_k = type;
            opts.type_v = type_v;
            opts.draft_type = test.draft;
            // Stop the follower before freeing its target, including on failure.
            test_spec_session sess;
            require(kvmem_spec_start(sess, model.get(), ctx.get(), opts), "MTP init failed");
            auto * mtp = dynamic_cast<llama_memory_kvmem_mtp *>(llama_get_memory(sess.ctx_dft));
            require(mtp != nullptr, "missing KVMem MTP follower");
            require(mtp->target()->get_kv()->type_k() == type && mtp->target()->get_kv()->type_v() == type_v,
                    "draft override changed target KV types");
            check_transfers(*mtp, draft_k, draft_v);
            check_target_transfers(*mtp->target(), type, type_v);
            std::printf("PASS target=%s/%s draft=%s/%s override=%d: GPU save/restore, cycle, partial block\n",
                        ggml_type_name(type), ggml_type_name(type_v), ggml_type_name(draft_k), ggml_type_name(draft_v), test.draft != GGML_TYPE_COUNT);
        }
        check_replay_logits(model.get(), GGML_TYPE_Q8_0);
        check_replay_logits(model.get(), GGML_TYPE_Q5_0);
        check_replay_logits(model.get(), GGML_TYPE_Q8_0, 0, GGML_TYPE_Q4_0);
        if (!target_only) {
            for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0}) {
                check_gdn_transactions(model.get(), type);
            }
            for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q5_0}) {
                for (int drafts : {3, 4, 5}) check_gdn_transactions(model.get(), type, drafts);
            }
            check_replay_logits(model.get(), GGML_TYPE_Q5_0, 2);
            check_replay_logits(model.get(), GGML_TYPE_Q8_0, 2, GGML_TYPE_Q4_0);
            for (int drafts : {2, 3, 4, 5}) check_gdn_transactions(model.get(), GGML_TYPE_Q8_0, drafts, GGML_TYPE_Q4_0);
        }
        llama_kvmem_set_params(nullptr);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
