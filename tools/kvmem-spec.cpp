#include "llama-kvmem-diag.h"
#include "kvmem-spec.h"
#include "llama-kvmem-hooks.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

struct gdn_replay_transaction {
    llama_context * ctx;
    bool active = false;
    ~gdn_replay_transaction() {
        if (active) llama_kvmem_gdn_replay_commit(ctx, 0);
    }
    bool commit(uint32_t keep) {
        const bool ok = llama_kvmem_gdn_replay_commit(ctx, keep);
        active = false;
        return ok;
    }
};

ggml_type kvmem_parse_cache_type(const char * s, bool * ok) {
    if (ok) {
        *ok = true;
    }
    if (!s) {
        if (ok) {
            *ok = false;
        }
        return GGML_TYPE_F16;
    }
    if (std::strcmp(s, "f16") == 0 || std::strcmp(s, "fp16") == 0) {
        return GGML_TYPE_F16;
    }
    if (std::strcmp(s, "f32") == 0 || std::strcmp(s, "fp32") == 0) {
        return GGML_TYPE_F32;
    }
    if (std::strcmp(s, "q8_0") == 0 || std::strcmp(s, "q8") == 0) {
        return GGML_TYPE_Q8_0;
    }
    if (std::strcmp(s, "q4_0") == 0 || std::strcmp(s, "q4") == 0) {
        return GGML_TYPE_Q4_0;
    }
    if (std::strcmp(s, "q5_0") == 0 || std::strcmp(s, "q5") == 0) {
        return GGML_TYPE_Q5_0;
    }
    if (ok) {
        *ok = false;
    }
    return GGML_TYPE_F16;
}

bool kvmem_cache_types_ok(ggml_type type_k, ggml_type type_v) {
    if (ggml_is_quantized(type_k) || ggml_is_quantized(type_v)) {
        const auto supported_quant = [](ggml_type type) {
            return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q4_0;
        };
        return supported_quant(type_k) && supported_quant(type_v);
    }
    return true;
}

bool kvmem_spec_start(kvmem_spec_session & sess,
                      llama_model * model_tgt,
                      llama_context * ctx_tgt,
                      const kvmem_spec_opts & opts) {
    sess = kvmem_spec_session{};
    if (!model_tgt || !ctx_tgt) {
        fprintf(stderr, "kvmem_spec_start: missing target model/context\n");
        return false;
    }

    common_init();

    common_params p;
    p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    p.speculative.draft.n_max = opts.n_max;
    p.speculative.draft.n_min = opts.n_min;
    p.speculative.draft.p_min = opts.p_min;
    p.speculative.draft.n_gpu_layers = opts.n_gpu_layers;
    if (!opts.draft_model.empty()) {
        p.speculative.draft.mparams.path = opts.draft_model;
    }
    p.n_gpu_layers = opts.n_gpu_layers;
    p.n_ctx = opts.n_ctx > 0 ? opts.n_ctx : (int) llama_n_ctx(ctx_tgt);
    p.n_batch = opts.n_batch;
    p.n_ubatch = opts.n_ubatch > 0 ? opts.n_ubatch : opts.n_batch;
    p.n_parallel = 1;
    if (opts.n_threads > 0) p.cpuparams.n_threads = opts.n_threads;
    if (opts.n_threads_batch > 0) p.cpuparams_batch.n_threads = opts.n_threads_batch;
    p.flash_attn_type = opts.flash_attn;
    p.n_outputs_max = 1 + std::max(0, opts.n_max);
    p.n_outputs_max_per_seq = p.n_outputs_max;
    p.cache_type_k = opts.type_k;
    p.cache_type_v = opts.type_v;
    // The speculative conversion reads the draft cache types, not the base types.
    p.speculative.draft.cache_type_k = opts.draft_type == GGML_TYPE_COUNT ? opts.type_k : opts.draft_type;
    p.speculative.draft.cache_type_v = opts.draft_type == GGML_TYPE_COUNT ? opts.type_v : opts.draft_type;

    common_params p_dft = common_base_params_to_speculative(p);
    sess.init = common_speculative_init_from_params(p_dft, model_tgt, ctx_tgt);
    sess.ctx_dft = sess.init ? sess.init->context() : nullptr;
    if (!sess.ctx_dft) {
        fprintf(stderr,
                "kvmem_spec_start: MTP draft context is null "
                "(GGUF missing nextn / qwen35.nextn_predict_layers?)\n");
        return false;
    }

    kvmem_diag("KVMEM_CONTEXT draft threads=%d threads_batch=%d ubatch=%u flash_attn_requested=%s\n",
            llama_n_threads(sess.ctx_dft), llama_n_threads_batch(sess.ctx_dft), llama_n_ubatch(sess.ctx_dft),
            llama_flash_attn_type_name(p_dft.flash_attn_type));
    sess.spec_params = std::move(p);
    sess.spec_params.speculative.draft.ctx_tgt = ctx_tgt;
    sess.spec_params.speculative.draft.ctx_dft = sess.ctx_dft;
    sess.spec = common_speculative_init(sess.spec_params.speculative, 1);
    if (!sess.spec) {
        fprintf(stderr, "kvmem_spec_start: common_speculative_init failed\n");
        return false;
    }

    // Prefer llama.cpp n_rs_seq GPU planes (qw3-style per-token GDN rollback).
    // Host PARTIAL_ONLY is only the fallback when planes are missing or the
    // draft is longer than n_rs_seq. Do not probe can_seq_rm on live KVMem:
    // hybrid seq_rm skips GDN on query-replay holes, which would look like PART.
    sess.n_rs_tgt = llama_n_rs_seq(ctx_tgt);
    sess.use_gdn_replay = llama_kvmem_gdn_replay_enabled();
    sess.use_ckpt_dft = false;
    if (sess.use_gdn_replay) {
        sess.use_ckpt_tgt = false;
        kvmem_diag("KVMEM_TRACE spec_ckpt tgt=REPLAY (FP32 GDN records)\n");
    } else if (sess.n_rs_tgt > 0) {
        sess.use_ckpt_tgt = false;
        kvmem_diag("KVMEM_TRACE spec_ckpt tgt=RS n_rs_seq=%u (GPU GDN planes; host ckpt if draft > n_rs)\n",
                sess.n_rs_tgt);
    } else if (llama_kvmem_has_recurrent()) {
        sess.use_ckpt_tgt = true;
        kvmem_diag("KVMEM_TRACE spec_ckpt tgt=PARTIAL_ONLY (hybrid GDN; n_rs_seq=0 host fallback)\n");
    } else if (!opts.kvmem_enabled) {
        sess.use_ckpt_tgt =
                common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        sess.use_ckpt_dft =
                common_context_can_seq_rm(sess.ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        kvmem_diag("KVMEM_TRACE spec_ckpt tgt=%d dft=%d (vanilla probe)\n",
                (int) sess.use_ckpt_tgt, (int) sess.use_ckpt_dft);
    } else {
        sess.use_ckpt_tgt = false;
        kvmem_diag("KVMEM_TRACE spec_ckpt tgt=0 (dense KVMem seq_rm)\n");
    }

    sess.ok = true;
    kvmem_diag("KVMEM_TRACE spec_start type=draft-mtp n_max=%d p_min=%.3f dft=%p\n",
            opts.n_max, opts.p_min, (void *) sess.ctx_dft);
    return true;
}

void kvmem_spec_stop(kvmem_spec_session & sess) {
    if (sess.spec) {
        common_speculative_free(sess.spec);
        sess.spec = nullptr;
    }
    sess.init.reset();
    sess.ctx_dft = nullptr;
    sess.ok = false;
}

int kvmem_spec_decode_span(llama_context * ctx,
                           common_speculative * spec,
                           const llama_token * toks,
                           int pos0, int pos1, int n_batch,
                           const char * what,
                           const std::function<bool()> & abort) {
    if (pos0 >= pos1) {
        return 0;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_pos = pos0;
    while (n_pos < pos1) {
        if (abort && abort()) {
            kvmem_diag("KVMEM_TRACE stream_abort phase=prefill pos=%d what=%s\n",
                    n_pos, what ? what : "");
            llama_batch_free(batch);
            return KVMEM_DECODE_ABORT;
        }
        const int n = std::min(n_batch, pos1 - n_pos);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, toks[n_pos + i], n_pos + i, { 0 }, false);
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n",
                    what, rc, n_pos, n);
            llama_batch_free(batch);
            return rc;
        }
        if (spec && !common_speculative_process(spec, batch)) {
            fprintf(stderr, "common_speculative_process(%s) failed at pos=%d n=%d\n",
                    what, n_pos, n);
            llama_batch_free(batch);
            return 1;
        }
        n_pos += n;
    }
    llama_batch_free(batch);
    return 0;
}

kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        float temp,
        kvmem_spec_on_token on_token) {
    common_params_sampling sparams;
    if (temp <= 0.0f) {
        sparams.temp = 0.0f;
        sparams.min_p = 0.0f;
        sparams.top_p = 1.0f;
        sparams.top_k = 0;
        sparams.penalty_repeat = 1.0f;
        sparams.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    } else {
        sparams.temp = temp;
    }
    return kvmem_spec_generate(ctx_tgt, model_tgt, sess, prompt, n_predict,
                               std::move(sparams), std::move(on_token));
}

kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        common_params_sampling sparams,
        kvmem_spec_on_token on_token,
        const std::function<bool()> & abort,
        llama_pos position_offset) {
    kvmem_spec_gen_stats st;
    if (!sess.ok || !sess.spec || prompt.empty() || n_predict <= 0) {
        st.failed = true;
        return st;
    }

    llama_kvmem_end_prefill_capture();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    llama_context * ctx_dft = sess.ctx_dft;
    common_speculative * spec = sess.spec;
    const llama_seq_id seq_id = 0;

    common_sampler_ptr smpl;
    try {
        smpl.reset(common_sampler_init(model_tgt, sparams));
    } catch (const std::exception & e) {
        fprintf(stderr, "kvmem_spec_generate sampler init failed: %s\n", e.what());
        st.failed = true;
        return st;
    }
    if (!smpl) {
        st.failed = true;
        return st;
    }

    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_token id_last = prompt.back();
    int n_past = (int) prompt.size() - 1;
    llama_batch batch_tgt = llama_batch_init((int) llama_n_batch(ctx_tgt), 0, 1);
    llama_tokens draft;
    std::vector<llama_pos> logical_positions(llama_n_batch(ctx_tgt));
    batch_tgt.logical_pos = logical_positions.data();
    std::vector<uint8_t> driver_ckpt;
    common_prompt_checkpoint ckpt;
    bool has_eos = false;
    int64_t verify_us = 0, fold_us = 0;
    uint64_t verify_calls = 0, committed_rows = 0;

    while (st.n_gen < n_predict && !has_eos) {
        if (abort && abort()) {
            kvmem_diag("KVMEM_TRACE stream_abort phase=spec_gen n_gen=%d\n", st.n_gen);
            break;
        }
        if (draft.empty()) {
            common_speculative_get_state(spec, seq_id, driver_ckpt);
            llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
            ckpt.update_pos(
                    (int64_t) prompt_tgt.size(),
                    mem_tgt ? llama_memory_seq_pos_min(mem_tgt, seq_id) : 0,
                    mem_tgt ? llama_memory_seq_pos_max(mem_tgt, seq_id) : 0);

            if (sess.use_ckpt_dft && ctx_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            n_draft_max = std::min(n_draft_max, n_predict - st.n_gen - 1);
            n_draft_max = std::max(n_draft_max, 0);

            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .n_past     = */ n_past + position_offset,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft,
                /* .n_past_logical = */ n_past,
            };
            common_speculative_draft(spec);

            const bool host_ckpt = sess.use_ckpt_tgt
                    || (sess.n_rs_tgt > 0 && draft.size() > sess.n_rs_tgt);
            if (!draft.empty() && host_ckpt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            if (ctx_dft) {
                if (sess.use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    llama_kvmem_remove_logical(ctx_dft, (llama_pos) ckpt.n_tokens, -1);
                }
            }
        }

        common_batch_clear(batch_tgt);
        logical_positions[0] = n_past;
        common_batch_add(batch_tgt, id_last, n_past++ + position_offset, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            logical_positions[batch_tgt.n_tokens] = n_past + (llama_pos) i;
            common_batch_add(batch_tgt, draft[i], n_past + position_offset + (llama_pos) i, { seq_id }, true);
        }

        gdn_replay_transaction transaction{ctx_tgt};
        if (sess.use_gdn_replay) {
            transaction.active = llama_kvmem_gdn_replay_begin(batch_tgt.pos[0], batch_tgt.n_tokens);
            if (!transaction.active) {
                fprintf(stderr, "GDN replay begin failed at pos=%d width=%d\n", batch_tgt.pos[0], batch_tgt.n_tokens);
                st.failed = true;
                break;
            }
        }
        const auto verify_start = ggml_time_us();
        const int rc = llama_decode(ctx_tgt, batch_tgt);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(spec verify) failed rc=%d n_draft=%zu\n",
                    rc, draft.size());
            st.failed = true;
            break;
        }
        if (!common_speculative_process(spec, batch_tgt)) {
            fprintf(stderr, "common_speculative_process(verify) failed\n");
            st.failed = true;
            break;
        }

        const size_t n_draft = draft.size();
        const bool host_ckpt = sess.use_ckpt_tgt
                || (sess.n_rs_tgt > 0 && n_draft > sess.n_rs_tgt);
        common_sampler_ptr smpl_save;
        if (host_ckpt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        verify_us += ggml_time_us() - verify_start;
        ++verify_calls;
        ids.resize(std::min(ids.size(), (size_t) (n_predict - st.n_gen)));
        for (size_t i = 0; i < ids.size(); ++i) {
            if (llama_vocab_is_eog(vocab, ids[i])) {
                ids.resize(i + 1);
                break;
            }
        }
        const bool restore = host_ckpt && !ids.empty() && ids.size() - 1 < n_draft;
        kvmem_diag("KVMEM_TRACE spec_verify n_draft=%zu n_accept=%zu restore=%d pos=%d ckpt_bytes=%zu n_rs=%u\n",
                n_draft, ids.size() > 0 ? ids.size() - 1 : 0, (int) restore, n_past - 1,
                ckpt.data_tgt.size(), sess.n_rs_tgt);

        if (restore) {
            ++st.n_restore;
            llama_kvmem_decode_mean_discard();
            draft = std::move(ids);
            ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
            if (mem_tgt) {
                llama_kvmem_remove_logical(ctx_tgt, (llama_pos) ckpt.n_tokens, -1);
            }
            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    llama_kvmem_remove_logical(ctx_dft, (llama_pos) ckpt.n_tokens, -1);
                }
            }
            prompt_tgt.resize((size_t) ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            llama_kvmem_truncate_cached(n_past);
            if (!driver_ckpt.empty()) common_speculative_set_state(spec, seq_id, driver_ckpt);
            continue;
        }

        if (ids.empty() || (sess.use_gdn_replay && abort && abort())) {
            if (sess.use_gdn_replay) {
                if (!transaction.commit(0)) st.failed = true;
                n_past = logical_positions[0];
                if (!llama_kvmem_remove_logical(ctx_tgt, n_past, -1) ||
                        (ctx_dft && !llama_kvmem_remove_logical(ctx_dft, n_past, -1))) st.failed = true;
                llama_kvmem_truncate_cached(n_past);
                if (!driver_ckpt.empty()) common_speculative_set_state(spec, seq_id, driver_ckpt);
            }
            llama_kvmem_decode_mean_discard();
            break;
        }
        if (sess.use_gdn_replay) {
            const auto fold_start = ggml_time_us();
            const bool ok = transaction.commit((uint32_t) ids.size());
            fold_us += ggml_time_us() - fold_start;
            if (!ok) {
                fprintf(stderr, "GDN replay commit failed\n");
                st.failed = true;
                break;
            }
        }
        committed_rows += ids.size();
        llama_kvmem_decode_mean_commit((uint32_t) ids.size());
        common_speculative_accept(spec, seq_id, (uint16_t) (ids.size() - 1));
        n_past += (int) ids.size() - 1;
        st.n_drafted += (int) n_draft;
        st.n_accept += (int) ids.size() - 1;

        if (sess.use_gdn_replay) {
            if (!llama_kvmem_remove_logical(ctx_tgt, n_past, -1) ||
                    (ctx_dft && !llama_kvmem_remove_logical(ctx_dft, n_past, -1))) {
                st.failed = true;
                break;
            }
            llama_kvmem_truncate_cached(n_past);
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }
            const std::string piece = common_token_to_piece(ctx_tgt, id_last);
            if (on_token) {
                on_token(id_last, piece, i + 1 < ids.size());
            }
            ++st.n_gen;
            if (st.n_gen >= n_predict) {
                break;
            }
        }

        draft.clear();
        if (!sess.use_gdn_replay) {
            llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
            if (mem_tgt) {
                llama_kvmem_remove_logical(ctx_tgt, n_past, -1);
            }
            if (ctx_dft) {
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    llama_kvmem_remove_logical(ctx_dft, n_past, -1);
                }
            }
        }
        if (!sess.use_gdn_replay) llama_kvmem_truncate_cached(n_past);
    }

    kvmem_diag("KVMEM_TRACE spec_stats n_gen=%d n_drafted=%d n_accept=%d n_restore=%d accept_pct=%.1f\n",
            st.n_gen, st.n_drafted, st.n_accept, st.n_restore,
            st.n_drafted > 0 ? 100.0 * st.n_accept / st.n_drafted : 0.0);
    const char * perf = std::getenv("KVMEM_PERF");
    if (perf && perf[0] && std::strcmp(perf, "0") != 0) {
        fprintf(stderr, "KVMEM_SPEC_STATS n_gen=%d n_drafted=%d n_accept=%d n_restore=%d\n",
                st.n_gen, st.n_drafted, st.n_accept, st.n_restore);
    }
    kvmem_diag("KVMEM_GDN_PERF mode=%s verify_calls=%llu committed_rows=%llu verify_ms=%.3f fold_ms=%.3f\n",
            sess.use_gdn_replay ? "replay" : "snapshots", (unsigned long long) verify_calls,
            (unsigned long long) committed_rows, verify_us / 1000.0, fold_us / 1000.0);
    llama_kvmem_decode_mean_flush();
    llama_kvmem_decode_mean_discard();
    common_speculative_print_stats(spec);

    st.n_past = n_past;
    llama_batch_free(batch_tgt);
    return st;
}
