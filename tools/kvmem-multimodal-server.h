#pragma once

#include <set>
#include <numeric>

static void multimodal_validate_capacity(const ServerState & st, const kvmem_prompt & prompt, int end) {
    if (!st.kparams.enabled || !st.kparams.budget || !prompt.has_media()) return;
    const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
    const uint32_t budget = st.kparams.budget / block;
    std::vector<std::pair<uint32_t, uint32_t>> groups;
    for (const auto & range : prompt.media_ranges()) {
        const uint32_t lo = (range.first ? range.first - 1 : 0) / block;
        const uint32_t hi = (std::min<uint32_t>(end, range.second + 1) + block - 1) / block;
        if (!groups.empty() && lo < groups.back().second) groups.back().second = hi;
        else groups.emplace_back(lo, hi);
    }
    const uint32_t sink = std::max(1u, st.kparams.sink_tokens / block);
    for (const auto & group : groups) {
        if (group.second - group.first + std::min(group.first, sink) > budget)
            throw std::invalid_argument("image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget");
    }
    // Text suffixes may be trimmed after the first pass. Only an image group
    // that cannot fit intact with the sink is a hard capacity error.
}

// Included after the single-slot server state and stream helpers.
static MultimodalCheckpoint multimodal_checkpoint(ServerState & st, int row) {
    MultimodalCheckpoint result;
    result.row = row;
    if (st.mm_live_checkpoint && st.mm_live_row == row) {
        result.data = st.mm_live_checkpoint;
        ++st.mm_perf.shared;
        return result;
    }
    kvmem_scoped_ms timer(st.mm_perf.save_ms);
    ++st.mm_perf.saves;
    auto data = std::make_shared<MultimodalCheckpointData>();
    llama_synchronize(st.ctx);
    {
        kvmem_scoped_ms mean_timer(st.mm_perf.mean_ms);
        llama_kvmem_get_tail_mean(row, data->tail_mean);
    }
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t size = llama_state_seq_get_size_ext(st.ctx, 0, flags);
    data->recurrent.resize(size);
    if (llama_state_seq_get_data_ext(st.ctx, data->recurrent.data(), size, 0, flags) != size) {
        throw std::runtime_error("multimodal recurrent checkpoint failed");
    }
    {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        if (st.spec.ok && !common_speculative_get_state(st.spec.spec, 0, data->draft_carry)) {
            throw std::runtime_error("MTP carry checkpoint failed");
        }
    }
    data->accounting = st.mm_checkpoint_accounting;
    data->accounting->live_bytes += data->bytes();
    data->accounting->peak_bytes = std::max(data->accounting->peak_bytes, data->accounting->live_bytes);
    result.data = std::move(data);
    st.mm_live_checkpoint = result.data;
    return result;
}

static void multimodal_remember(ServerState & st, MultimodalCheckpoint checkpoint) {
    auto & entries = st.mm_checkpoints;
    entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto & entry) {
        return entry.row >= checkpoint.row;
    }), entries.end());
    entries.push_back(std::move(checkpoint));
    if (entries.size() > 4) {
        const auto media_count = std::count_if(entries.begin(), entries.end(), [](const auto & e) { return e.media_boundary; });
        auto victim = std::find_if(entries.begin(), entries.end(), [&](const auto & e) {
            return e.media_boundary == (media_count > 2);
        });
        entries.erase(victim == entries.end() ? entries.begin() : victim);
    }
}

static void multimodal_restore(ServerState & st, const MultimodalCheckpoint & checkpoint, bool truncate) {
    kvmem_scoped_ms timer(st.mm_perf.restore_ms);
    const bool live = st.mm_live_checkpoint == checkpoint.data && st.mm_live_row == checkpoint.row;
    if (!checkpoint.data) throw std::runtime_error("missing multimodal checkpoint data");
    if (live) ++st.mm_perf.restore_skips;
    else ++st.mm_perf.restores;
    llama_synchronize(st.ctx);
    if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
    llama_kvmem_decode_mean_flush();
    llama_kvmem_decode_mean_discard();
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const auto & data = checkpoint.data->recurrent;
    if (!live && llama_state_seq_set_data_ext(st.ctx, data.data(), data.size(), 0, flags) != data.size()) {
        throw std::runtime_error("multimodal recurrent restore failed");
    }
    if (!llama_kvmem_remove_logical(st.ctx, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted target rows");
    }
    if (st.spec.ctx_dft && !llama_kvmem_remove_logical(st.spec.ctx_dft, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted MTP rows");
    }
    if (st.spec.ok && !live) {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        common_speculative_set_state(st.spec.spec, 0, checkpoint.data->draft_carry);
    }
    if (truncate) {
        llama_kvmem_truncate_cached(checkpoint.row);
        llama_kvmem_set_tail_mean(checkpoint.row, checkpoint.data->tail_mean);
    }
    st.mm_live_row = checkpoint.row;
    st.mm_live_checkpoint = checkpoint.data;
}

static void multimodal_finish_request(ServerState & st) {
    if (st.mm_committed || !st.mm_rollback) return;
    try {
        llama_kvmem_set_replay(false);
        multimodal_restore(st, *st.mm_rollback, true);
        llama_kvmem_begin_cached_turn();
        st.mm_query.reset();
        st.mm_pending_query.reset();
        st.cached_prompt = st.mm_rollback_prompt;
        st.cached_tokens = st.cached_prompt ? st.cached_prompt->tokens : std::vector<llama_token>{};
        st.cached_tokens.resize(std::min(st.cached_tokens.size(), (size_t) st.mm_live_row));
        multimodal_remember(st, *st.mm_rollback);
        kvmem_diag("KVMEM_TRACE multimodal_rollback context=%p row=%d\n", (void *) st.ctx, st.mm_live_row);
        kvmem_diag("KVMEM_CHECKPOINT_ROLLBACK live_bytes=%zu peak_bytes=%zu\n",
                st.mm_checkpoint_accounting->live_bytes, st.mm_checkpoint_accounting->peak_bytes);
        st.mm_committed = true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        LOG_ERR("srv    KVMEM_TRACE multimodal_rollback_failed error=%s\n", e.what());
    }
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
}

static int multimodal_decode_span(ServerState & st, int begin, int end, bool replay, StreamIo * io) {
    const auto & prompt = *st.active_prompt;
    auto dispatch = [&](llama_batch batch) -> int {
        if (!stream_heartbeat(io)) return KVMEM_DECODE_ABORT;
        kvmem_diag("KVMEM_TRACE multimodal_decode context=%p rows=[%d,%d) model_pos=%d image=%d replay=%d\n",
                (void *) st.ctx, batch.logical_pos[0], batch.logical_pos[batch.n_tokens - 1] + 1,
                batch.pos[0], batch.token == nullptr, replay);
        const auto start = std::chrono::steady_clock::now();
        const bool diagnostic = llama_kvmem_get_transfer_stats().enabled;
        st.mm_live_checkpoint.reset();
        int rc = llama_decode(st.ctx, batch);
        if (diagnostic) {
            llama_synchronize(st.ctx);
            st.mm_perf.target_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        const auto draft_start = std::chrono::steady_clock::now();
        if (rc == 0 && st.spec.ok && !common_speculative_process(st.spec.spec, batch)) rc = -1;
        if (diagnostic && st.spec.ok) {
            llama_synchronize(st.spec.ctx_dft);
            st.mm_perf.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - draft_start).count();
        }
        llama_synchronize(st.ctx);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        (replay ? st.mm_perf.replay_ms : st.mm_perf.first_ms) += elapsed;
        kvmem_diag("KVMEM_TRACE multimodal_compute rows=%d elapsed_ms=%.3f image=%d replay=%d\n",
                batch.n_tokens, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                batch.token == nullptr, replay);
        if (rc == 0) {
            st.mm_live_row = batch.logical_pos[batch.n_tokens - 1] + 1;
            if (replay) st.mm_replayed += batch.n_tokens;
            else {
                const int tail = std::clamp(st.mm_lcp - batch.logical_pos[0], 0, batch.n_tokens);
                st.mm_tail_replayed += tail;
                if (batch.token) st.mm_new_text += batch.n_tokens - tail;
                else st.mm_new_image += batch.n_tokens - tail;
                // Count actual first-pass work, including reconstruction after the
                // reused checkpoint; exclude the later retrieval query replay.
                st.log.prefilled(batch.n_tokens, (int) prompt.tokens.size() - batch.logical_pos[0]);
            }
        }
        return rc;
    };
    int row = begin;
    while (row < end) {
        if (!stream_heartbeat(io)) return KVMEM_DECODE_ABORT;
        if (prompt.tokens[row] == LLAMA_TOKEN_NULL) {
            const int next = (int) prompt.media_end(row);
            if (next > end) throw std::runtime_error("prefill boundary splits an image");
            if (!replay) {
                auto checkpoint = multimodal_checkpoint(st, row);
                checkpoint.media_boundary = true;
                multimodal_remember(st, std::move(checkpoint));
            }
            const int rc = st.vision->decode(st.ctx, prompt, row, st.n_batch, dispatch);
            if (rc != 0) return rc;
            row = next;
            continue;
        }
        const int limit = std::min(end, row + st.n_batch);
        int next = row;
        while (next < limit && prompt.tokens[next] != LLAMA_TOKEN_NULL) ++next;
        std::vector<llama_pos> pos(next - row), logical(next - row);
        const auto pos0 = prompt.model_pos(row);
        for (int i = row; i < next; ++i) {
            pos[i - row] = pos0 + i - row;
            logical[i - row] = i;
        }
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.tokens.data()) + row, next - row);
        batch.pos = pos.data();
        batch.logical_pos = logical.data();
        std::vector<int32_t> n_seq(next - row, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seq_ids(next - row, &seq);
        std::vector<int8_t> outputs(next - row, 0);
        outputs.back() = !st.spec.ok;
        batch.n_seq_id = n_seq.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        const int rc = dispatch(batch);
        if (rc != 0) return rc;
        row = next;
    }
    return 0;
}

static bool run_prefill_multimodal(ServerState & st, StreamIo * io, int * n_cache_hit) {
    const auto started = std::chrono::steady_clock::now();
    st.mm_perf = {};
    const auto copies_before = llama_kvmem_get_transfer_stats();
    st.mm_checkpoint_accounting->peak_bytes = st.mm_checkpoint_accounting->live_bytes;
    st.mm_pending_query.reset();
    try {
        st.mm_error.clear();
        st.mm_error_status = 500;
        if (st.mm_reset_requested) {
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=explicit_cache_reset\n", (void *) st.ctx);
            memory_clear_all(st);
            st.mm_reset_requested = false;
        }
        st.mm_new_text = st.mm_new_image = st.mm_replayed = st.mm_tail_replayed = 0;
        if (st.vision) st.vision->reset_stats();
        const auto & prompt = *st.active_prompt;
        const int eval_end = (int) prompt.tokens.size() - (st.spec.ok ? 1 : 0);
        const int lcp = st.cached_prompt ? (int) prompt.common_prefix(*st.cached_prompt) : 0;
        st.mm_lcp = lcp;
        // Sequence checkpoints do not restore logits. Ordinary decoding must evaluate
        // at least one token; MTP evaluates the pending prompt token in spec_generate.
        const int keep = std::min({lcp, st.mm_live_row, eval_end - (st.spec.ok ? 0 : 1)});
        MultimodalCheckpoint base;
        bool found = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.select_checkpoint_ms);
            for (const auto & checkpoint : st.mm_checkpoints) {
                if (checkpoint.row <= keep && (!found || checkpoint.row > base.row)) {
                    base = checkpoint;
                    found = true;
                }
            }
        }
        if (!found) {
            // Shared template tokens do not identify a conversation. Like llama-server,
            // treat a missing recurrent checkpoint as a cache miss and evaluate the supplied prompt.
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=%s lcp=%d keep=%d cached_rows=%zu live_rows=%d oldest_checkpoint=%d checkpoint_count=%zu\n",
                    (void *) st.ctx, st.cached_prompt ? "no_recurrent_checkpoint" : "new_conversation",
                    lcp, keep, st.cached_tokens.size(), st.mm_live_row,
                    st.mm_checkpoints.empty() ? -1 : st.mm_checkpoints.front().row, st.mm_checkpoints.size());
            memory_clear_all(st);
            st.mm_lcp = 0; // No old rows survived the reset; count all evaluated rows as new.
            base = multimodal_checkpoint(st, 0);
        }
        st.mm_rollback = std::make_shared<MultimodalCheckpoint>(base);
        st.mm_rollback_prompt = st.cached_prompt ? st.cached_prompt->prefix(base.row) : nullptr;
        st.mm_committed = false;
        multimodal_restore(st, base, true);
        llama_kvmem_begin_cached_turn();
        std::vector<uint32_t> starts, ends;
        const auto ranges = prompt.media_ranges();
        for (const auto & range : ranges) {
            starts.push_back(range.first > 0 ? range.first - 1 : 0);
            ends.push_back(std::min<uint32_t>(prompt.tokens.size(), range.second + 1));
        }
        llama_kvmem_set_media_ranges(starts.data(), ends.data(), starts.size());
        const int user_begin = std::clamp(st.kparams.query_begin, 0, eval_end);
        const int user_end = std::clamp(st.kparams.query_end, user_begin, eval_end);
        const auto media = prompt.media_identity();
        const auto cached_query = st.mm_query;
        const bool same_query = st.query_policy_user && st.turn_query_exact && cached_query &&
            cached_query->begin == user_begin && cached_query->end == user_end &&
            cached_query->force == st.kparams.force_pos && cached_query->user == st.turn_last_user &&
            base.row >= user_end && cached_query->media == media &&
            prompt.common_prefix(*cached_query->prefix) >= (size_t) user_end;
        const bool capture_user = st.query_policy_user && st.turn_query_exact &&
            user_begin >= base.row && user_end > user_begin;
        int query = std::max(base.row, std::min(st.kparams.query_begin, eval_end));
        if (!ranges.empty()) query = std::max(query, (int) ranges.back().second);
        query = std::min(query, eval_end);
        const bool retrieve = st.kparams.enabled && st.kparams.method == 1 && query < eval_end;
        llama_kvmem_set_request_span(query, eval_end, st.kparams.force_pos);
        llama_kvmem_turn_spans spans;
        spans.query = {{query, eval_end}};
        spans.mandatory = {{query, eval_end}};
        spans.replay_begin = query;
        if (st.query_policy_user) {
            spans.query = capture_user || same_query
                ? std::vector<llama_kvmem_row_range>{{user_begin, user_end}}
                : std::vector<llama_kvmem_row_range>{{std::max(query, eval_end - st.query_max_tokens), eval_end}};
        }
        llama_kvmem_set_turn_spans(spans);
        std::string path = "legacy", reason = "legacy_requested";
        std::string reuse_fallback = "none";
        bool reused_query = false;
        const bool imported_query = same_query && st.kparams.enabled && st.kparams.method == 1 &&
            llama_kvmem_set_query(cached_query->state);
        if (imported_query) {
            kvmem_scoped_ms timer(st.mm_perf.decision_ms);
            if (llama_kvmem_can_append(eval_end, st.turn_generation_rows, false, reason)) {
                path = "keep_selected";
                reused_query = true;
            } else {
                auto select_spans = spans;
                // Q predates these tool observations. Do not let stale query
                // scores discard newly learned facts during a fast reselect.
                // If the observed tail no longer fits the selection budget,
                // use a fresh suffix probe instead. The old Q source itself
                // is not a replay dependency and need not consume mandatory slots.
                select_spans.mandatory = {{user_end, base.row}};
                const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
                if (base.row % block) select_spans.mandatory.push_back({base.row - 1, base.row});
                llama_kvmem_set_turn_spans(select_spans);
                const auto selection = llama_kvmem_preview_retrieval();
                if (llama_kvmem_selection_fits(selection, eval_end, st.turn_generation_rows)) {
                    {
                        kvmem_scoped_ms retrieval_timer(st.mm_perf.retrieval_ms);
                        llama_kvmem_apply_selection(selection);
                    }
                    llama_kvmem_begin_cached_turn_keep_query();
                    llama_kvmem_freeze_query(true);
                    reused_query = llama_kvmem_can_append(eval_end, st.turn_generation_rows, false, reason);
                    if (reused_query) path = "cached_q_reselect";
                } else reason = "observed_tail_or_append_exceeds_capacity";
            }
            if (reused_query) {
                st.mm_pending_query = cached_query;
                llama_kvmem_keep_selected();
            }
        }
        if (same_query && !reused_query) {
            reuse_fallback = imported_query ? reason : "invalid_query_state";
            llama_kvmem_reset_query();
            llama_kvmem_freeze_query(false);
            spans.query = {{std::max(query, eval_end - st.query_max_tokens), eval_end}};
            llama_kvmem_set_turn_spans(spans);
        }
        bool all_resident = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.decision_ms);
            all_resident = !reused_query && retrieve && st.query_replay_auto &&
                llama_kvmem_can_append(eval_end, st.turn_generation_rows, true, reason);
        }
        if (all_resident) llama_kvmem_keep_selected();
        if (n_cache_hit) *n_cache_hit = base.row;
        if (multimodal_decode_span(st, base.row, query, false, io) != 0) throw std::runtime_error("multimodal prefill failed or cancelled");
        auto query_checkpoint = multimodal_checkpoint(st, query);
        multimodal_remember(st, query_checkpoint);
        const auto probe_view = llama_kvmem_get_attention_view();
        if (multimodal_decode_span(st, query, eval_end, false, io) != 0) throw std::runtime_error("multimodal query prefill failed or cancelled");
        if (reused_query) {
            if (!llama_kvmem_commit_resident(false)) throw std::runtime_error("incomplete KV after query continuation");
        } else if (retrieve) {
            bool replay = true;
            bool replay_fits = true;
            {
                kvmem_scoped_ms timer(st.mm_perf.retrieval_ms);
                if (all_resident && llama_kvmem_commit_resident()) {
                    path = "all_resident";
                    replay = false;
                } else {
                    const auto selection = llama_kvmem_preview_retrieval();
                    if (st.query_replay_auto && llama_kvmem_commit_unchanged(probe_view, selection)) {
                        path = "unchanged_selection";
                        reason = "same_attention_view";
                        replay = false;
                    } else {
                        path = "query_replay";
                        if (st.query_replay_auto) reason = "selection_or_attention_view_changed";
                        // Check the actual selection: complete image groups also
                        // consume slots, so a text-only budget estimate is insufficient.
                        const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
                        for (uint32_t id = query / block; id < ((uint32_t) eval_end + block - 1) / block; ++id) {
                            if (!std::binary_search(selection.blocks.begin(), selection.blocks.end(), id)) {
                                replay_fits = false;
                                break;
                            }
                        }
                        llama_kvmem_apply_selection(selection);
                    }
                }
            }
            if (replay && !replay_fits) {
                // Selection may trim the mandatory suffix when a long tool history
                // exceeds the retrieval budget. Keep the completed first-pass
                // recurrent state, logits and MTP carry; restoring the query
                // checkpoint would require replaying rows with no resident slot.
                path = "query_replay_skipped";
                reason = "replay_exceeds_budget";
                replay = false;
                kvmem_diag("KVMEM_TRACE replay_skipped reason=over_budget query=[%d,%d) replay_rows=%d budget_tokens=%u\n",
                        query, eval_end, eval_end - query, st.kparams.budget);
            }
            if (replay) {
                multimodal_restore(st, query_checkpoint, false);
                llama_kvmem_set_replay(true);
                const int rc = multimodal_decode_span(st, query, eval_end, true, io);
                llama_kvmem_set_replay(false);
                if (rc != 0) throw std::runtime_error("multimodal query replay failed or cancelled");
            }
        }
        if (!reused_query && capture_user && st.kparams.enabled && st.kparams.method == 1) {
            auto saved = std::make_shared<MultimodalQuery>();
            if (llama_kvmem_get_query(saved->state) &&
                    *std::max_element(saved->state.count.begin(), saved->state.count.end()) == (uint32_t) (user_end - user_begin)) {
                saved->begin = user_begin;
                saved->end = user_end;
                saved->force = st.kparams.force_pos;
                saved->user = st.turn_last_user;
                saved->media = media;
                saved->prefix = prompt.prefix(user_end);
                st.mm_pending_query = std::move(saved);
            }
        }
        const auto & counts = st.mm_pending_query ? st.mm_pending_query->state.count : std::vector<uint32_t>{};
        const uint32_t q_rows = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());
        const char * source = !st.query_policy_user ? "legacy_suffix" : reused_query ? "cached_user" :
            capture_user ? "user" : "bootstrap_suffix";
        if (!retrieve && !reused_query) reason = "no_query_suffix";
        kvmem_diag("KVMEM_PREFILL_DECISION path=%s reason=%s reuse_fallback=%s append=[%d,%d) query=[%d,%d) feature=[%d,%d) query_source=%s query_reused=%d q_rows=%u replay_rows=%u decision_ms=%.3f\n",
                path.c_str(), reason.c_str(), reuse_fallback.c_str(), base.row, eval_end, query, eval_end,
                spans.query.front().begin, spans.query.back().end, source, reused_query, q_rows, st.mm_replayed, st.mm_perf.decision_ms);
        llama_kvmem_pin_working_set();
        multimodal_remember(st, multimodal_checkpoint(st, eval_end));
        std::vector<uint8_t> carry;
        llama_pos synced = 0;
        if (st.spec.ok) {
            common_speculative_get_state(st.spec.spec, 0, carry);
            if (carry.size() < sizeof(synced)) throw std::runtime_error("MTP carry missing");
            std::memcpy(&synced, carry.data(), sizeof(synced));
            if (synced != eval_end) throw std::runtime_error("MTP has unsynchronized visual rows");
        }
        kvmem_diag("KVMEM_TRACE multimodal_prefill context=%p prefix_hit_rows=%d lcp=%d new_text_rows=%u new_image_rows=%u replayed_rows=%u vision_encode_calls=%u encoder_ms=%.2f logical_cursor=%d model_cursor=%d mtp_synced_rows=%d cached_tail_rows=%u replay_reason=%s embedding_cache_bytes=%zu checkpoint_bytes=%zu\n",
                (void *) st.ctx, base.row, lcp, st.mm_new_text, st.mm_new_image, st.mm_replayed,
                st.vision ? st.vision->encode_calls : 0, st.vision ? st.vision->encode_ms : 0.0,
                eval_end, prompt.model_pos(eval_end), synced, st.mm_tail_replayed, st.mm_replayed ? reason.c_str() : "none",
                st.vision ? st.vision->cache_bytes() : 0,
                std::accumulate(st.mm_checkpoints.begin(), st.mm_checkpoints.end(), size_t(0),
                    [](size_t n, const auto & c) { return n + c.data->bytes(); }));
        std::set<const MultimodalCheckpointData *> unique;
        size_t unique_bytes = 0;
        size_t ref_bytes = 0;
        auto count = [&](const MultimodalCheckpoint & cp) {
            if (cp.data && unique.insert(cp.data.get()).second) {
                unique_bytes += cp.data->bytes();
                ref_bytes += cp.data->bytes() * cp.data.use_count();
            }
        };
        for (const auto & cp : st.mm_checkpoints) count(cp);
        if (st.mm_rollback) count(*st.mm_rollback);
        count(base);
        count(query_checkpoint);
        const auto & p = st.mm_perf;
        kvmem_diag("KVMEM_PREFILL_PERF total_ms=%.3f first_ms=%.3f replay_ms=%.3f retrieval_ms=%.3f checkpoint_select_ms=%.3f checkpoint_save_ms=%.3f checkpoint_restore_ms=%.3f mean_nested_ms=%.3f carry_nested_ms=%.3f saves=%u restores=%u shared=%u restore_skips=%u checkpoint_unique_bytes=%zu\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count(),
                p.first_ms, p.replay_ms, p.retrieval_ms, p.select_checkpoint_ms, p.save_ms, p.restore_ms,
                p.mean_ms, p.carry_ms, p.saves, p.restores, p.shared, p.restore_skips, unique_bytes);
        kvmem_diag("KVMEM_CHECKPOINT_MEMORY ref_bytes=%zu unique_bytes=%zu live_bytes=%zu peak_bytes=%zu\n",
                ref_bytes, unique_bytes, st.mm_checkpoint_accounting->live_bytes, st.mm_checkpoint_accounting->peak_bytes);
        if (copies_before.enabled) {
            const auto copies = llama_kvmem_get_transfer_stats();
            fprintf(stderr, "KVMEM_PREFILL_PHASE path=%s first_ms=%.3f replay_ms=%.3f retrieval_ms=%.3f replay_rows=%u query=[%d,%d)\n",
                    path.c_str(), p.first_ms, p.replay_ms, p.retrieval_ms,
                    st.mm_replayed, query, eval_end);
            fprintf(stderr, "KVMEM_PREFILL_DIAGNOSTIC target_ms=%.3f draft_ms=%.3f extra_sync=1 adapter_h2d_bytes=%llu adapter_d2h_bytes=%llu adapter_d2d_bytes=%llu h2d_calls=%llu d2h_calls=%llu d2d_calls=%llu\n",
                    p.target_ms, p.draft_ms,
                    (unsigned long long) (copies.bytes[0] - copies_before.bytes[0]),
                    (unsigned long long) (copies.bytes[1] - copies_before.bytes[1]),
                    (unsigned long long) (copies.bytes[2] - copies_before.bytes[2]),
                    (unsigned long long) (copies.calls[0] - copies_before.calls[0]),
                    (unsigned long long) (copies.calls[1] - copies_before.calls[1]),
                    (unsigned long long) (copies.calls[2] - copies_before.calls[2]));
        }
        return true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        if (dynamic_cast<const std::invalid_argument *>(&e)) st.mm_error_status = 400;
        LOG_ERR("srv    KVMEM_TRACE multimodal_error error=%s\n", e.what());
        multimodal_finish_request(st);
        return false;
    }
}

static void multimodal_commit(ServerState & st, const std::vector<llama_token> & gen) {
    st.cached_prompt = st.active_prompt->with_generated(gen);
    st.mm_live_row = st.kparams.enabled ? (int) llama_kvmem_store_n_tokens()
        : st.mm_live_row;
    multimodal_remember(st, multimodal_checkpoint(st, st.mm_live_row));
    st.mm_query = st.mm_pending_query;
    st.mm_pending_query.reset();
    st.mm_committed = true;
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
    kvmem_diag("KVMEM_CHECKPOINT_COMMIT live_bytes=%zu peak_bytes=%zu\n",
            st.mm_checkpoint_accounting->live_bytes, st.mm_checkpoint_accounting->peak_bytes);
}

static int multimodal_decode_generated(ServerState & st, llama_token id, int row) {
    st.mm_live_checkpoint.reset();
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_pos logical = row;
    llama_pos pos = st.active_prompt->model_pos(row);
    batch.logical_pos = &logical;
    batch.pos = &pos;
    const int rc = llama_decode(st.ctx, batch);
    if (rc == 0) st.mm_live_row = row + 1;
    return rc;
}
