#pragma once

// Included after ServerState and the existing model/cache helpers.
// Transport and generation are shared; each response policy owns its wire format.
template <class Response>
static void kvmem_execute_chat(ServerState &st, const json &default_params, const httplib::Request &req,
                               httplib::Response &res, ChatRequest cr,
                               const std::vector<std::vector<uint8_t>> &media_files, bool cache_reset,
                               Response response, std::time_t created) {
    std::string err;
    auto slot = std::make_shared<kvmem_server_slot_guard>(st.mu, st.progress);
    if (req.is_connection_closed && req.is_connection_closed()) return;
    st.mm_reset_requested = cache_reset;

    const auto inputs = kvmem_chat_inputs(cr, st.tmpls.get());
    common_chat_params formatted;
    try {
        formatted = common_chat_templates_apply(st.tmpls.get(), inputs);
    } catch (const std::exception &e) {
        res.status = 400;
        response.error(res, res.status, std::string("chat template: ") + e.what());
        return;
    }
    const std::string &prompt = formatted.prompt;
    std::shared_ptr<kvmem_prompt> parsed_prompt;
    try {
        parsed_prompt = media_files.empty() ? std::make_shared<kvmem_prompt>(tokenize_text(st.vocab, prompt, true))
                                            : st.vision->tokenize(prompt, media_files);
    } catch (const std::exception &e) {
        res.status = 400;
        response.error(res, res.status, e.what());
        return;
    }
    st.active_prompt = parsed_prompt;
    st.turn_generation_rows = (uint32_t)std::min<uint64_t>(
        UINT32_MAX, (uint64_t)std::max(0, cr.max_tokens) + (st.spec.ok ? std::max(0, st.spec_n_max) + 1u : 0u));
    auto toks = parsed_prompt->tokens;
    if (toks.empty()) {
        res.status = 400;
        response.error(res, res.status, "empty prompt");
        return;
    }
    if ((int)toks.size() + cr.max_tokens > (int)llama_n_ctx(st.ctx)) {
        res.status = 400;
        response.error(res, res.status, "prompt + max_tokens exceeds n_ctx");
        return;
    }

    int qbegin = cr.query_begin;
    int qend = cr.query_end;
    st.turn_query_exact = false;
    st.turn_last_user = cr.last_user;
    if (qbegin < 0 || qend < 0) {
        derive_query_span(st, prompt, cr.last_user, toks, qbegin, qend);
    }
    if (st.query_policy_user) {
        st.turn_query_exact = cr.query_begin >= 0 && cr.query_end > cr.query_begin && cr.query_end <= (int)toks.size();
        if (cr.query_begin < 0 && cr.query_end < 0) {
            st.turn_query_exact = derive_native_query_span(st, prompt, inputs, *parsed_prompt, qbegin, qend);
        }
    }
    if (parsed_prompt->has_media() && cr.query_begin < 0 && !st.turn_query_exact) {
        // The final text question follows native visual chunks and their boundaries.
        int last_media_end = 0;
        for (const auto &range : parsed_prompt->media_ranges())
            last_media_end = range.second;
        qend = (int)toks.size() - (st.spec.ok ? 1 : 0);
        qbegin = std::max(last_media_end, qend - st.query_max_tokens);
    }
    clamp_query_span(st, qbegin, qend);
    if (st.turn_query_exact &&
        std::find(toks.begin() + qbegin, toks.begin() + qend, LLAMA_TOKEN_NULL) != toks.begin() + qend) {
        st.turn_query_exact = false;
        kvmem_diag("KVMEM_TRACE query_loc fallback=explicit_span_contains_media\n");
    }
    try {
        multimodal_validate_capacity(st, *parsed_prompt, st.query_policy_user ? (int)toks.size() : qbegin,
                                     (int)toks.size());
    } catch (const std::exception &e) {
        res.status = 400;
        response.error(res, res.status, e.what());
        return;
    }
    const int force = force_pos_from_substr(st.vocab, toks, cr.force_substr);
    st.kparams.query_begin = qbegin;
    st.kparams.query_end = qend;
    st.kparams.force_pos = force;
    if (st.kparams.enabled) {
        llama_kvmem_set_request_span(qbegin, qend, force);
    }
    int n_tool_hist = 0;
    for (const auto &m : cr.msgs) {
        if (m.role == "tool" || !m.tool_calls.empty()) {
            n_tool_hist++;
        }
    }
    bool prompt_has_tool = false;
    for (const auto &t : cr.tools) {
        if (!t.name.empty() && prompt.find(t.name) != std::string::npos) {
            prompt_has_tool = true;
            break;
        }
    }
    kvmem_diag("KVMEM_TRACE n_prompt=%d query=[%d,%d) force_pos=%d last_user_chars=%zu\n", (int)toks.size(), qbegin,
               qend, force, cr.last_user.size());
    kvmem_diag("KVMEM_TRACE chat_parse n_msg=%zu n_tools=%zu tool_choice=%s tool_hist=%d "
               "prompt_has_tool=%d grammar_bytes=%zu think=%d reasoning=%s parser_bytes=%zu\n",
               cr.msgs.size(), cr.tools.size(), tool_choice_cstr(cr.tool_choice), n_tool_hist, (int)prompt_has_tool,
               formatted.grammar.size(), (int)cr.enable_thinking, common_reasoning_format_name(inputs.reasoning_format),
               formatted.parser.size());

    const std::string request_id = kvmem_chat_request_id();
    llama_context *ctx = st.ctx;
    const llama_vocab *vocab = st.vocab;

    common_params_sampling sparams = make_chat_sampling(vocab, formatted, cr);
    if (!kvmem_chat_reasoning_budget_supported(sparams, cr.enable_thinking, err)) {
        res.status = 400;
        response.error(res, res.status, err);
        return;
    }
    kvmem_diag("KVMEM_TRACE sampling thinking=%d temperature=%.6g top_p=%.6g top_k=%d min_p=%.6g "
               "presence_penalty=%.6g frequency_penalty=%.6g repetition_penalty=%.6g seed=%u\n",
               (int)cr.enable_thinking, sparams.temp, sparams.top_p, sparams.top_k, sparams.min_p,
               sparams.penalty_present, sparams.penalty_freq, sparams.penalty_repeat, sparams.seed);
    std::vector<std::string> stops = cr.stop;
    stops.insert(stops.end(), formatted.additional_stops.begin(), formatted.additional_stops.end());
    kvmem_diag("KVMEM_TRACE chat_sample grammar_type=%s lazy=%d n_trig=%zu gen_prompt_bytes=%zu "
               "think_start_bytes=%zu think_end_n=%zu rbudget=%d start_toks=%zu end_seqs=%zu forced_toks=%zu\n",
               grammar_type_cstr(sparams.grammar.type), (int)sparams.grammar_lazy, sparams.grammar_triggers.size(),
               sparams.generation_prompt.size(), formatted.thinking_start_tag.size(),
               formatted.thinking_end_tags.size(), sparams.reasoning_budget_tokens,
               sparams.reasoning_budget_start.size(), sparams.reasoning_budget_end.size(),
               sparams.reasoning_budget_forced.size());
    const bool parse_tools = !cr.tools.empty() && cr.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    response.init(formatted, parse_tools, request_id, st.model_name, (int)toks.size(), created);

    bool use_spec = st.spec.ok;
    if (use_spec && !sparams.grammar.empty()) {
        try {
            common_params_sampling probe = sparams;
            common_sampler_ptr test(common_sampler_init(st.model, probe));
            if (!test) {
                use_spec = false;
            }
        } catch (const std::exception &e) {
            fprintf(stderr, "KVMEM_TRACE spec sampler init failed (%s); greedy fallback\n", e.what());
            use_spec = false;
        }
    }

    if ((st.vision || st.query_policy_user) && st.spec.ok && !use_spec) {
        res.status = 400;
        response.error(res, res.status, "MTP sampler could not initialize for this request");
        return;
    }

    json request_params = default_params;
    request_params["n_predict"] = cr.max_tokens;
    request_params["max_tokens"] = cr.max_tokens;
    request_params["temperature"] = sparams.temp;
    st.progress.prompt((int)toks.size(), cr.max_tokens, std::move(request_params));
    st.log.start();
    LOG_INF("slot   processing task, n_prompt = %d, n_predict = %d\n", (int)toks.size(), cr.max_tokens);
    auto timings = std::make_shared<json>(json::object());
    // 1:1 upstream timings block (server-context.cpp): prompt/predicted counts, ms, per-token and per-second rates.
    // 中文：对齐上游的 timings 统计块——prompt/predicted 的计数、耗时、每 token 与每秒速率，供 /v1 响应回传
    auto make_emit_gen_wall = [timings, &st, n_prompt = (int)toks.size()](std::chrono::steady_clock::time_point t_turn0,
                                                                          std::chrono::steady_clock::time_point t_pf1,
                                                                          double prefill_ms, int n_cache_hit) {
        const int cache_n = std::clamp(n_cache_hit, 0, n_prompt);
        const int prompt_n = n_prompt - cache_n;
        st.progress.prefilled(cache_n);
        st.log.start_generation();
        return [timings, &st, t_turn0, t_pf1, prefill_ms, n_prompt, prompt_n, cache_n](int n_gen, bool verbose = true) {
            st.progress.generated(n_gen);
            st.log.generated(n_gen);
            const auto now = std::chrono::steady_clock::now();
            const double gen_ms = std::chrono::duration<double, std::milli>(now - t_pf1).count();
            const double prompt_per_second = prefill_ms > 0.0 ? 1000.0 * (double)prompt_n / prefill_ms : 0.0;
            const double predicted_per_second = gen_ms > 0.0 ? 1000.0 * (double)n_gen / gen_ms : 0.0;
            *timings = {{"prompt_n", prompt_n},
                        {"prompt_ms", prefill_ms},
                        {"prompt_per_token_ms", prompt_n > 0 ? prefill_ms / (double)prompt_n : 0.0},
                        {"prompt_per_second", prompt_per_second},
                        {"predicted_n", n_gen},
                        {"predicted_ms", gen_ms},
                        {"predicted_per_token_ms", n_gen > 0 ? gen_ms / (double)n_gen : 0.0},
                        {"predicted_per_second", predicted_per_second},
                        {"cache_n", cache_n}};
            if (!verbose) {
                return;
            }
            LOG_INF("slot   prompt eval time = %10.2f ms / %5d tokens (%8.2f tokens per second), cache = %d\n",
                    prefill_ms, prompt_n, prompt_per_second, cache_n);
            LOG_INF("slot          eval time = %10.2f ms / %5d tokens (%8.2f tokens per second)\n", gen_ms, n_gen,
                    predicted_per_second);
            LOG_INF("slot         total time = %10.2f ms / %5d tokens\n", prefill_ms + gen_ms, prompt_n + n_gen);
            const double wall_ms = std::chrono::duration<double, std::milli>(now - t_turn0).count();
            const double tps = gen_ms > 0.0 ? 1000.0 * (double)n_gen / gen_ms : 0.0;
            kvmem_diag("KVMEM_GEN_WALL n=%d ms=%.2f toks=%.2f\n", n_gen, gen_ms, tps);
            kvmem_diag("KVMEM_CHAT_TURN n_prompt=%d n_gen=%d prefill_ms=%.2f gen_ms=%.2f "
                       "wall_ms=%.2f gen_toks=%.2f\n",
                       n_prompt, n_gen, prefill_ms, gen_ms, wall_ms, tps);
        };
    };

    int n_cache_hit = 0;
    auto emit_json = [&](const std::string &content, int n_gen, bool hit_limit) {
        response.result(res, content, n_gen, hit_limit, n_cache_hit, *timings);
    };

    if (cr.stream) {
        const int max_tokens = cr.max_tokens;
        const bool spec_stream = use_spec;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [slot, &st, &req, toks, response, max_tokens, sparams,
                                                               parse_tools, formatted, stops, spec_stream, ctx, vocab,
                                                               make_emit_gen_wall,
                                                               timings](size_t, httplib::DataSink &sink) mutable {
            struct stream_guard {
                ServerState &st;
                std::shared_ptr<kvmem_server_slot_guard> slot;
                httplib::DataSink &sink;
                ~stream_guard() {
                    multimodal_finish_request(st);
                    slot->unlock();
                    sink.done();
                }
            } guard{st, slot, sink};
            StreamIo io;
            io.sink = &sink;
            io.req = &req;
            auto send = [&](const std::vector<std::string> &frames) -> bool {
                for (const auto &frame : frames) {
                    if (!sink.write(frame.data(), frame.size())) {
                        io.aborted = true;
                        return false;
                    }
                }
                return true;
            };
            auto update = [&](const std::string &content, bool partial) {
                try {
                    return send(response.update(content, partial, *timings));
                } catch (const std::exception &e) {
                    send(response.error_event(e.what()));
                    send(response.error_end());
                    io.aborted = true;
                    return false;
                }
            };
            try {
                if (!send(response.start())) return true;
                const auto t_turn0 = std::chrono::steady_clock::now();
                int n_cache_hit = 0;
                if (!run_prefill_retrieval(st, toks, &io, &n_cache_hit)) {
                    if (!io.aborted) {
                        send(response.error_event(st.mm_error.empty() ? "prefill/retrieval failed" : st.mm_error));
                        send(response.error_end());
                    }
                    return true;
                }
                const auto t_pf1 = std::chrono::steady_clock::now();
                const double prefill_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_turn0).count();
                kvmem_diag("KVMEM_CHAT_PREFILL ms=%.2f n_prompt=%d\n", prefill_ms, (int)toks.size());
                llama_kvmem_end_prefill_capture();
                st.mm_live_checkpoint.reset();
                auto emit_gen_wall = make_emit_gen_wall(t_turn0, t_pf1, prefill_ms, n_cache_hit);

                std::vector<llama_token> gen;
                std::string content;
                bool aborted = false;
                if (spec_stream) {
                    const auto gst = kvmem_spec_generate(
                        st.ctx, st.model, st.spec, toks, max_tokens, sparams,
                        [&](llama_token id, const std::string &piece, bool) {
                            gen.push_back(id);
                            content += piece;
                            response.observe_spec(content, stops);
                            emit_gen_wall((int)gen.size(), false);
                            update(content, true);
                        },
                        [&]() { return io.aborted || response.stop_requested() || !stream_heartbeat(&io); },
                        st.active_prompt->model_pos(toks.size()) - (llama_pos)toks.size());
                    aborted = io.aborted || gst.failed;
                    st.mm_live_row = gst.n_past;
                    if (gst.failed) send(response.error_event("speculative decode failed"));
                } else {
                    common_sampler_ptr smpl;
                    try {
                        common_params_sampling sp = sparams;
                        smpl.reset(common_sampler_init(st.model, sp));
                    } catch (const std::exception &e) {
                        fprintf(stderr, "sampler init failed: %s\n", e.what());
                        send(response.error_event(std::string("sampler init failed: ") + e.what()));
                        send(response.error_end());
                        return true;
                    }
                    if (!smpl) {
                        send(response.error_event("sampler init failed"));
                        send(response.error_end());
                        return true;
                    }
                    bool stopped = false;
                    bool hit_stop = false;
                    while ((int)gen.size() < max_tokens && !stopped) {
                        if (!stream_heartbeat(&io)) {
                            aborted = true;
                            break;
                        }
                        llama_token id = common_sampler_sample(smpl.get(), ctx, -1);
                        common_sampler_accept(smpl.get(), id, true);
                        if (llama_vocab_is_eog(vocab, id)) {
                            stopped = true;
                            break;
                        }
                        std::string piece = token_piece(vocab, id);
                        if (multimodal_decode_generated(st, id, (int)toks.size() + (int)gen.size()) != 0) {
                            fprintf(stderr, "llama_decode(gen) failed\n");
                            aborted = true;
                            send(response.error_event("decode failed"));
                            break;
                        }
                        content += piece;
                        gen.push_back(id);
                        hit_stop = response.trim_stop(content, stops);
                        emit_gen_wall((int)gen.size(), false);
                        if (!update(content, !hit_stop)) {
                            aborted = true;
                            break;
                        }
                        if (hit_stop) {
                            break;
                        }
                    }
                    smpl.reset();
                    if (aborted) {
                        return true;
                    }
                    const bool hit_limit = !stopped && !hit_stop && (int)gen.size() >= max_tokens;
                    emit_gen_wall((int)gen.size(), false);
                    if (!update(content, false)) return true;
                    response.trace(hit_limit);
                    llama_kvmem_decode_mean_flush();
                    emit_gen_wall((int)gen.size());
                    commit_cached(st, toks, gen);
                    send(response.finish(hit_limit, (int)gen.size(), n_cache_hit, *timings));
                    return true;
                }
                if (aborted) {
                    return true;
                }
                const bool hit_limit = (int)gen.size() >= max_tokens;
                emit_gen_wall((int)gen.size(), false);
                if (!update(content, false)) return true;
                response.trace(hit_limit);
                emit_gen_wall((int)gen.size());
                commit_cached(st, toks, gen);
                send(response.finish(hit_limit, (int)gen.size(), n_cache_hit, *timings));
                return true;
            } catch (const std::exception &e) {
                if (!io.aborted) {
                    send(response.error_event(e.what()));
                    send(response.error_end());
                }
                return true;
            }
        });
        return;
    }

    struct request_guard {
        ServerState &st;
        ~request_guard() { multimodal_finish_request(st); }
    } guard{st};
    StreamIo io;
    io.req = &req;
    const auto t_turn0 = std::chrono::steady_clock::now();
    if (!run_prefill_retrieval(st, toks, &io, &n_cache_hit)) {
        if (io.aborted) {
            kvmem_diag("KVMEM_TRACE stream_abort phase=prefill n_prompt=%d\n", (int)toks.size());
            return;
        }
        res.status = (st.vision || st.query_policy_user) ? st.mm_error_status : 500;
        response.error(res, res.status, st.mm_error.empty() ? "prefill/retrieval failed" : st.mm_error);
        return;
    }
    const auto t_pf1 = std::chrono::steady_clock::now();
    const double prefill_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_turn0).count();
    kvmem_diag("KVMEM_CHAT_PREFILL ms=%.2f n_prompt=%d\n", prefill_ms, (int)toks.size());
    llama_kvmem_end_prefill_capture();
    st.mm_live_checkpoint.reset();
    auto emit_gen_wall = make_emit_gen_wall(t_turn0, t_pf1, prefill_ms, n_cache_hit);

    if (use_spec) {
        std::string content;
        std::vector<llama_token> gen;
        const kvmem_spec_gen_stats gst = kvmem_spec_generate(
            ctx, st.model, st.spec, toks, cr.max_tokens, sparams,
            [&](llama_token id, const std::string &piece, bool) {
                gen.push_back(id);
                content += piece;
                response.observe_spec(content, stops);
                st.progress.generated((int)gen.size());
                st.log.generated((int)gen.size());
            },
            [&]() { return response.stop_requested() || !stream_heartbeat(&io); },
            st.active_prompt->model_pos(toks.size()) - (llama_pos)toks.size());
        if (gst.failed) {
            res.status = 500;
            response.error(res, res.status, "speculative decode failed");
            return;
        }
        st.mm_live_row = gst.n_past;
        if (io.aborted) return;
        emit_gen_wall((int)gen.size());
        commit_cached(st, toks, gen);
        emit_json(content, (int)gen.size(), (int)gen.size() >= cr.max_tokens);
        return;
    }

    common_sampler *smpl = nullptr;
    try {
        common_params_sampling sp = sparams;
        smpl = common_sampler_init(st.model, sp);
    } catch (const std::exception &e) {
        res.status = 500;
        response.error(res, res.status, std::string("sampler init failed: ") + e.what());
        return;
    }
    if (!smpl) {
        res.status = 500;
        response.error(res, res.status, "sampler init failed");
        return;
    }

    int next_row = (int)toks.size();
    auto gen_one = [ctx, smpl, vocab, &st, &next_row](std::string &piece, bool &stopped, llama_token &id_out) -> bool {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(vocab, id)) {
            stopped = true;
            return true;
        }
        id_out = id;
        piece = token_piece(vocab, id);
        if (multimodal_decode_generated(st, id, next_row++) != 0) {
            fprintf(stderr, "llama_decode(gen) failed\n");
            return false;
        }
        return true;
    };

    std::string content;
    std::vector<llama_token> gen;
    bool stopped = false;
    while ((int)gen.size() < cr.max_tokens && !stopped) {
        if (!stream_heartbeat(&io)) {
            common_sampler_free(smpl);
            return;
        }
        std::string piece;
        llama_token id = 0;
        if (!gen_one(piece, stopped, id)) {
            common_sampler_free(smpl);
            res.status = 500;
            response.error(res, res.status, "decode failed");
            return;
        }
        if (stopped) {
            break;
        }
        content += piece;
        gen.push_back(id);
        st.progress.generated((int)gen.size());
        st.log.generated((int)gen.size());
        if (response.trim_stop(content, stops)) {
            break;
        }
    }
    llama_kvmem_decode_mean_flush();
    emit_gen_wall((int)gen.size());
    commit_cached(st, toks, gen);
    common_sampler_free(smpl);
    emit_json(content, (int)gen.size(), !stopped && (int)gen.size() >= cr.max_tokens);
}
