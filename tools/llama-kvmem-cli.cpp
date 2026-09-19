#include "llama.h"
#include "llama-kvmem-hooks.h"
#include "kvmem-spec.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf [options] [prompt]\n"
            "\n"
            "  -m, --model PATH           GGUF path\n"
            "  -f, --file PATH            read prompt from file (instead of trailing args)\n"
            "  -n, --n-predict N          tokens to generate (default 32)\n"
            "  -c, --ctx-size N           context size (default prompt + n_predict)\n"
            "  -b, --batch-size N         logical batch (default 512)\n"
            "  -ub, --ubatch-size N       physical ubatch (default 512)\n"
            "  -ngl, --n-gpu-layers N     GPU layers (default 99)\n"
            "  --temp T                   temperature; 0 = greedy (default 0)\n"
            "  --tokens-only              print generated token ids, one per line\n"
            "  --no-prompt                do not echo the prompt (generation only)\n"
            "  --kvmem                    enable KVMem slot-pool memory\n"
            "  --kvmem-block-tokens N     block size (default 32)\n"
            "  --kvmem-budget N           GPU working-set tokens; 0 = n_ctx (identity)\n"
            "  --kvmem-gen-reserve N      extra GPU tokens for decode (default 256)\n"
            "  --kvmem-sink-tokens N      always-kept prefix; 0 = one block\n"
            "  --kvmem-recent-tokens N    always-kept suffix blocks (default 0)\n"
            "  --kvmem-method NAME        recency | retrieval (default retrieval)\n"
            "  --kvmem-query-last N       last N prompt tokens are the retrieval query (default 64)\n"
            "  --kvmem-force-substr S     force-select the block containing substring S\n"
            "  --kvmem-gpu-ratio R        cap slot pool at this fraction of GPU VRAM (default 0.50)\n"
            "  --kvmem-gpu-high R         prefill offload high watermark (default 0.95)\n"
            "  --kvmem-gpu-low R          prefill offload low watermark (default 0.85)\n"
            "  --kvmem-cpu-gb GB          CPU spill arena in GiB (0 = off)\n"
            "  --kvmem-nvme-gb GB         NVMe spill file in GiB (0 = off)\n"
            "  --kvmem-nvme-dir PATH      NVMe spill directory (default /tmp/kvmem_nvme)\n"
            "  --kvmem-harvest-v          prefill D2H V with raw-K (default off; RAM until NVMe flush)\n"
            "  --kvmem-raw-k-nvme         store raw-K and V on NVMe (needs --kvmem-nvme-gb)\n"
            "  --kvmem-dump-kv            after prefill, compare raw-rebuild KV vs GPU KV\n"
            "  --kv-dtype NAME            GPU KV cache type for K and V: f16 | f32 | q8_0 | q5_0 | q4_0 (default q8_0)\n"
            "  -ctk, --cache-type-k TYPE  GPU K cache type (llama.cpp name; default q8_0)\n"
            "  -ctv, --cache-type-v TYPE  GPU V cache type (must match K when quantized)\n"
            "  --spec-type TYPE           none | draft-mtp (default none)\n"
            "  --spec-kv-dtype TYPE       MTP K/V type (default: inherit target K/V types)\n"
            "  --spec-draft-n-max N       MTP draft tokens (default 2)\n"
            "  --spec-draft-p-min P       min draft probability (default 0)\n"
            "  --spec-draft-model PATH    optional sidecar MTP GGUF\n",
            argv0);
}

static bool eq(const char * a, const char * b) {
    return std::strcmp(a, b) == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "Hello my name is";
    std::string prompt_file;
    int n_predict = 32;
    int n_ctx = 0;
    int n_batch = 512;
    int n_ubatch = 512;
    int ngl = 99;
    float temp = 0.0f;
    bool tokens_only = false;
    bool no_prompt = false;

    llama_kvmem_params kparams = {};
    kparams.block_tokens = 32;
    kparams.gen_reserve = 256;
    kparams.method = 1;  // retrieval
    kparams.query_begin = -1;
    kparams.query_end = -1;
    kparams.force_pos = -1;
    int query_last = 64;
    std::string force_substr;
    std::string nvme_dir;
    bool dump_kv = false;
    ggml_type cache_type_k = GGML_TYPE_Q8_0;
    ggml_type cache_type_v = GGML_TYPE_Q8_0;
    ggml_type spec_cache_type = GGML_TYPE_COUNT;
    bool spec_mtp = false;
    int spec_n_max = 2;
    float spec_p_min = 0.0f;
    std::string spec_draft_model;

    int i = 1;
    for (; i < argc; ++i) {
        const char * arg = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                print_usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        if (eq(arg, "-h") || eq(arg, "--help")) {
            print_usage(argv[0]);
            return 0;
        } else if (eq(arg, "-m") || eq(arg, "--model")) {
            model_path = need(arg);
        } else if (eq(arg, "-f") || eq(arg, "--file")) {
            prompt_file = need(arg);
        } else if (eq(arg, "-n") || eq(arg, "--n-predict")) {
            n_predict = std::atoi(need(arg));
        } else if (eq(arg, "-c") || eq(arg, "--ctx-size")) {
            n_ctx = std::atoi(need(arg));
        } else if (eq(arg, "-b") || eq(arg, "--batch-size")) {
            n_batch = std::atoi(need(arg));
        } else if (eq(arg, "-ub") || eq(arg, "--ubatch-size")) {
            n_ubatch = std::atoi(need(arg));
        } else if (eq(arg, "-ngl") || eq(arg, "--n-gpu-layers")) {
            ngl = std::atoi(need(arg));
        } else if (eq(arg, "--temp")) {
            temp = std::atof(need(arg));
        } else if (eq(arg, "--tokens-only")) {
            tokens_only = true;
        } else if (eq(arg, "--no-prompt")) {
            no_prompt = true;
        } else if (eq(arg, "--kvmem")) {
            kparams.enabled = true;
        } else if (eq(arg, "--kvmem-block-tokens")) {
            kparams.block_tokens = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-budget")) {
            kparams.budget = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-gen-reserve")) {
            kparams.gen_reserve = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-sink-tokens")) {
            kparams.sink_tokens = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-recent-tokens")) {
            kparams.recent_tokens = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-method")) {
            const char * m = need(arg);
            if (eq(m, "retrieval") || eq(m, "retrieve")) {
                kparams.method = 1;
            } else {
                kparams.method = 0;
            }
        } else if (eq(arg, "--kvmem-query-last")) {
            query_last = std::atoi(need(arg));
        } else if (eq(arg, "--kvmem-force-substr")) {
            force_substr = need(arg);
        } else if (eq(arg, "--kvmem-dump-kv")) {
            dump_kv = true;
        } else if (eq(arg, "--kv-dtype") || eq(arg, "-ctk") || eq(arg, "--cache-type-k")
                   || eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
            bool ok = false;
            const ggml_type t = kvmem_parse_cache_type(need(arg), &ok);
            if (!ok) {
                fprintf(stderr, "unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)\n");
                return 1;
            }
            if (eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
                cache_type_v = t;
            } else if (eq(arg, "-ctk") || eq(arg, "--cache-type-k")) {
                cache_type_k = t;
            } else {
                cache_type_k = t;
                cache_type_v = t;
            }
        } else if (eq(arg, "--kvmem-gpu-ratio")) {
            kparams.gpu_memory_ratio = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-gpu-high")) {
            kparams.gpu_high_watermark = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-gpu-low")) {
            kparams.gpu_low_watermark = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-cpu-gb")) {
            const double gb = std::atof(need(arg));
            kparams.cpu_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (eq(arg, "--kvmem-nvme-gb")) {
            const double gb = std::atof(need(arg));
            kparams.nvme_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (eq(arg, "--kvmem-nvme-dir")) {
            nvme_dir = need(arg);
        } else if (eq(arg, "--kvmem-harvest-v")) {
            kparams.harvest_v = true;
        } else if (eq(arg, "--kvmem-raw-k-nvme")) {
            kparams.raw_k_nvme = true;
        } else if (eq(arg, "--spec-kv-dtype")) {
            bool ok = false;
            spec_cache_type = kvmem_parse_cache_type(need(arg), &ok);
            if (!ok) {
                fprintf(stderr, "unsupported MTP cache type (want f16|q8_0|q5_0|q4_0|f32)\n");
                return 1;
            }
        } else if (eq(arg, "--spec-type")) {
            const char * t = need(arg);
            if (eq(t, "draft-mtp")) {
                spec_mtp = true;
            } else if (eq(t, "none")) {
                spec_mtp = false;
            } else {
                fprintf(stderr, "unsupported --spec-type %s (P7-0: draft-mtp|none)\n", t);
                return 1;
            }
        } else if (eq(arg, "--spec-draft-n-max")) {
            spec_n_max = std::atoi(need(arg));
        } else if (eq(arg, "--spec-draft-p-min")) {
            spec_p_min = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--spec-draft-model") || eq(arg, "-md")) {
            spec_draft_model = need(arg);
        } else if (arg[0] == '-') {
            fprintf(stderr, "unknown flag: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        } else {
            break;
        }
    }
#if !KVMEM_ENABLE_NVME
    if (kparams.nvme_bytes || kparams.raw_k_nvme) {
        fprintf(stderr, "NVMe offload is disabled in this build (KVMEM_ENABLE_NVME=OFF)\n");
        return 1;
    }
#endif
    // Validate before backend initialization and loading a potentially large model.
    if (!kvmem_cache_types_ok(cache_type_k, cache_type_v)) {
        fprintf(stderr, "incompatible KV cache types: K=%s, V=%s; quantized K/V must match; "
                "set both -ctk and -ctv, or use --kv-dtype TYPE to set both\n",
                ggml_type_name(cache_type_k), ggml_type_name(cache_type_v));
        return 1;
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (!prompt_file.empty()) {
        std::ifstream in(prompt_file);
        if (!in) {
            fprintf(stderr, "failed to read prompt file: %s\n", prompt_file.c_str());
            return 1;
        }
        prompt.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    } else if (i < argc) {
        prompt = argv[i++];
        for (; i < argc; ++i) {
            prompt += " ";
            prompt += argv[i];
        }
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    model_params.load_mtp = spec_mtp;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }
    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }

    if (n_ctx <= 0) {
        n_ctx = n_prompt + n_predict;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    if (n_ubatch <= 0) {
        n_ubatch = n_batch;
    }

    if (kparams.enabled && kparams.budget > 0) {
        const uint32_t bt = kparams.block_tokens ? kparams.block_tokens : 32u;
        const uint32_t sink = kparams.sink_tokens == 0 ? bt : kparams.sink_tokens;
        const uint32_t room = kparams.budget > sink ? kparams.budget - sink : bt;
        const uint32_t cap = std::min(kparams.gen_reserve ? kparams.gen_reserve : 256u, room);
        if (static_cast<uint32_t>(n_batch) > cap) {
            fprintf(stderr, "llama-kvmem-cli: clamping -b %d → %u so each prefill chunk fits the slot pool\n",
                    n_batch, cap);
            n_batch = static_cast<int>(cap);
        }
        if (n_ubatch > n_batch) {
            n_ubatch = n_batch;
        }
    }

    if (kparams.enabled) {
        if (!nvme_dir.empty()) {
            kparams.nvme_dir = nvme_dir.c_str();
        }
        if (dump_kv) {
#if defined(_WIN32)
            // _putenv_s always overwrites, which is setenv(..., 1) semantics.
            _putenv_s("KVMEM_DUMP_CAPTURE", "1");
#else
            setenv("KVMEM_DUMP_CAPTURE", "1", 1);
#endif
        }
        if (query_last > 0 && n_prompt > 0) {
            const int last = std::min(query_last, n_prompt);
            kparams.query_begin = n_prompt - last;
            kparams.query_end = n_prompt;
        }
        if (!force_substr.empty()) {
            std::string acc;
            for (int ti = 0; ti < n_prompt; ++ti) {
                char buf[256];
                const int n = llama_token_to_piece(vocab, prompt_tokens[ti], buf, sizeof(buf), 0, true);
                if (n > 0) {
                    acc.append(buf, static_cast<size_t>(n));
                }
                if (acc.find(force_substr) != std::string::npos) {
                    kparams.force_pos = ti;
                    break;
                }
            }
            fprintf(stderr, "KVMEM_TRACE n_prompt=%d force_substr='%s' force_pos=%d block~=%d\n",
                    n_prompt, force_substr.c_str(), kparams.force_pos,
                    kparams.force_pos >= 0 && kparams.block_tokens
                        ? kparams.force_pos / static_cast<int>(kparams.block_tokens)
                        : -1);
        } else {
            fprintf(stderr, "KVMEM_TRACE n_prompt=%d\n", n_prompt);
        }
        llama_kvmem_set_params(&kparams);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    ctx_params.n_batch = static_cast<uint32_t>(n_batch);
    ctx_params.n_ubatch = static_cast<uint32_t>(n_ubatch);
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = false;
    ctx_params.type_k = cache_type_k;
    ctx_params.type_v = cache_type_v;
    if (spec_mtp) {
        const uint32_t n_out = (uint32_t) (1 + std::max(0, spec_n_max));
        ctx_params.n_outputs_max = n_out;
        ctx_params.n_outputs_max_per_seq = n_out;
        // GPU GDN snapshot planes for MTP verify reject (qw3-style). Host
        // PARTIAL_ONLY is only used if the arch clamps this to 0.
        ctx_params.n_rs_seq = (uint32_t) std::max(0, spec_n_max);
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create llama_context\n");
        return 1;
    }

    kvmem_spec_session spec_sess;
    if (spec_mtp) {
        kvmem_spec_opts sopts;
        sopts.n_max = spec_n_max;
        sopts.p_min = spec_p_min;
        sopts.n_gpu_layers = ngl;
        sopts.n_ctx = n_ctx;
        sopts.n_batch = n_batch;
        sopts.n_ubatch = n_ubatch;
        sopts.kvmem_enabled = kparams.enabled;
        sopts.draft_model = spec_draft_model;
        sopts.type_k = cache_type_k;
        sopts.type_v = cache_type_v;
        sopts.draft_type = spec_cache_type;
        if (!kvmem_spec_start(spec_sess, model, ctx, sopts)) {
            return 1;
        }
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
    }

    if (!tokens_only && !no_prompt) {
        for (auto id : prompt_tokens) {
            char buf[256];
            const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "token_to_piece failed\n");
                return 1;
            }
            fwrite(buf, 1, static_cast<size_t>(n), stdout);
        }
        fflush(stdout);
    }

    auto decode_span = [&](int pos0, int pos1, const char * what) -> int {
        if (spec_sess.ok) {
            return kvmem_spec_decode_span(ctx, spec_sess.spec, prompt_tokens.data(),
                                          pos0, pos1, n_batch, what);
        }
        int n_pos = pos0;
        while (n_pos < pos1) {
            const int n = std::min(n_batch, pos1 - n_pos);
            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + n_pos, n);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n",
                        what, rc, n_pos, n);
                return rc;
            }
            n_pos += n;
        }
        return 0;
    };

    // Speculative-simple leaves the last prompt token as id_last (not in KV).
    const int eval_end = spec_sess.ok ? n_prompt - 1 : n_prompt;
    if (spec_sess.ok && n_prompt < 1) {
        fprintf(stderr, "speculative decode needs a non-empty prompt\n");
        return 1;
    }

    const bool do_retr = kparams.enabled && kparams.method == 1 && kparams.query_begin > 0;
    const bool recr_ckpt = do_retr && llama_kvmem_has_recurrent();
    const int prefix_end = recr_ckpt ? kparams.query_begin : eval_end;

    if (decode_span(0, prefix_end, "prefill") != 0) {
        return 1;
    }

    std::vector<uint8_t> gdn_ckpt;
    if (recr_ckpt) {
        llama_synchronize(ctx);
        llama_kvmem_trace_cells(ctx, "before_gdn_ckpt");
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, fl);
        if (sz == 0) {
            fprintf(stderr, "llama_state_seq_get_size_ext(PARTIAL_ONLY) returned 0\n");
            return 1;
        }
        gdn_ckpt.resize(sz);
        const size_t ncopy = llama_state_seq_get_data_ext(ctx, gdn_ckpt.data(), sz, 0, fl);
        if (ncopy != sz) {
            fprintf(stderr, "GDN checkpoint copy failed (%zu != %zu)\n", ncopy, sz);
            return 1;
        }
        fprintf(stderr, "KVMEM_TRACE gdn_ckpt pos_end=%d bytes=%zu query_begin=%d\n",
                prefix_end, sz, kparams.query_begin);
        if (decode_span(prefix_end, eval_end, "prefill-query") != 0) {
            return 1;
        }
    }
    llama_synchronize(ctx);

    const llama_perf_context_data perf_prefill = llama_perf_context(ctx);
    double retrieval_ms = 0.0;
    double replay_wall_ms = 0.0;
    int replay_n = 0;
    double replay_eval_ms = 0.0;

    if (kparams.enabled && dump_kv) {
        fprintf(stderr, "KVMEM_KV --- after prefill ---\n");
        llama_kvmem_dump_kv_compare(ctx, 0);
        llama_kvmem_dump_kv_compare(ctx, -1);
        fprintf(stderr, "KVMEM_KV --- overwrite needle block from raw, compare to snapshot ---\n");
        llama_kvmem_dump_kv_writeback(ctx, -1);
    }

    if (do_retr) {
        const auto t0 = std::chrono::steady_clock::now();
        llama_kvmem_apply_retrieval(ctx);
        retrieval_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        if (recr_ckpt && !gdn_ckpt.empty()) {
            const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            const size_t nset = llama_state_seq_set_data_ext(
                    ctx, gdn_ckpt.data(), gdn_ckpt.size(), 0, fl);
            if (nset != gdn_ckpt.size()) {
                fprintf(stderr, "GDN restore failed (%zu != %zu)\n", nset, gdn_ckpt.size());
                return 1;
            }
            fprintf(stderr, "KVMEM_TRACE gdn_restore bytes=%zu\n", gdn_ckpt.size());
            llama_kvmem_trace_cells(ctx, "after_gdn_restore");
        }
        llama_memory_t mem = llama_get_memory(ctx);
        llama_kvmem_set_replay(true);
        if (mem) {
            fprintf(stderr, "KVMEM_TRACE before_seq_rm seq_pos=[%d,%d] query=[%d,%d)\n",
                    llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                    kparams.query_begin, kparams.query_end);
            llama_memory_seq_rm(mem, 0, kparams.query_begin, kparams.query_end);
            fprintf(stderr, "KVMEM_TRACE after_seq_rm seq_pos=[%d,%d] auto_pos0=%d\n",
                    llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                    llama_memory_seq_pos_max(mem, 0) + 1);
            llama_kvmem_trace_cells(ctx, "after_seq_rm");
        }
        if (spec_sess.ctx_dft) {
            llama_memory_t md = llama_get_memory(spec_sess.ctx_dft);
            if (md) {
                llama_memory_seq_rm(md, 0, kparams.query_begin, kparams.query_end);
                fprintf(stderr, "KVMEM_TRACE mtp_after_seq_rm seq_pos=[%d,%d]\n",
                        llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
            }
        }
        const int q0 = kparams.query_begin;
        const int qn = eval_end - q0;
        if (qn > 0) {
            llama_synchronize(ctx);
            const llama_perf_context_data before_replay = llama_perf_context(ctx);
            const auto r0 = std::chrono::steady_clock::now();
            const int rc = decode_span(q0, q0 + qn, "query replay");
            llama_synchronize(ctx);
            replay_wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - r0).count();
            const llama_perf_context_data after_replay = llama_perf_context(ctx);
            replay_n = after_replay.n_p_eval - before_replay.n_p_eval;
            replay_eval_ms = after_replay.t_p_eval_ms - before_replay.t_p_eval_ms;
            if (rc != 0) {
                fprintf(stderr, "llama_decode(query replay) failed rc=%d\n", rc);
                return 1;
            }
        }
        llama_kvmem_set_replay(false);
        fprintf(stderr, "KVMEM_TRACE query_replay begin=%d n=%d recr_ckpt=%d\n",
                q0, qn, (int) recr_ckpt);
        llama_kvmem_trace_cells(ctx, "after_query_replay");
        if (spec_sess.ctx_dft) {
            llama_memory_t md = llama_get_memory(spec_sess.ctx_dft);
            if (md) {
                fprintf(stderr, "KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                        llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
            }
        }
    }

    std::vector<llama_token> gen;
    gen.reserve(static_cast<size_t>(n_predict));
    llama_kvmem_end_prefill_capture();
    llama_synchronize(ctx);
    const auto t_gen0 = std::chrono::steady_clock::now();
    if (spec_sess.ok) {
        const kvmem_spec_gen_stats gst = kvmem_spec_generate(
                ctx, model, spec_sess, prompt_tokens, n_predict, temp,
                [&](llama_token id, const std::string & piece, bool /*from_draft*/) {
                    gen.push_back(id);
                    if (tokens_only) {
                        printf("%d\n", id);
                    } else {
                        fwrite(piece.data(), 1, piece.size(), stdout);
                        fflush(stdout);
                    }
                });
        if (gst.failed) {
            kvmem_spec_stop(spec_sess);
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    } else {
        for (int n_gen = 0; n_gen < n_predict; ++n_gen) {
            const llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }
            gen.push_back(new_token);
            if (tokens_only) {
                printf("%d\n", new_token);
            } else {
                char buf[256];
                const int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    fprintf(stderr, "token_to_piece failed\n");
                    return 1;
                }
                fwrite(buf, 1, static_cast<size_t>(n), stdout);
                fflush(stdout);
            }
            llama_batch batch = llama_batch_get_one(&gen.back(), 1);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                fprintf(stderr, "llama_decode(gen) failed rc=%d\n", rc);
                return 1;
            }
        }
        llama_kvmem_decode_mean_flush();
    }
    if (!tokens_only) {
        printf("\n");
    }

    llama_synchronize(ctx);
    const double gen_wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_gen0).count();
    llama_perf_context_print(ctx);
    {
        const llama_perf_context_data p = llama_perf_context(ctx);
        const double prompt_tps =
            p.t_p_eval_ms > 0.0 ? 1000.0 * static_cast<double>(p.n_p_eval) / p.t_p_eval_ms : 0.0;
        const double gen_tps =
            p.t_eval_ms > 0.0 ? 1000.0 * static_cast<double>(p.n_eval) / p.t_eval_ms : 0.0;
        fprintf(stderr,
                "KVMEM_PERF load_ms=%.2f prompt_n=%d prompt_ms=%.2f prompt_toks=%.2f gen_n=%d gen_ms=%.2f gen_toks=%.2f\n",
                p.t_load_ms, p.n_p_eval, p.t_p_eval_ms, prompt_tps, p.n_eval, p.t_eval_ms, gen_tps);
        const double prefill_tps = perf_prefill.t_p_eval_ms > 0.0
            ? 1000.0 * static_cast<double>(perf_prefill.n_p_eval) / perf_prefill.t_p_eval_ms
            : 0.0;
        fprintf(stderr,
                "KVMEM_STAGE prefill_n=%d prefill_ms=%.2f prefill_toks=%.2f "
                "retrieval_ms=%.2f replay_n=%d replay_eval_ms=%.2f replay_wall_ms=%.2f\n",
                perf_prefill.n_p_eval, perf_prefill.t_p_eval_ms, prefill_tps,
                retrieval_ms, replay_n, replay_eval_ms, replay_wall_ms);
        const int n_out = (int) gen.size();
        const double gen_wall_tps =
            gen_wall_ms > 0.0 ? 1000.0 * static_cast<double>(n_out) / gen_wall_ms : 0.0;
        fprintf(stderr, "KVMEM_GEN_WALL n=%d ms=%.2f toks=%.2f\n",
                n_out, gen_wall_ms, gen_wall_tps);
    }

    kvmem_spec_stop(spec_sess);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
