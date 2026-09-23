#include "llama-kvmem-diag.h"
#include "llama.h"
#include "llama-kvmem-hooks.h"
#include "kvmem-spec.h"
#include "kvmem-chat-sampling.h"
#include "kvmem-chat-template.h"
#include "kvmem-chat-id.h"
#include "kvmem-webui.h"
#include "kvmem-server-options.h"
#include "kvmem-server-auth.h"
#include "kvmem-server-progress.h"
#include "kvmem-server-devices.h"
#include "kvmem-server-env.h"
#include "kvmem-vision.h"
#include "kvmem-prefill-policy.h"

#include "chat.h"
#include "common.h"
#include "log.h"
#include "kvmem-server-log.h"
#include "mtmd-helper.h"
#include "arg.h"
#include "json.h"
#include "build-info.h"
#include "sampling.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static bool eq(const char * a, const char * b) {
    return std::strcmp(a, b) == 0;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf [options]\n"
            "\n"
            "  Independent single-slot OpenAI-compatible server. Does not patch llama-server.\n"
            "\n"
            "  -m, --model PATH           GGUF path\n"
            "  --mmproj PATH              vision projector GGUF\n"
            "  --mmproj-offload           place vision encoder on GPU (default)\n"
            "  --no-mmproj-offload        place vision encoder on CPU\n"
            "  --image-min-tokens N       native minimum image token count\n"
            "  --image-max-tokens N       native maximum image token count\n"
            "  -lv, --verbosity N         log level: 0 silent, 1 error, 2 warn, 3 info (default), 4 trace, 5 debug\n"
            "  --log-verbosity N          alias of --verbosity\n"
            "  --kvmem-trace              raw KVMEM_* diagnostics (or KVMEM_TRACE=1)\n"
            "  --no-kvmem-trace           disable diagnostics, overriding the environment\n"
            "  --host HOST                bind address (default 127.0.0.1)\n"
            "  --port N                   port (default 8080)\n"
            "  --ui-dir PATH              serve static chat UI from PATH\n"
            "  --no-ui                    disable bundled chat UI\n"
            "  -c, --ctx-size N           context size (default 2048)\n"
            "  -n, --n-predict N          default max_tokens (-1 = no extra cap, default -1)\n"
            "  -b, --batch-size N         logical batch (default 512)\n"
            "  -ngl, --n-gpu-layers N     GPU layers (default 99)\n"
            "  --gpu-layers N             alias of --n-gpu-layers; all supported, auto unsupported\n"
            "  -t, --threads N            CPU generation threads; <=0 = hardware concurrency\n"
            "  -tb, --threads-batch N     CPU batch threads (defaults to --threads)\n"
            "  -ub, --ubatch-size N       physical batch size (defaults to --batch-size)\n"
            "  -fa, --flash-attn MODE     on | off | auto\n"
            "  -a, --alias NAME           model name exposed by the API\n"
            "  --api-key KEY[,KEY...]     allowed API keys\n"
            "  --api-key-file PATH        one key per line; blank/# lines ignored\n"
            "  -np, --parallel N          only 1 is currently supported\n"
            "  -lm, --load-mode MODE      auto | none | mmap | mlock | mmap+mlock | dio\n"
            "  --mmap / --no-mmap         legacy aliases for load-mode mmap / none\n"
            "  --mlock                   legacy alias for load-mode mlock\n"
            "  --timeout, -to N          HTTP read/write timeout seconds (default 1800)\n"
            "  --threads-http N          HTTP worker threads; <=0 = automatic\n"
            "  --device, -dev NAME       one offload device, e.g. CUDA0; none = CPU\n"
            "  --list-devices            list available offload devices and exit\n"
            "  --main-gpu, -mg N         main device index (default 0)\n"
            "  --split-mode, -sm MODE    none | layer; multi-GPU modes are not supported\n"
            "  --tensor-split, -ts N     one device proportion; multiple entries are rejected\n"
            "  Aliases: --usage, --predict, -s, -mm, --no-webui, --path\n"
            "  Environment: supported LLAMA_ARG_* settings apply before CLI; API keys append.\n"
            "  --ui / --webui            enable UI (overrides LLAMA_ARG_UI=0)\n"
            "  Sampling defaults: Qwen3.8-27B Thinking / non-Thinking, selected per request.\n"
            "  --temp, --temperature T    temperature [0,2] (1.0 / 0.7); 0 = greedy\n"
            "  --top-p P                  nucleus threshold [0,1] (0.95 / 0.80)\n"
            "  --top-k K                  integer >= 0; 0 disables (20)\n"
            "  --min-p P                  minimum relative probability [0,1] (0)\n"
            "  --presence-penalty P       presence penalty [-2,2] (0 / 1.5)\n"
            "  --frequency-penalty P      frequency penalty [-2,2] (0)\n"
            "  --repeat-penalty P         repetition penalty > 0 (1); --repetition-penalty alias\n"
            "  --seed N                   uint32 seed (default random)\n"
            "                            request fields override these process defaults\n"
            "  --kvmem / --no-kvmem       enable KVMem (default on)\n"
            "  --kvmem-budget N           GPU working-set tokens; 0 = n_ctx\n"
            "  --kvmem-block-tokens N     block size (default 128)\n"
            "  --kvmem-sink-tokens N      always-kept prefix; default 0 = one block; rounds down, minimum one block\n"
            "  --kvmem-gen-reserve N      decode slack (default 256)\n"
            "  --kvmem-recent-tokens N    always-kept newest suffix in select budget (default 0)\n"
            "  --kvmem-method NAME        recency | retrieval (default retrieval)\n"
            "  --kvmem-query-last N       fallback query-last if last-user span missing (default 64)\n"
            "  --kvmem-query-max-tokens N cap last-user retrieval query to this many tokens\n"
            "                            from the end of the span (default 512; qw3-style)\n"
            "  --kvmem-query-replay MODE  legacy or auto (default auto)\n"
            "  --kvmem-query-policy MODE  legacy or user (default user)\n"
            "  --kvmem-mtp-state MODE     snapshots, auto or replay (default replay with MTP)\n"
            "  --kvmem-gpu-ratio R        cap slot pool at this fraction of GPU VRAM (default 0.50)\n"
            "  --kvmem-cpu-gb GB          CPU spill arena in GiB (0 = off)\n"
            "  --kvmem-nvme-gb GB         NVMe file in GiB (0 = off)\n"
            "  --kvmem-nvme-dir PATH      NVMe directory (default /tmp/kvmem_nvme)\n"
            "  --kvmem-harvest-v          prefill D2H V with raw-K (default off; RAM until NVMe flush)\n"
            "  --kvmem-raw-k-nvme         store raw-K and V on NVMe (needs --kvmem-nvme-gb)\n"
            "  --kv-dtype NAME            GPU KV cache type for K and V: f16 | f32 | q8_0 | q5_0 | q4_0 (default q8_0)\n"
            "  -ctk, --cache-type-k TYPE  GPU K cache type (llama.cpp name; default q8_0)\n"
            "  -ctv, --cache-type-v TYPE  GPU V cache type (quantized: independently q8_0 | q5_0 | q4_0)\n"
            "  --spec-type TYPE           none | draft-mtp (default none)\n"
            "  --spec-kv-dtype TYPE       MTP K/V type (default f16)\n"
            "  --spec-draft-n-max N       MTP draft tokens (default 3)\n"
            "  --spec-draft-p-min P       min draft probability (default 0)\n"
            "  --jinja                    native Jinja rendering (always enabled)\n"
            "  --chat-template TEMPLATE   override model chat template (Jinja text)\n"
            "  --chat-template-file PATH  load a Jinja template file\n"
            "  --chat-template-kwargs JSON  default template arguments\n"
            "  --reasoning-effort LEVEL   template effort; default uses template default, none disables thinking\n"
            "  --enable-thinking          Qwen thinking on (default off; request can override)\n"
            "  --no-think                 force thinking off\n"
            "  --reasoning-budget N       thinking token budget: -1 unlimited, 0 end immediately,\n"
            "                            N>0 force </think> after N think tokens (default -1)\n"
            "  --reasoning-budget-message MSG  injected before forced </think> (default none)\n"
            "  --auto-continue-thinking   when thinking hits the output limit, keep the chain in\n"
            "                            context and start another round (default off)\n"
            "  --act-max-rounds N         safety cap on continuation rounds (default 8)\n"
            "  --act-round-tokens N       per-round token budget; 0 = request max_tokens (default 0)\n"
            "  --act-total-tokens N       total token budget across rounds; 0 = unlimited\n"
            "  --act-continue-cue STR     text injected between rounds (default \"\\n\")\n"
            "  --act-prefill-mode MODE    how a continuation round re-ingests the chain:\n"
            "                            auto | legacy | restart (default auto)\n"
            "  --act-think-end STR        thinking end tag (default </think>)\n"
            "  --webui                    serve bundled chat UI (default on)\n",
            argv0);
}

static std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> out;
    if (n <= 0) {
        return out;
    }
    out.resize((size_t) n);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), n, add_special, true);
    return out;
}

static std::string token_piece(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n <= 0) {
        return {};
    }
    return std::string(buf, (size_t) n);
}

static int force_pos_from_substr(const llama_vocab * vocab, const std::vector<llama_token> & toks,
                                 const std::string & needle) {
    if (needle.empty()) {
        return -1;
    }
    std::string acc;
    for (int i = 0; i < (int) toks.size(); ++i) {
        if (toks[(size_t) i] == LLAMA_TOKEN_NULL) continue;
        acc += token_piece(vocab, toks[(size_t) i]);
        if (acc.find(needle) != std::string::npos) {
            return i;
        }
    }
    return -1;
}

struct MultimodalCheckpointAccounting {
    size_t live_bytes = 0;
    size_t peak_bytes = 0;
};

struct MultimodalCheckpointData {
    MultimodalCheckpointData() = default;
    MultimodalCheckpointData(const MultimodalCheckpointData &) = delete;
    MultimodalCheckpointData & operator=(const MultimodalCheckpointData &) = delete;
    ~MultimodalCheckpointData() { if (accounting) accounting->live_bytes -= bytes(); }
    std::vector<uint8_t> recurrent;
    std::vector<uint8_t> draft_carry;
    std::vector<float> tail_mean;
    std::shared_ptr<MultimodalCheckpointAccounting> accounting;
    size_t bytes() const { return recurrent.size() + draft_carry.size() + tail_mean.size()*sizeof(float); }
};

struct MultimodalCheckpoint {
    int row = 0;
    bool media_boundary = false;
    std::shared_ptr<const MultimodalCheckpointData> data;
};

struct MultimodalQuery {
    int begin = -1, end = -1, force = -1;
    std::string user;
    std::shared_ptr<kvmem_prompt> prefix;
    std::vector<std::pair<uint32_t, std::string>> media;
    llama_kvmem_query_state state;
};

// ---- auto-continue-thinking (ACT) ------------------------------------------
// When one generation hits its output limit while the model is still inside a
// thinking block, keep the chain in context and start another round instead of
// cutting the answer short. Required because KVMem caps a single generation at
// --kvmem-gen-reserve (thinking included), so a long chain must be split.
struct kvmem_act_params {
    bool        enabled      = false;
    int         max_rounds   = 8;    // hard safety valve
    int         round_tokens = 0;    // per-round budget; 0 = request max_tokens
    int         total_tokens = 0;    // total budget; 0 = unlimited
    std::string cue          = "\n"; // injected between rounds (keeps LCP reuse cheap)
    std::string think_start  = "<think>";
    std::string think_end    = "</think>";
    bool        thinking_on  = true; // per-request: is thinking enabled at all
    // how a continuation round re-ingests the chain:
    //   auto    - advance the prompt view so the "user" query policy sees it
    //   legacy  - force the non-multimodal prefill path for continuation rounds
    //   restart - drop the GPU window and re-evaluate the whole prompt
    std::string prefill_mode = "auto";
};

// true while the model is still inside its thinking block.
// NOTE: the opening <think> tag is emitted by the chat template, i.e. it lives
// in the prompt, not in the generated text. So "still thinking" simply means
// the closing tag has not been produced yet.
static bool act_inside_thinking(const std::string & s, const kvmem_act_params & a) {
    if (!a.thinking_on || a.think_end.empty()) {
        return false;
    }
    return s.find(a.think_end) == std::string::npos;
}

// idempotent: only appends the closing tag when the block is still open
static bool act_close_thinking(std::string & s, const kvmem_act_params & a) {
    if (a.think_end.empty() || s.find(a.think_end) != std::string::npos) {
        return false;
    }
    s += a.think_end;
    return true;
}

// per-round budget actually handed to the generator
struct ServerState {
    std::mutex mu;
    kvmem_server_progress progress;
    kvmem_server_log log;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr tmpls;
    llama_kvmem_params kparams {};
    int n_batch = 512;
    int n_predict_default = -1;
    json sampling_overrides = json::object();
    int query_last_fallback = 64;
    int query_max_tokens = 512;
    bool query_replay_auto = true;
    bool query_policy_user = true;
    uint32_t turn_generation_rows = 0;
    bool turn_query_exact = false;
    std::string model_name = "kvmem";
    kvmem_spec_session spec;
    ggml_type cache_type_k = GGML_TYPE_Q8_0;
    ggml_type cache_type_v = GGML_TYPE_Q8_0;
    ggml_type spec_cache_type = GGML_TYPE_F16;
    bool spec_mtp = false;
    int spec_n_max = 3;
    float spec_p_min = 0.0f;
    bool enable_thinking_default = false;
    std::map<std::string, std::string> template_kwargs;
    int reasoning_budget_default = -1;
    std::string reasoning_budget_message;
    std::vector<llama_token> cached_tokens;
    std::unique_ptr<kvmem_vision> vision;
    std::shared_ptr<kvmem_prompt> active_prompt;
    std::shared_ptr<kvmem_prompt> cached_prompt;
    std::vector<MultimodalCheckpoint> mm_checkpoints;
    std::shared_ptr<MultimodalCheckpoint> mm_rollback;
    std::shared_ptr<kvmem_prompt> mm_rollback_prompt;
    int mm_live_row = 0;
    std::shared_ptr<const MultimodalCheckpointData> mm_live_checkpoint;
    kvmem_prefill_perf mm_perf;
    std::shared_ptr<MultimodalCheckpointAccounting> mm_checkpoint_accounting = std::make_shared<MultimodalCheckpointAccounting>();
    std::shared_ptr<const MultimodalQuery> mm_query;
    std::shared_ptr<const MultimodalQuery> mm_pending_query;
    bool mm_committed = true;
    uint32_t mm_new_text = 0;
    uint32_t mm_new_image = 0;
    uint32_t mm_replayed = 0;
    uint32_t mm_tail_replayed = 0;
    int mm_lcp = 0;
    std::string mm_error;
    int mm_error_status = 500;
    bool mm_reset_requested = false;
    int perf_p_eval = 0;
    std::vector<uint8_t> gdn_ckpt;
    std::vector<uint8_t> gdn_carry, gdn_query_carry;
    int gdn_ckpt_pos = -1; // gen-start (eval_end-1); next-turn suffix rewind
    std::vector<uint8_t> gdn_ckpt_query;
    int gdn_ckpt_query_pos = -1; // last query-begin; fallback if LCP < gen-start
    // Last query span that actually landed query+suffix on GPU (retrieval
    // replay or a later same-query skip). -1 = nothing to skip against.
    int last_query_begin = -1;
    int last_query_end = -1;
    std::string last_user_text;
    std::string turn_last_user;
    int last_n_gen = 0;
    kvmem_act_params act;              // auto-continue-thinking
};

// Prepare the state a continuation round needs.
//
// run_prefill_multimodal() takes no prompt argument: it reads st.active_prompt
// and st.cached_prompt. st.active_prompt is only assigned once per request, so
// without advancing it here the multimodal ("user" query policy) path computes
// eval_end/lcp from the ORIGINAL prompt, decides that there is nothing new to
// evaluate, and the next spec verify then trips over "non-consecutive token
// position". Pushing the committed chain into active_prompt makes it see the
// new rows and decode only the cue.
static void act_prepare_next_round(ServerState & st, const llama_vocab * vocab,
                                   const std::vector<llama_token> & gen, int budget) {
    if (!st.query_policy_user || !st.active_prompt) {
        return;
    }
    std::vector<llama_token> tail = gen;
    if (!st.act.cue.empty() && vocab) {
        const auto cue_ids = common_tokenize(vocab, st.act.cue, false, true);
        tail.insert(tail.end(), cue_ids.begin(), cue_ids.end());
    }
    st.active_prompt = st.active_prompt->with_generated(tail);
    // can_append() budgets the rows it may append using turn_generation_rows,
    // which is sized for a whole request; resize it to this round's budget.
    st.turn_generation_rows = (uint32_t) std::max(0, budget)
            + (st.spec.ok ? (uint32_t) std::max(0, st.spec_n_max) + 1u : 0u);
}

// per-round budget actually handed to the generator
static int act_round_budget(const kvmem_act_params & a, bool on, int max_tokens) {
    if (!on || a.round_tokens <= 0) {
        return max_tokens;
    }
    return a.round_tokens;
}

struct StreamIo {
    httplib::DataSink * sink = nullptr;
    const httplib::Request * req = nullptr;
    std::chrono::steady_clock::time_point last_beat{};
    bool aborted = false;
};

static bool stream_peer_gone(const StreamIo * io) {
    if (!io) {
        return false;
    }
    if (io->req && io->req->is_connection_closed && io->req->is_connection_closed()) {
        return true;
    }
    if (io->sink && io->sink->is_writable && !io->sink->is_writable()) {
        return true;
    }
    return false;
}

static bool stream_heartbeat(StreamIo * io) {
    if (!io) {
        return true;
    }
    if (stream_peer_gone(io)) {
        io->aborted = true;
        return false;
    }
    if (!io->sink || !io->sink->write) {
        return true;
    }
    const auto now = std::chrono::steady_clock::now();
    if (io->last_beat.time_since_epoch().count() == 0) {
        io->last_beat = now;
        return true;
    }
    if (now - io->last_beat < std::chrono::seconds(10)) {
        return true;
    }
    static const char beat[] = ": keepalive\n\n";
    if (!io->sink->write(beat, sizeof(beat) - 1)) {
        io->aborted = true;
        return false;
    }
    io->last_beat = now;
    return true;
}

static int decode_span(llama_context * ctx, const llama_token * toks, int pos0, int pos1, int n_batch,
                       const char * what, StreamIo * io = nullptr);
static int decode_span_maybe_spec(ServerState & st, const llama_token * toks, int pos0, int pos1,
                                  const char * what, StreamIo * io = nullptr);
static void multimodal_commit(ServerState & st, const std::vector<llama_token> & gen);

static bool gdn_sync_to(ServerState & st, const std::vector<llama_token> & prompt, int n_past,
                        StreamIo * io = nullptr) {
    if (!llama_kvmem_has_recurrent()) {
        return true;
    }
    const llama_pos want = n_past - 1;
    llama_pos rmax = llama_kvmem_recr_pos_max();
    std::vector<uint8_t> carry;
    llama_pos draft_rows = n_past;
    if (st.spec.ok) {
        common_speculative_get_state(st.spec.spec, 0, carry);
        if (carry.size() >= sizeof(draft_rows)) std::memcpy(&draft_rows, carry.data(), sizeof(draft_rows));
    }
    if (rmax == want && draft_rows == n_past) return true;
    const std::vector<uint8_t> * saved_carry = nullptr;
    const uint8_t * blob = nullptr;
    size_t blob_n = 0;
    int ckpt_pos = -1;
    auto consider = [&](const std::vector<uint8_t> & buf, int pos, const std::vector<uint8_t> & saved) {
        if (buf.empty() || pos < 0 || pos > want) {
            return;
        }
        if (pos >= ckpt_pos) {
            blob = buf.data();
            blob_n = buf.size();
            ckpt_pos = pos;
            saved_carry = &saved;
        }
    };
    consider(st.gdn_ckpt, st.gdn_ckpt_pos, st.gdn_carry);
    consider(st.gdn_ckpt_query, st.gdn_ckpt_query_pos, st.gdn_query_carry);
    if (blob == nullptr) {
        fprintf(stderr, "KVMEM_TRACE gdn_sync fail rmax=%d want=%d ckpt_pos=%d query_pos=%d\n",
                (int) rmax, (int) want, st.gdn_ckpt_pos, st.gdn_ckpt_query_pos);
        return false;
    }
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    if (llama_state_seq_set_data_ext(st.ctx, blob, blob_n, 0, fl) != blob_n) {
        fprintf(stderr, "GDN catch-up restore failed\n");
        return false;
    }
    const int from = ckpt_pos + 1;
    if (st.spec.ok) {
        if (!saved_carry || saved_carry->empty()) return false;
        common_speculative_set_state(st.spec.spec, 0, *saved_carry);
        if (!llama_kvmem_remove_logical(st.spec.ctx_dft, from, -1)) return false;
    }
    if (from < n_past) {
        llama_kvmem_set_replay(true);
        const int rc = decode_span_maybe_spec(st, prompt.data(), from, n_past, "gdn-catchup", io);
        llama_kvmem_set_replay(false);
        if (rc == KVMEM_DECODE_ABORT) {
            if (io) {
                io->aborted = true;
            }
            return false;
        }
        if (rc != 0) {
            return false;
        }
    }
    rmax = llama_kvmem_recr_pos_max();
    kvmem_diag("KVMEM_TRACE gdn_sync ckpt_pos=%d from=%d n_past=%d rmax=%d\n",
            ckpt_pos, from, n_past, (int) rmax);
    return rmax == want;
}

static int common_token_prefix(const std::vector<llama_token> & a,
                               const std::vector<llama_token> & b) {
    const int n = (int) std::min(a.size(), b.size());
    int i = 0;
    while (i < n && a[(size_t) i] == b[(size_t) i]) {
        i++;
    }
    return i;
}

static void memory_clear_all(ServerState & st) {
    llama_memory_t mem = llama_get_memory(st.ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            llama_memory_clear(md, true);
        }
    }
    st.cached_tokens.clear();
    st.cached_prompt.reset();
    st.mm_checkpoints.clear();
    st.mm_live_row = 0;
    st.mm_live_checkpoint.reset();
    st.mm_query.reset();
    st.mm_pending_query.reset();
    st.gdn_ckpt.clear();
    st.gdn_carry.clear();
    st.gdn_query_carry.clear();
    if (st.spec.ok) {
        std::vector<uint8_t> carry;
        common_speculative_get_state(st.spec.spec, 0, carry);
        std::fill(carry.begin(), carry.end(), 0);
        common_speculative_set_state(st.spec.spec, 0, carry);
    }
    st.gdn_ckpt_pos = -1;
    st.gdn_ckpt_query.clear();
    st.gdn_ckpt_query_pos = -1;
    st.last_query_begin = -1;
    st.last_query_end = -1;
    st.last_user_text.clear();
    st.last_n_gen = 0;
}

// Persist GDN after a successful prefill (eval_end-1) for the next turn's
// suffix rewind. Intra-turn query rewind uses a local snapshot, not this slot.
static void persist_gdn_ckpt_gen_start(ServerState & st, int eval_end) {
    if (!st.ctx || eval_end <= 0 || !llama_kvmem_has_recurrent()) {
        return;
    }
    llama_synchronize(st.ctx);
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t sz = llama_state_seq_get_size_ext(st.ctx, 0, fl);
    if (sz == 0) {
        kvmem_diag("KVMEM_TRACE gdn_ckpt gen_start skipped size=0 eval_end=%d\n", eval_end);
        return;
    }
    std::vector<uint8_t> buf(sz);
    if (llama_state_seq_get_data_ext(st.ctx, buf.data(), sz, 0, fl) != sz) {
        fprintf(stderr, "KVMEM_TRACE gdn_ckpt gen_start copy failed eval_end=%d\n", eval_end);
        return;
    }
    st.gdn_ckpt.swap(buf);
    if (st.spec.ok) common_speculative_get_state(st.spec.spec, 0, st.gdn_carry);
    st.gdn_ckpt_pos = eval_end - 1;
    kvmem_diag("KVMEM_TRACE gdn_ckpt pos_end=%d bytes=%zu ckpt_pos=%d what=gen_start\n",
            eval_end, sz, st.gdn_ckpt_pos);
}

static void commit_cached(ServerState & st, const std::vector<llama_token> & prompt,
                          const std::vector<llama_token> & gen) {
    st.cached_tokens = prompt;
    st.cached_tokens.insert(st.cached_tokens.end(), gen.begin(), gen.end());
    st.last_n_gen = (int) gen.size();
    if (st.vision || st.query_policy_user) multimodal_commit(st, gen);
    else st.cached_prompt = st.active_prompt->with_generated(gen);
    kvmem_diag("KVMEM_TRACE cache_commit n_prompt=%d n_gen=%d n_cached=%d stored=%u\n",
            (int) prompt.size(), (int) gen.size(), (int) st.cached_tokens.size(),
            llama_kvmem_store_n_tokens());
}

static int decode_span(llama_context * ctx, const llama_token * toks, int pos0, int pos1, int n_batch,
                       const char * what, StreamIo * io) {
    if (pos0 >= pos1) {
        return 0;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    // Explicit pos: T5 query sits in the middle of the prompt (last user, then
    // assistant tool XML + role=tool). llama_batch_get_one would append at
    // seq_pos_max+1 and miss the hole after seq_rm(q0,q1).
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_pos = pos0;
    while (n_pos < pos1) {
        if (!stream_heartbeat(io)) {
            kvmem_diag("KVMEM_TRACE stream_abort phase=prefill pos=%d what=%s\n",
                    n_pos, what ? what : "");
            llama_batch_free(batch);
            return KVMEM_DECODE_ABORT;
        }
        const int n = std::min(n_batch, pos1 - n_pos);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, toks[n_pos + i], n_pos + i, { 0 }, i == n - 1);
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n", what, rc, n_pos, n);
            llama_batch_free(batch);
            return rc;
        }
        n_pos += n;
    }
    llama_batch_free(batch);
    return 0;
}

static int decode_span_maybe_spec(ServerState & st, const llama_token * toks, int pos0, int pos1,
                                 const char * what, StreamIo * io) {
    if (st.spec.ok) {
        auto abort_fn = [io]() { return !stream_heartbeat(io); };
        return kvmem_spec_decode_span(st.ctx, st.spec.spec, toks, pos0, pos1, st.n_batch, what, abort_fn);
    }
    return decode_span(st.ctx, toks, pos0, pos1, st.n_batch, what, io);
}

#include "kvmem-multimodal-server.h"

static bool run_prefill_retrieval(ServerState & st, const std::vector<llama_token> & prompt,
                                 StreamIo * io = nullptr, int * n_cache_hit = nullptr) {
    if (st.vision || st.query_policy_user) return run_prefill_multimodal(st, io, n_cache_hit);
    if (!stream_heartbeat(io)) {
        kvmem_diag("KVMEM_TRACE stream_abort phase=prefill_start n_prompt=%d\n",
                (int) prompt.size());
        return false;
    }
    const int n_prompt = (int) prompt.size();
    llama_context * ctx = st.ctx;
    const int eval_end = st.spec.ok ? n_prompt - 1 : n_prompt;

    const bool do_retr = st.kparams.enabled && st.kparams.method == 1 && st.kparams.query_begin > 0;
    int q0 = st.kparams.query_begin;
    int q1 = st.kparams.query_end;
    if (q0 < 0) {
        q0 = 0;
    }
    if (q1 <= q0 || q1 > eval_end) {
        q1 = eval_end;
    }
    const bool replay_fits = llama_kvmem_query_replay_fits(
            (uint32_t) std::max(q0, 0), (uint32_t) std::max(eval_end, 0));

    int n_past = 0;
    bool reused = false;
    uint32_t stored = st.kparams.enabled ? llama_kvmem_store_n_tokens()
                                         : (uint32_t) st.cached_tokens.size();
    if (!st.cached_tokens.empty() && n_prompt > 1) {
        const int lcp = common_token_prefix(st.cached_tokens, prompt);
        kvmem_diag("KVMEM_TRACE prefix_try lcp=%d n_cached=%d stored=%u n_prompt=%d\n",
                lcp, (int) st.cached_tokens.size(), stored, n_prompt);
        reused = lcp > 0 && lcp < n_prompt && stored >= (uint32_t) lcp;
        if (reused) {
            n_past = lcp;
        }
    }

    llama_pos kv_smax = -1;
    if (llama_memory_t mem = llama_get_memory(ctx)) {
        kv_smax = llama_memory_seq_pos_max(mem, 0);
    }
    const llama_pos gdn_rmax = llama_kvmem_has_recurrent()
            ? llama_kvmem_recr_pos_max()
            : (n_past > 0 ? (llama_pos) (n_past - 1) : (llama_pos) -1);
    const bool same_query = !st.last_user_text.empty() &&
            st.last_user_text == st.turn_last_user;
    const bool gdn_at_tip = !llama_kvmem_has_recurrent() ||
            (n_past > 0 && gdn_rmax == (llama_pos) (n_past - 1));
    const bool kv_at_tip = n_past > 0 && kv_smax >= (llama_pos) (n_past - 1);
    // Same last-user: keep the GPU window and only prefill the new tail.
    // Suffix after query is recency (recent_tokens), not skip-gated.
    // New user / miss / GDN not at tip / no gen slots → full retrieval.
    const int n_cached = (int) st.cached_tokens.size();
    // Continuation: LCP covers the previous cache except last gen (thinking
    // stripped / re-templated). +64 is a few prompt-side template tokens.
    // Compact leaves LCP far short of n_cached.
    const uint32_t suffix_slack = (uint32_t) std::max(0, st.last_n_gen) + 64u;
    const bool suffix_cont = reused
            && (uint32_t) (n_cached - n_past) <= suffix_slack;
    if (reused && !suffix_cont) {
        kvmem_diag("KVMEM_TRACE prefix_rewrite drop_reuse=1 n_past=%d n_cached=%d "
                "n_prompt=%d last_n_gen=%d slack=%u same_query=%d\n",
                n_past, n_cached, n_prompt, st.last_n_gen, suffix_slack,
                (int) same_query);
        memory_clear_all(st);
        n_past = 0;
        reused = false;
    }
    const uint32_t n_new_tok = (uint32_t) std::max(0, eval_end - n_past);
    const uint32_t bt = std::max(1u, st.kparams.block_tokens);
    const uint32_t need_slots = n_new_tok == 0 ? 0u : (n_new_tok + bt - 1) / bt;
    const uint32_t free_slots = llama_kvmem_free_slots();
    bool past_query = n_past > q1;
    bool warm_skip = do_retr && reused && suffix_cont && same_query && past_query &&
            gdn_at_tip && kv_at_tip && free_slots >= need_slots;

    if (reused) {
        if (st.kparams.enabled) {
            if (past_query && same_query) {
                llama_kvmem_begin_cached_turn_keep_query();
            } else {
                llama_kvmem_begin_cached_turn();
            }
            if (warm_skip) {
                llama_kvmem_keep_selected();
            }
        }
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_seq_rm(mem, 0, n_past, -1);
        }
        if (st.spec.ctx_dft) {
            llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
            if (md) {
                llama_memory_seq_rm(md, 0, n_past, -1);
            }
        }
        if (st.kparams.enabled) {
            llama_kvmem_truncate_cached((uint32_t) n_past);
        }
        // Continuation already has GDN at n_past. Catch-up from query would
        // llama_decode at q0 while seq_pos_max is n_past-1 (M-RoPE X < Y).
        if (!warm_skip && !gdn_sync_to(st, prompt, n_past, io)) {
            if (io && io->aborted) {
                return false;
            }
            reused = false;
            warm_skip = false;
        }
    }
    if (!reused) {
        memory_clear_all(st);
        n_past = 0;
        warm_skip = false;
        past_query = false;
    }
    // DeepSeek usage: prefix cache hit = kept LCP (n_past). Full miss if reuse
    // was dropped (gdn_sync fail / empty cache).
    if (n_cache_hit) {
        *n_cache_hit = n_past < 0 ? 0 : n_past;
        if (*n_cache_hit > n_prompt) {
            *n_cache_hit = n_prompt;
        }
    }

    // GDN ckpt/rewind only on the first pass of this query. Continuation
    // already has GDN at n_past; replaying the decode suffix is recency, not
    // a mandatory catch-up.
    const bool recr_ckpt = do_retr && replay_fits && llama_kvmem_has_recurrent() &&
            !warm_skip && !past_query;
    kvmem_diag("KVMEM_TRACE prefix_reuse reused=%d n_past=%d n_prompt=%d n_cached=%d "
            "n_new=%d stored=%u query=[%d,%d) replay_fits=%d warm_skip=%d "
            "same_query=%d suffix_cont=%d last_n_gen=%d slack=%u "
            "gdn_rmax=%d kv_smax=%d free_slots=%u need_slots=%u\n",
            (int) reused, n_past, n_prompt, (int) st.cached_tokens.size(),
            eval_end - n_past, llama_kvmem_store_n_tokens(),
            q0, q1, (int) replay_fits, (int) warm_skip, (int) same_query,
            (int) suffix_cont, st.last_n_gen, suffix_slack,
            (int) gdn_rmax, (int) kv_smax, free_slots, need_slots);

    auto take_rc = [&](int rc) -> bool {
        if (rc == KVMEM_DECODE_ABORT) {
            if (io) {
                io->aborted = true;
            }
            return false;
        }
        return rc == 0;
    };
    auto dec = [&](int a, int b, const char * what) -> bool {
        if (a < 0) {
            a = 0;
        }
        if (b > eval_end) {
            b = eval_end;
        }
        if (a >= b) {
            return true;
        }
        return take_rc(decode_span_maybe_spec(st, prompt.data(), a, b, what, io));
    };
    // Hole-fill / recapture of positions that may already sit in KV. Trunk
    // only: MTP draft is M-RoPE and cannot decode Y while X (seq_pos_max)
    // is still ahead of Y.
    auto replay = [&](int a, int b, const char * what) -> bool {
        if (a < 0) {
            a = 0;
        }
        if (b > eval_end) {
            b = eval_end;
        }
        if (a >= b) {
            return true;
        }
        llama_kvmem_set_replay(true);
        const int rc = decode_span(st.ctx, prompt.data(), a, b, st.n_batch, what, io);
        llama_kvmem_set_replay(false);
        return take_rc(rc);
    };

    auto note_prefill = [&]() {
        llama_synchronize(ctx);
        const llama_perf_context_data p = llama_perf_context(ctx);
        const int d = p.n_p_eval - st.perf_p_eval;
        st.perf_p_eval = p.n_p_eval;
        kvmem_diag("KVMEM_TRACE prefix_prefill n_p_eval=%d reused=%d n_past=%d n_new=%d\n",
                d, (int) reused, n_past, eval_end - n_past);
    };
    auto commit_last_query = [&](bool ok) {
        if (ok) {
            st.last_query_begin = q0;
            st.last_query_end = q1;
            st.last_user_text = st.turn_last_user;
        } else {
            st.last_query_begin = -1;
            st.last_query_end = -1;
            st.last_user_text.clear();
        }
    };

    if (warm_skip) {
        kvmem_diag("KVMEM_TRACE query_replay_skip_same query=[%d,%d) n_past=%d n_new=%d\n",
                q0, q1, n_past, eval_end - n_past);
        if (!dec(n_past, eval_end, "prefill-tail")) {
            return false;
        }
        if (st.spec.ctx_dft) {
            llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
            if (md) {
                kvmem_diag("KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                        llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
            }
        }
        llama_kvmem_pin_working_set();
        note_prefill();
        commit_last_query(true);
        persist_gdn_ckpt_gen_start(st, eval_end);
        return true;
    }

    // Same last-user, query already in the prefix, but skip could not keep
    // the window (usually gen-reserve full). Reuse the captured Q for top-k;
    // do not llama_decode at q0 (M-RoPE requires seq_pos_max < q0).
    if (do_retr && reused && suffix_cont && same_query && past_query) {
        kvmem_diag("KVMEM_TRACE query_reuse_q reselect=1 query=[%d,%d) n_past=%d n_new=%d\n",
                q0, q1, n_past, eval_end - n_past);
        llama_kvmem_apply_retrieval(ctx);
        if (!dec(n_past, eval_end, "prefill-tail")) {
            return false;
        }
        if (st.spec.ctx_dft) {
            llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
            if (md) {
                kvmem_diag("KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                        llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
            }
        }
        llama_kvmem_pin_working_set();
        note_prefill();
        commit_last_query(true);
        persist_gdn_ckpt_gen_start(st, eval_end);
        return true;
    }

    if (!do_retr) {
        if (!dec(n_past, eval_end, reused ? "prefill-suffix" : "prefill")) {
            return false;
        }
        note_prefill();
        commit_last_query(false);
        persist_gdn_ckpt_gen_start(st, eval_end);
        return true;
    }

    llama_kvmem_reset_query();
    if (!dec(n_past, q0, reused ? "prefill-suffix" : "prefill")) {
        return false;
    }
    std::vector<uint8_t> gdn_ckpt;
    if (recr_ckpt) {
        if (n_past > q0 && !gdn_sync_to(st, prompt, q0, io)) {
            LOG_ERR("srv    KVMEM_TRACE gdn_sync to query_begin failed\n");
            return false;
        }
        llama_synchronize(ctx);
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, fl);
        if (sz == 0) {
            fprintf(stderr, "GDN checkpoint size 0\n");
            return false;
        }
        gdn_ckpt.resize(sz);
        if (llama_state_seq_get_data_ext(ctx, gdn_ckpt.data(), sz, 0, fl) != sz) {
            fprintf(stderr, "GDN checkpoint copy failed\n");
            return false;
        }
        st.gdn_ckpt_query = gdn_ckpt;
        if (st.spec.ok) common_speculative_get_state(st.spec.spec, 0, st.gdn_query_carry);
        st.gdn_ckpt_query_pos = q0 > 0 ? q0 - 1 : -1;
        kvmem_diag("KVMEM_TRACE gdn_ckpt_query pos_end=%d bytes=%zu query_begin=%d ckpt_pos=%d\n",
                q0, sz, q0, st.gdn_ckpt_query_pos);
    }
    // Query may already sit inside the reused prefix (T5: last user, then
    // assistant tool XML + role=tool). Recapture Q over the cached part
    // (replay skips K/mean); prefill only the missing tail of the span.
    if (n_past > q0 && n_past < q1) {
        if (!replay(q0, n_past, "query-q-capture")) {
            return false;
        }
    }
    if (n_past < q1) {
        if (!dec(std::max(n_past, q0), q1, "prefill-query")) {
            return false;
        }
    } else if (!replay(q0, q1, "query-q-capture")) {
        return false;
    }
    kvmem_diag("KVMEM_TRACE query_q_capture n_past=%d query=[%d,%d) recapture=%d\n",
            n_past, q0, q1, (int) (n_past > q0));
    llama_synchronize(ctx);
    {
        const llama_perf_context_data p = llama_perf_context(ctx);
        const int d = p.n_p_eval - st.perf_p_eval;
        st.perf_p_eval = p.n_p_eval;
        kvmem_diag("KVMEM_TRACE prefix_prefill n_p_eval=%d reused=%d n_past=%d n_new=%d\n",
                d, (int) reused, n_past, eval_end - n_past);
    }

    const int tail0 = std::max(n_past, q1);
    const uint32_t tail_tok = (uint32_t) std::max(0, eval_end - tail0);
    const uint32_t gen_res = std::max(1u, st.kparams.gen_reserve);
    const bool tail_fits_gen = tail_tok <= gen_res;

    if (tail_fits_gen) {
        llama_kvmem_apply_retrieval(ctx);
        if (!replay_fits) {
            kvmem_diag("KVMEM_TRACE query_replay_skip query=[%d,%d) eval_end=%d "
                    "(sink+suffix exceeds GPU budget)\n",
                    q0, q1, eval_end);
        } else {
            if (recr_ckpt && !gdn_ckpt.empty()) {
                const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
                if (llama_state_seq_set_data_ext(ctx, gdn_ckpt.data(), gdn_ckpt.size(), 0, fl) != gdn_ckpt.size()) {
                    fprintf(stderr, "GDN restore failed\n");
                    return false;
                }
                kvmem_diag("KVMEM_TRACE gdn_restore bytes=%zu\n", gdn_ckpt.size());
            }
            llama_memory_t mem = llama_get_memory(ctx);
            if (mem) {
                kvmem_diag("KVMEM_TRACE before_seq_rm seq_pos=[%d,%d] query=[%d,%d)\n",
                        llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                        q0, q1);
                llama_memory_seq_rm(mem, 0, q0, q1);
                kvmem_diag("KVMEM_TRACE after_seq_rm seq_pos=[%d,%d] auto_pos0=%d\n",
                        llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                        llama_memory_seq_pos_max(mem, 0) + 1);
            }
            if (st.spec.ctx_dft && !past_query) {
                llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
                if (md) {
                    llama_memory_seq_rm(md, 0, q0, q1);
                    kvmem_diag("KVMEM_TRACE mtp_after_seq_rm seq_pos=[%d,%d]\n",
                            llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
                }
            }
            if (!replay(q0, q1, "query replay")) {
                return false;
            }
            llama_synchronize(ctx);
            kvmem_diag("KVMEM_TRACE query_replay begin=%d n=%d recr_ckpt=%d\n",
                    q0, q1 - q0, (int) recr_ckpt);
            if (st.spec.ctx_dft && !past_query) {
                // First pass of this query: seq_rm left a hole in the draft cache.
                // M-RoPE cannot fill it while a suffix remains, so drop [q0, inf)
                // and append in order up to q1. Continuation keeps draft suffix.
                llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
                if (md) {
                    llama_memory_seq_rm(md, 0, q0, -1);
                }
                if (q1 > q0) {
                    const int rc = decode_span(st.spec.ctx_dft, prompt.data(), q0, q1, st.n_batch,
                                               "mtp-resync", io);
                    if (!take_rc(rc)) {
                        return false;
                    }
                    kvmem_diag("KVMEM_TRACE mtp_resync query=[%d,%d) to=%d\n", q0, q1, q1);
                }
            }
        }
        if (!dec(tail0, eval_end, "prefill-tail")) {
            return false;
        }
    } else {
        // Compact / long history after last-user: tail is not this turn's
        // decode slack. Prefill with spill, then retrieve.
        kvmem_diag("KVMEM_TRACE prefill_tail_offload n=%u gen_reserve=%u query=[%d,%d)\n",
                tail_tok, gen_res, q0, q1);
        if (!dec(tail0, eval_end, "prefill-tail")) {
            return false;
        }
        llama_kvmem_apply_retrieval(ctx);
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            kvmem_diag("KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                    llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
        }
    }
    commit_last_query(true);
    persist_gdn_ckpt_gen_start(st, eval_end);
    return true;
}

static std::string trim_copy(const std::string & s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) {
        ++a;
    }
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\r' || s[b - 1] == '\t')) {
        --b;
    }
    return s.substr(a, b - a);
}

// Last ChatML user *role block* in the rendered prompt, not rfind(content).
// Thinking/tool text can quote last_user; only <|im_start|>user ... <|im_end|>
// counts. If last_user is set, pick the last block whose content equals it
// (two identical user turns → the later block).
static bool find_last_user_role_block(const std::string & prompt, const std::string & last_user,
                                      size_t & content0, size_t & content1, int & n_blocks, int & pick) {
    static const char * hdrs[] = {
        "<|im_start|>user\n",
        "<|im_start|>user\r\n",
        "<|im_start|>user",
    };
    struct Blk {
        size_t c0;
        size_t c1;
    };
    std::vector<Blk> blks;
    size_t search = 0;
    while (search < prompt.size()) {
        size_t best = std::string::npos;
        size_t best_len = 0;
        for (const char * h : hdrs) {
            const size_t n = std::strlen(h);
            const size_t p = prompt.find(h, search);
            if (p == std::string::npos) {
                continue;
            }
            if (best == std::string::npos || p < best || (p == best && n > best_len)) {
                best = p;
                best_len = n;
            }
        }
        if (best == std::string::npos) {
            break;
        }
        const size_t c0 = best + best_len;
        const size_t end = prompt.find("<|im_end|>", c0);
        if (end == std::string::npos) {
            break;
        }
        blks.push_back(Blk{c0, end});
        search = best + 1;
    }
    n_blocks = (int) blks.size();
    pick = -1;
    if (blks.empty()) {
        return false;
    }
    const std::string want = trim_copy(last_user);
    if (!want.empty()) {
        for (int i = n_blocks - 1; i >= 0; --i) {
            const std::string got = trim_copy(prompt.substr(blks[(size_t) i].c0,
                    blks[(size_t) i].c1 - blks[(size_t) i].c0));
            if (got == want) {
                pick = i;
                break;
            }
        }
    }
    if (pick < 0) {
        // No exact content match (template wrapping). Do not fall back to
        // the last user-role header: Qwen tools are often rendered as user
        // + <tool_response>. Leave the caller to query-last fallback.
        if (!want.empty()) {
            return false;
        }
        pick = n_blocks - 1;
    }
    content0 = blks[(size_t) pick].c0;
    content1 = blks[(size_t) pick].c1;
    return content0 < content1;
}

static void derive_query_span(ServerState & st, const std::string & prompt, const std::string & last_user,
                              const std::vector<llama_token> & toks, int & qbegin, int & qend) {
    qbegin = -1;
    qend = (int) toks.size();
    size_t c0 = 0;
    size_t c1 = 0;
    int n_blocks = 0;
    int pick = -1;
    if (find_last_user_role_block(prompt, last_user, c0, c1, n_blocks, pick)) {
        const std::string prefix = prompt.substr(0, c0);
        const std::string through = prompt.substr(0, c1);
        qbegin = (int) tokenize_text(st.vocab, prefix, true).size();
        qend = (int) tokenize_text(st.vocab, through, true).size();
        kvmem_diag("KVMEM_TRACE query_loc method=role_block n_user_blocks=%d pick=%d "
                "span=[%zu,%zu) tokens=[%d,%d)\n",
                n_blocks, pick, c0, c1, qbegin, qend);
    }
    if (qend > (int) toks.size()) {
        qend = (int) toks.size();
    }
    if (qbegin < 0 || qend <= qbegin) {
        qend = (int) toks.size();
        const int last = std::min(st.query_last_fallback, qend);
        qbegin = qend > last ? qend - last : 0;
        kvmem_diag("KVMEM_TRACE query_loc method=query_last tokens=[%d,%d)\n",
                qbegin, qend);
    }
    if (qbegin >= qend) {
        qbegin = 0;
    }
}

static bool derive_native_query_span(const ServerState & st, const std::string & formatted,
                                      const common_chat_templates_inputs & inputs, const kvmem_prompt & prompt,
                                      int & begin, int & end) {
    // Render the structured prefix ending at the real last user. This retains
    // native vision wrappers and does not confuse tool_response user-style
    // blocks with an actual user message. Require an exact causal prefix match.
    auto prefix_inputs = inputs;
    while (!prefix_inputs.messages.empty() && prefix_inputs.messages.back().role != "user")
        prefix_inputs.messages.pop_back();
    if (prefix_inputs.messages.empty()) return false;
    prefix_inputs.add_generation_prompt = false;
    std::string prefix;
    try {
        prefix = common_chat_templates_apply(st.tmpls.get(), prefix_inputs).prompt;
    } catch (const std::exception &) {
        return false; // A template may require the full trailing tool sequence.
    }
    size_t c0 = 0, c1 = 0;
    int count = 0, pick = -1;
    if (!find_last_user_role_block(prefix, "", c0, c1, count, pick) ||
            formatted.compare(0, c1, prefix, 0, c1) != 0) return false;
    int full_count = 0, unused = -1;
    if (!find_last_user_role_block(formatted, "", c0, c1, full_count, unused)) return false;
    common_chat_msg_delimiters delimiters;
    delimiters.add(COMMON_CHAT_ROLE_USER, "<|im_start|>user");
    delimiters.add(COMMON_CHAT_ROLE_UNKNOWN, "<|im_end|>");
    delimiters.tokenize(st.vocab);
    const auto spans = prompt.message_spans(delimiters);
    std::vector<common_chat_msg_span> users;
    for (const auto & span : spans.spans) if (span.role == COMMON_CHAT_ROLE_USER) users.push_back(span);
    if ((int) users.size() != full_count || pick < 0 || pick >= (int) users.size()) return false;
    begin = users[pick].pos + delimiters.delimiters.front().tokens.size();
    end = users[pick].pos + users[pick].len;
    for (const auto & image : prompt.media_ranges()) {
        if ((int) image.first >= begin && (int) image.second <= end) begin = image.second;
    }
    if (begin >= end) return false;
    std::string text;
    for (int row = begin; row < end; ++row)
        text += common_token_to_piece(st.vocab, prompt.tokens[row], false);
    if (trim_copy(text).empty()) return false;
    kvmem_diag("KVMEM_TRACE query_loc method=native_role pick=%d tokens=[%d,%d)\n", pick, begin, end);
    return true;
}

static void clamp_query_span(const ServerState & st, int & qbegin, int & qend) {
    const int cap = st.query_max_tokens;
    if (cap > 0 && qend > qbegin && (qend - qbegin) > cap) {
        kvmem_diag("KVMEM_TRACE query_clamp span=[%d,%d) tokens=%d cap=%d -> [%d,%d)\n",
                qbegin, qend, qend - qbegin, cap, qend - cap, qend);
        qbegin = qend - cap;
    }
}

struct ChatRequest {
    std::vector<common_chat_msg> msgs;
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool parallel_tool_calls = false;
    bool parallel_tool_calls_set = false;
    std::string json_schema;
    std::string grammar;
    std::vector<std::string> stop;
    std::string last_user;
    int max_tokens = 128;
    common_params_sampling sampling;
    bool stream = false;
    int query_begin = -1;
    int query_end = -1;
    std::string force_substr;
    bool enable_thinking = false;
    std::map<std::string, std::string> template_kwargs;
    int reasoning_budget_tokens = -1;
    std::string reasoning_budget_message;
    int act_override = -1;  // -1 = use the server default, 0 = off, 1 = on
};

static common_json nlohmann_to_common(const json & j) {
    return common_json::parse(j.dump());
}

static const char * tool_choice_cstr(common_chat_tool_choice c) {
    switch (c) {
        case COMMON_CHAT_TOOL_CHOICE_NONE:     return "none";
        case COMMON_CHAT_TOOL_CHOICE_REQUIRED: return "required";
        default:                               return "auto";
    }
}

static const char * grammar_type_cstr(common_grammar_type t) {
    switch (t) {
        case COMMON_GRAMMAR_TYPE_TOOL_CALLS:    return "tool_calls";
        case COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT: return "output_format";
        case COMMON_GRAMMAR_TYPE_USER:          return "user";
        default:                                return "none";
    }
}

static common_params_sampling make_chat_sampling(
        const llama_vocab * vocab,
        const common_chat_params & chat,
        const ChatRequest & cr) {
    common_params_sampling sp = cr.sampling;
    kvmem_chat_sampling_normalize(sp);
    std::string g = chat.grammar.empty() ? cr.grammar : chat.grammar;
    if (!g.empty()) {
        common_grammar_type ty = COMMON_GRAMMAR_TYPE_USER;
        if (!cr.tools.empty() && cr.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            ty = COMMON_GRAMMAR_TYPE_TOOL_CALLS;
        } else if (!cr.json_schema.empty()) {
            ty = COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT;
        }
        sp.grammar = {ty, std::move(g)};
    }
    sp.grammar_lazy = chat.grammar_lazy;
    sp.generation_prompt = chat.generation_prompt;
    if (vocab) {
        for (const auto & t : chat.preserved_tokens) {
            const auto ids = common_tokenize(vocab, t, false, true);
            if (ids.size() == 1) {
                sp.preserved_tokens.insert(ids[0]);
            }
        }
        for (const auto & trigger : chat.grammar_triggers) {
            if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                const auto ids = common_tokenize(vocab, trigger.value, false, true);
                if (ids.size() == 1) {
                    common_grammar_trigger tr;
                    tr.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    tr.value = trigger.value;
                    tr.token = ids[0];
                    sp.grammar_triggers.push_back(std::move(tr));
                    continue;
                }
            }
            sp.grammar_triggers.push_back(trigger);
        }
    } else {
        sp.grammar_triggers = chat.grammar_triggers;
    }
    sp.reasoning_budget_tokens = cr.reasoning_budget_tokens;
    sp.reasoning_budget_message = cr.reasoning_budget_message;
    if (vocab && !chat.thinking_end_tags.empty()) {
        if (!chat.thinking_start_tag.empty()) {
            sp.reasoning_budget_start = common_tokenize(vocab, chat.thinking_start_tag, false, true);
        }
        for (const auto & tag : chat.thinking_end_tags) {
            if (tag.empty()) {
                continue;
            }
            auto ids = common_tokenize(vocab, tag, false, true);
            if (!ids.empty()) {
                sp.reasoning_budget_end.push_back(std::move(ids));
            }
        }
        if (!sp.reasoning_budget_end.empty()) {
            llama_tokens forced = sp.reasoning_budget_end.front();
            if (!cr.reasoning_budget_message.empty()) {
                auto msg = common_tokenize(vocab, cr.reasoning_budget_message, false, true);
                forced.insert(forced.begin(), msg.begin(), msg.end());
            }
            sp.reasoning_budget_forced = std::move(forced);
        }
    }
    return sp;
}

static common_chat_msg parse_assistant_output(
        const std::string & content,
        const common_chat_params & chat,
        bool parse_tools) {
    try {
        common_chat_parser_params pp(chat);
        pp.parse_tool_calls = parse_tools;
        if (!chat.parser.empty()) {
            pp.parser.load(chat.parser);
        }
        common_chat_msg msg = common_chat_parse(content, false, pp);
        if (msg.role.empty()) {
            msg.role = "assistant";
        }
        return msg;
    } catch (const std::exception & e) {
        LOG_WRN("srv    KVMEM_TRACE chat_out_parse_fail %s\n", e.what());
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = content;
        return msg;
    }
}

static json message_to_nlohmann(const common_chat_msg & msg) {
    return json::parse(msg.to_json_oaicompat().dump());
}

static json chat_diff_to_delta(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({std::move(tool_call)});
    }
    return delta;
}

struct StreamChatOut {
    common_chat_parser_params pp;
    common_chat_msg prev;
    std::string acc;
    std::vector<std::string> tc_ids;
    std::string request_id;
    int n_id = 0;
    int n_tc_delta = 0;

    StreamChatOut(const common_chat_params & chat, bool parse_tools, const std::string & id) : request_id(id) {
        pp = common_chat_parser_params(chat);
        pp.parse_tool_calls = parse_tools;
        if (!chat.parser.empty()) {
            pp.parser.load(chat.parser);
        }
    }

    std::vector<json> set_text(const std::string & text, bool partial) {
        acc = text;
        std::vector<json> chunks;
        try {
            common_chat_msg msg = common_chat_parse(acc, partial, pp);
            if (msg.empty() && partial) {
                return chunks;
            }
            if (msg.role.empty()) {
                msg.role = "assistant";
            }
            msg.set_tool_call_ids(tc_ids, [this]() {
                return kvmem_chat_tool_id(request_id, ++n_id);
            });
            const auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
            prev = std::move(msg);
            for (const auto & d : diffs) {
                json delta = chat_diff_to_delta(d);
                if (delta.empty()) {
                    continue;
                }
                if (d.tool_call_index != std::string::npos) {
                    n_tc_delta++;
                }
                chunks.push_back(std::move(delta));
            }
        } catch (const std::exception & e) {
            if (!partial) {
                LOG_WRN("srv    KVMEM_TRACE chat_stream_parse_fail %s\n", e.what());
            }
        }
        return chunks;
    }

    const char * finish_reason(bool hit_limit) const {
        if (!prev.tool_calls.empty()) {
            return "tool_calls";
        }
        if (hit_limit) {
            return "length";
        }
        return "stop";
    }
};

static json stream_choice_chunk(const std::string & cid, const std::string & model, std::time_t created, const json & delta, const char * finish, const json * timings = nullptr) {
    json choice = {
        {"index", 0},
        {"delta", delta},
        {"finish_reason", finish ? json(finish) : json(nullptr)},
    };
    // 1:1 upstream: {choices, created, id, model, system_fingerprint, object} (server-task.cpp to_json_oaicompat_chat)
    // 中文：逐字段对齐上游 OpenAI 流式 chunk 结构，客户端可无差别解析
    json chunk = {
        {"choices", json::array({std::move(choice)})},
        {"created", created},
        {"id", cid},
        {"model", model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object", "chat.completion.chunk"},
    };
    if (timings != nullptr) {
        chunk["timings"] = *timings;
    }
    return chunk;
}

// DeepSeek Chat Completions usage: prompt_tokens = hit + miss.
// Hit = reused prefix (n_past / LCP). Do not also emit OpenAI
// prompt_tokens_details.cached_tokens — OpenCode isOverflow would
// double-count cache.read against the context window.
static json usage_json(int n_prompt, int n_gen, int n_cache_hit) {
    if (n_prompt < 0) {
        n_prompt = 0;
    }
    if (n_gen < 0) {
        n_gen = 0;
    }
    if (n_cache_hit < 0) {
        n_cache_hit = 0;
    }
    if (n_cache_hit > n_prompt) {
        n_cache_hit = n_prompt;
    }
    return json{
        {"prompt_tokens", n_prompt},
        {"completion_tokens", n_gen},
        {"total_tokens", n_prompt + n_gen},
        {"prompt_cache_hit_tokens", n_cache_hit},
        {"prompt_cache_miss_tokens", n_prompt - n_cache_hit},
    };
}

// OpenAI: last stream chunk has empty choices + usage, no finish_reason.
// 1:1 upstream final usage chunk: {choices, created, id, model, system_fingerprint, object, usage}
// 中文：流式结尾的 usage 块——choices 为空、无 finish_reason，携带 usage 统计
static json stream_usage_chunk(const std::string & cid, const std::string & model, std::time_t created, int n_prompt, int n_gen, int n_cache_hit) {
    return json{
        {"created", created},
        {"id", cid},
        {"object", "chat.completion.chunk"},
        {"system_fingerprint", std::string(llama_build_info())},
        {"model", model},
        {"choices", json::array()},
        {"usage", usage_json(n_prompt, n_gen, n_cache_hit)},
    };
}

static bool strip_stop(std::string & content, const std::vector<std::string> & stops) {
    for (const auto & s : stops) {
        if (s.empty() || content.size() < s.size()) {
            continue;
        }
        if (content.compare(content.size() - s.size(), s.size(), s) == 0) {
            content.resize(content.size() - s.size());
            return true;
        }
    }
    return false;
}

static bool parse_chat_request(const json & body, ChatRequest & out, std::string & err) {
    if (!body.is_object()) {
        err = "request must be a JSON object";
        return false;
    }
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        err = "messages array required";
        return false;
    }
    try {
        out.msgs = common_chat_msgs_parse_oaicompat(nlohmann_to_common(body.at("messages")));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (out.msgs.empty()) {
        err = "messages array required";
        return false;
    }
    for (auto it = out.msgs.rbegin(); it != out.msgs.rend(); ++it) {
        if (it->role == "user") {
            out.last_user = it->content.empty() ? it->render_content() : it->content;
            break;
        }
    }
    if (body.contains("tools") && !body["tools"].is_null()) {
        try {
            out.tools = common_chat_tools_parse_oaicompat(nlohmann_to_common(body.at("tools")));
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    }
    if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
        const auto & tc = body.at("tool_choice");
        try {
            if (tc.is_string()) {
                out.tool_choice = common_chat_tool_choice_parse_oaicompat(tc.get<std::string>());
            } else if (tc.is_object()) {
                // OpenAI named-function form → required (template/grammar in T2).
                out.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            } else {
                err = "tool_choice must be a string or object";
                return false;
            }
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    }
    if (body.contains("parallel_tool_calls") && body["parallel_tool_calls"].is_boolean()) {
        out.parallel_tool_calls = body["parallel_tool_calls"].get<bool>();
        out.parallel_tool_calls_set = true;
    }
    if (body.contains("grammar") && body["grammar"].is_string()) {
        out.grammar = body["grammar"].get<std::string>();
    }
    if (body.contains("json_schema") && !body["json_schema"].is_null()) {
        out.json_schema = body["json_schema"].dump();
    }
    if (body.contains("response_format") && body["response_format"].is_object()) {
        const auto & rf = body["response_format"];
        const std::string rtype = rf.value("type", "");
        if (rtype == "json_object") {
            if (out.json_schema.empty()) {
                out.json_schema = rf.contains("schema") ? rf["schema"].dump() : "{}";
            }
        } else if (rtype == "json_schema") {
            const auto schema_wrapper = rf.value("json_schema", json::object());
            if (schema_wrapper.contains("schema")) {
                out.json_schema = schema_wrapper["schema"].dump();
            }
        } else if (!rtype.empty() && rtype != "text") {
            err = "response_format type must be text, json_object, or json_schema";
            return false;
        }
    }
    if (!out.tools.empty() && !out.grammar.empty()) {
        err = "Cannot use custom grammar constraints with tools.";
        return false;
    }
    if (body.contains("stop")) {
        const auto & stop = body.at("stop");
        if (stop.is_string()) {
            out.stop.push_back(stop.get<std::string>());
        } else if (stop.is_array()) {
            for (const auto & s : stop) {
                if (s.is_string()) {
                    out.stop.push_back(s.get<std::string>());
                }
            }
        }
    }
    out.stream = body.value("stream", false);
    if (body.contains("kvmem") && body["kvmem"].is_object()) {
        const auto & k = body["kvmem"];
        if (k.contains("query_begin")) {
            out.query_begin = k["query_begin"].get<int>();
        }
        if (k.contains("query_end")) {
            out.query_end = k["query_end"].get<int>();
        }
        if (k.contains("force_substr") && k["force_substr"].is_string()) {
            out.force_substr = k["force_substr"].get<std::string>();
        }
        if (k.contains("pin")) {
            if (k["pin"].is_string()) {
                out.force_substr = k["pin"].get<std::string>();
            } else if (k["pin"].is_array() && !k["pin"].empty() && k["pin"][0].is_string()) {
                out.force_substr = k["pin"][0].get<std::string>();
            }
        }
    }
    if (!kvmem_chat_template_override(body, out.enable_thinking, out.template_kwargs, err)) {
        return false;
    }
    if (!kvmem_chat_reasoning_budget_override(body, out.reasoning_budget_tokens, err)) {
        return false;
    }
    if (body.contains("reasoning_budget_message") && body["reasoning_budget_message"].is_string()) {
        out.reasoning_budget_message = body["reasoning_budget_message"].get<std::string>();
    }
    if (body.contains("auto_continue_thinking")) {
        const auto & v = body["auto_continue_thinking"];
        if (v.is_boolean()) {
            out.act_override = v.get<bool>() ? 1 : 0;
        } else if (v.is_number_integer()) {
            out.act_override = v.get<int>() != 0 ? 1 : 0;
        } else {
            err = "auto_continue_thinking must be a boolean";
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    // Flush asynchronous common logs on every exit, including startup errors.
    struct log_flush_guard { ~log_flush_guard() { common_log_flush(common_log_main()); } } flush_logs;
    std::string ui_dir;
    bool no_ui = false;
    std::string model_path;
    std::string mmproj_path;
    std::string chat_template;
    bool template_set = false;
    std::string template_source;
    json template_defaults = json::object();
    bool mmproj_gpu = true;
    int image_min_tokens = -1, image_max_tokens = -1;
    std::string host = "127.0.0.1";
    std::string nvme_dir;
    int port = 8080;
    int n_ctx = 2048;
    int ngl = 99;
    kvmem_server_devices device_config;
    ServerState st;
    kvmem_server_options options;
    st.kparams.mtp_state = 2; // ReplaySSM by default when MTP is enabled.
    st.kparams.block_tokens = 128;
    st.kparams.gen_reserve = 256;
    st.kparams.recent_tokens = 0;
    st.kparams.method = 1;
    st.kparams.enabled = true;
    st.kparams.query_begin = -1;
    st.kparams.query_end = -1;
    st.kparams.force_pos = -1;

    json config_sources = json::object();
    std::vector<std::pair<std::string, std::string>> config_inputs;
    std::string argument_source = "environment";
    try {
    kvmem_check_environment(kvmem_process_environment());
    auto arguments = kvmem_environment_args([](const char * name) { return std::getenv(name); });
    for (int i = 1; i < argc; ++i) arguments.push_back({argv[i], "cli"});
    for (size_t i = 0; i < arguments.size(); ++i) {
        argument_source = arguments[i].source;
        const char * arg = kvmem_server_arg_alias(arguments[i].value.c_str());
        if (argument_source != "cli") config_inputs.emplace_back(arg, argument_source);
        const auto config_key = kvmem_config_key(arg);
        if (eq(arg, "--api-key") || eq(arg, "--api-key-file")) {
            if (!config_sources.contains(config_key)) config_sources[config_key] = json::array();
            config_sources[config_key].push_back(argument_source);
        } else {
            config_sources[config_key] = argument_source;
        }
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= arguments.size()) throw std::invalid_argument(std::string("missing value for ") + name);
            return arguments[++i].value.c_str();
        };
        if (options.parse(arg, need)) {
            continue;
        } else if (eq(arg, "-h") || eq(arg, "--help")) {
            print_usage(argv[0]);
            return 0;
        } else if (eq(arg, "-m") || eq(arg, "--model")) {
            model_path = need(arg);
        } else if (eq(arg, "--mmproj")) {
            mmproj_path = need(arg);
        } else if (eq(arg, "--mmproj-offload")) {
            mmproj_gpu = true;
        } else if (eq(arg, "--no-mmproj-offload")) {
            mmproj_gpu = false;
        } else if (eq(arg, "--image-min-tokens") || eq(arg, "--image-max-tokens")) {
            const std::string value = need(arg);
            try {
                size_t used = 0;
                const int n = std::stoi(value, &used);
                if (used != value.size() || n <= 0) throw std::invalid_argument("positive integer required");
                (eq(arg, "--image-min-tokens") ? image_min_tokens : image_max_tokens) = n;
            } catch (...) {
                fprintf(stderr, "%s requires a positive integer\n", arg);
                return 1;
            }
        } else if (eq(arg, "--ui-dir")) {
            ui_dir = need(arg);
        } else if (eq(arg, "--no-ui")) {
            no_ui = true;
        } else if (eq(arg, "--ui") || eq(arg, "--webui")) {
            no_ui = false;
        } else if (eq(arg, "--host")) {
            host = need(arg);
        } else if (eq(arg, "--port")) {
            port = kvmem_cli_int(arg, need(arg), 1, 65535);
        } else if (eq(arg, "-c") || eq(arg, "--ctx-size")) {
            n_ctx = kvmem_cli_int(arg, need(arg), 1);
        } else if (eq(arg, "-n") || eq(arg, "--n-predict")) {
            st.n_predict_default = kvmem_cli_int(arg, need(arg), -1);
            if (st.n_predict_default == 0) throw std::invalid_argument("--n-predict requires -1 or a positive integer");
        } else if (eq(arg, "-b") || eq(arg, "--batch-size")) {
            st.n_batch = kvmem_cli_int(arg, need(arg), 1);
        } else if (eq(arg, "-ngl") || eq(arg, "--n-gpu-layers") || eq(arg, "--gpu-layers")) {
            ngl = kvmem_cli_gpu_layers(arg, need(arg));
        } else if (!kvmem_chat_sampling_cli_key(arg).empty()) {
            const auto key = kvmem_chat_sampling_cli_key(arg);
            const char * value = need(arg);
            std::string err;
            try {
                auto parsed = json::parse(value);
                if (!parsed.is_number()) {
                    throw std::runtime_error("expected a number");
                }
                auto sp = kvmem_chat_sampling_defaults(true);
                if (!kvmem_chat_sampling_override(json{{key, parsed}}, sp, err)) {
                    throw std::runtime_error(err);
                }
                st.sampling_overrides[key] = parsed;
            } catch (const std::exception & e) {
                fprintf(stderr, "invalid %s: %s\n", arg, e.what());
                return 1;
            }
        } else if (eq(arg, "--kvmem")) {
            st.kparams.enabled = true;
        } else if (eq(arg, "--no-kvmem")) {
            st.kparams.enabled = false;
        } else if (eq(arg, "--kvmem-budget")) {
            st.kparams.budget = (uint32_t) kvmem_cli_int(arg, need(arg));
        } else if (eq(arg, "--kvmem-block-tokens")) {
            st.kparams.block_tokens = (uint32_t) kvmem_cli_int(arg, need(arg));
        } else if (eq(arg, "--kvmem-gen-reserve")) {
            st.kparams.gen_reserve = (uint32_t) kvmem_cli_int(arg, need(arg));
        } else if (eq(arg, "--kvmem-recent-tokens")) {
            const int v = kvmem_cli_int(arg, need(arg));
            if (v < 0) {
                fprintf(stderr, "invalid --kvmem-recent-tokens (want >= 0)\n");
                return 1;
            }
            st.kparams.recent_tokens = (uint32_t) v;
        } else if (eq(arg, "--kvmem-method")) {
            const char * m = need(arg);
            st.kparams.method = (eq(m, "retrieval") || eq(m, "retrieve")) ? 1 : 0;
        } else if (eq(arg, "--kvmem-query-last")) {
            st.query_last_fallback = kvmem_cli_int(arg, need(arg));
        } else if (eq(arg, "--kvmem-query-replay")) {
            const std::string mode = need(arg);
            if (mode != "legacy" && mode != "auto") { fprintf(stderr, "invalid query replay mode\n"); return 1; }
            st.query_replay_auto = mode == "auto";
        } else if (eq(arg, "--kvmem-query-policy")) {
            const std::string mode = need(arg);
            if (mode != "legacy" && mode != "user") { fprintf(stderr, "invalid query policy\n"); return 1; }
            st.query_policy_user = mode == "user";
        } else if (eq(arg, "--kvmem-mtp-state")) {
            const std::string mode = need(arg);
            if (mode != "snapshots" && mode != "auto" && mode != "replay") {
                fprintf(stderr, "invalid MTP state mode\n");
                return 1;
            }
            st.kparams.mtp_state = mode == "replay" ? 2 : mode == "auto" ? 1 : 0;
        } else if (eq(arg, "--kvmem-query-max-tokens")) {
            st.query_max_tokens = kvmem_cli_int(arg, need(arg));
            if (st.query_max_tokens <= 0) {
                fprintf(stderr, "invalid --kvmem-query-max-tokens (want > 0)\n");
                return 1;
            }
        } else if (eq(arg, "--kvmem-gpu-ratio")) {
            st.kparams.gpu_memory_ratio = static_cast<float>(kvmem_cli_real(arg, need(arg), 0, 1));
        } else if (eq(arg, "--kvmem-cpu-gb")) {
            const double gb = kvmem_cli_real(arg, need(arg), 0, 1048576);
            st.kparams.cpu_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (eq(arg, "--kvmem-nvme-gb")) {
            const double gb = kvmem_cli_real(arg, need(arg), 0, 1048576);
            st.kparams.nvme_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (eq(arg, "--kvmem-nvme-dir")) {
            nvme_dir = need(arg);
        } else if (eq(arg, "--kvmem-harvest-v")) {
            st.kparams.harvest_v = true;
        } else if (eq(arg, "--kvmem-raw-k-nvme")) {
            st.kparams.raw_k_nvme = true;
        } else if (eq(arg, "--kv-dtype") || eq(arg, "-ctk") || eq(arg, "--cache-type-k")
                   || eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
            bool ok = false;
            const ggml_type t = kvmem_parse_cache_type(need(arg), &ok);
            if (!ok) {
                fprintf(stderr, "unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)\n");
                return 1;
            }
            if (eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
                st.cache_type_v = t;
            } else if (eq(arg, "-ctk") || eq(arg, "--cache-type-k")) {
                st.cache_type_k = t;
            } else {
                st.cache_type_k = t;
                st.cache_type_v = t;
                config_sources["--cache-type-k"] = argument_source;
                config_sources["--cache-type-v"] = argument_source;
            }
        } else if (eq(arg, "--spec-kv-dtype")) {
            bool ok = false;
            st.spec_cache_type = kvmem_parse_cache_type(need(arg), &ok);
            if (!ok) {
                fprintf(stderr, "unsupported MTP cache type (want f16|q8_0|q5_0|q4_0|f32)\n");
                return 1;
            }
        } else if (eq(arg, "--spec-type")) {
            const char * t = need(arg);
            if (eq(t, "draft-mtp")) {
                st.spec_mtp = true;
            } else if (eq(t, "none")) {
                st.spec_mtp = false;
            } else {
                fprintf(stderr, "unsupported --spec-type %s (P7-0: draft-mtp|none)\n", t);
                return 1;
            }
        } else if (eq(arg, "--spec-draft-n-max")) {
            st.spec_n_max = kvmem_cli_int(arg, need(arg));
        } else if (eq(arg, "--spec-draft-p-min")) {
            st.spec_p_min = static_cast<float>(kvmem_cli_real(arg, need(arg), 0, 1));
        } else if (eq(arg, "--jinja")) {
            // Native Jinja rendering is always enabled in this server.
        } else if (eq(arg, "--no-jinja")) {
            throw std::invalid_argument("Jinja is required by this server; --no-jinja / LLAMA_ARG_JINJA=false is unsupported");
        } else if (eq(arg, "--chat-template") || eq(arg, "--chat-template-file")) {
            if (template_set && !(argument_source == "cli" && template_source != "cli")) {
                fprintf(stderr, "choose only one --chat-template or --chat-template-file\n");
                return 1;
            }
            template_set = true;
            template_source = argument_source;
            const std::string value = need(arg);
            if (eq(arg, "--chat-template-file")) {
                std::ifstream file(value);
                if (!file) { fprintf(stderr, "cannot read chat template: %s\n", value.c_str()); return 1; }
                chat_template.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            } else {
                chat_template = value;
            }
            if (chat_template.empty()) { fprintf(stderr, "chat template must not be empty\n"); return 1; }
        } else if (eq(arg, "--chat-template-kwargs")) {
            const auto kw = json::parse(need(arg), nullptr, false);
            if (!kw.is_object()) { fprintf(stderr, "--chat-template-kwargs requires a JSON object\n"); return 1; }
            template_defaults["chat_template_kwargs"] = kw;
        } else if (eq(arg, "--reasoning-effort")) {
            template_defaults["reasoning_effort"] = need(arg);
        } else if (eq(arg, "--enable-thinking")) {
            st.enable_thinking_default = true;
        } else if (eq(arg, "--no-think")) {
            st.enable_thinking_default = false;
        } else if (eq(arg, "--reasoning-budget")) {
            const char * value = need(arg);
            std::string err;
            const auto parsed = json::parse(value, nullptr, false);
            int budget = -1;
            if (parsed.is_discarded() || parsed.is_null() ||
                    !kvmem_chat_reasoning_budget_override({{"reasoning_budget_tokens", parsed}}, budget, err)) {
                fprintf(stderr, "invalid --reasoning-budget: %s\n", err.empty() ? "expected an integer >= -1" : err.c_str());
                return 1;
            }
            st.reasoning_budget_default = budget;
        } else if (eq(arg, "--reasoning-budget-message")) {
            st.reasoning_budget_message = need(arg);
        } else if (eq(arg, "--auto-continue-thinking")) {
            st.act.enabled = true;
        } else if (eq(arg, "--act-max-rounds")) {
            st.act.max_rounds = kvmem_cli_int(arg, need(arg), 8);
            if (st.act.max_rounds < 1) {
                fprintf(stderr, "invalid --act-max-rounds: expected an integer >= 1\n");
                return 1;
            }
        } else if (eq(arg, "--act-round-tokens")) {
            st.act.round_tokens = kvmem_cli_int(arg, need(arg), 0);
            if (st.act.round_tokens < 0) {
                fprintf(stderr, "invalid --act-round-tokens: expected 0 or a positive integer\n");
                return 1;
            }
        } else if (eq(arg, "--act-total-tokens")) {
            st.act.total_tokens = kvmem_cli_int(arg, need(arg), 0);
            if (st.act.total_tokens < 0) {
                fprintf(stderr, "invalid --act-total-tokens: expected 0 or a positive integer\n");
                return 1;
            }
        } else if (eq(arg, "--act-continue-cue")) {
            st.act.cue = need(arg);
        } else if (eq(arg, "--act-prefill-mode")) {
            st.act.prefill_mode = need(arg);
            if (st.act.prefill_mode != "auto" && st.act.prefill_mode != "legacy" &&
                    st.act.prefill_mode != "restart") {
                fprintf(stderr, "invalid --act-prefill-mode: expected auto | legacy | restart\n");
                return 1;
            }
        } else if (eq(arg, "--act-think-end")) {
            st.act.think_end = need(arg);
        } else {
            fprintf(stderr, "unknown flag: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }
    st.kparams.sink_tokens = static_cast<uint32_t>(options.sink_tokens);
    // Some pinned llama.cpp trace sites test presence rather than the value.
    // Normalize "0"/empty and CLI-off to an absent variable before loading models.
    const bool trace = options.trace == -1 ? kvmem_diag_enabled() : options.trace != 0;
#ifdef _WIN32
    if (_putenv_s("KVMEM_TRACE", trace ? "1" : "") != 0)
#else
    if ((trace ? setenv("KVMEM_TRACE", "1", 1) : unsetenv("KVMEM_TRACE")) != 0)
#endif
        throw std::runtime_error("cannot configure KVMEM_TRACE");
    kvmem_diag_set(trace);
    common_log_set_verbosity_thold(options.verbosity);
    for (const auto & input : config_inputs) {
        // Only option names and sources, never values (which may contain API keys).
        kvmem_diag("KVMEM_CONFIG input=%s source=%s\n", input.first.c_str(), input.second.c_str());
    }
    } catch (const std::exception & e) {
        fprintf(stderr, "invalid arguments (source=%s): %s\n", argument_source.c_str(), e.what());
        return 1;
    }
#if !KVMEM_ENABLE_NVME
    if (st.kparams.nvme_bytes || st.kparams.raw_k_nvme) {
        fprintf(stderr, "NVMe offload is disabled in this build (KVMEM_ENABLE_NVME=OFF)\n");
        return 1;
    }
#endif
    // Validate before backend initialization and loading a potentially large model.
    if (!kvmem_cache_types_ok(st.cache_type_k, st.cache_type_v)) {
        fprintf(stderr, "incompatible KV cache types: K=%s, V=%s; quantized K/V must both use q8_0, q5_0 or q4_0; "
                "set both -ctk and -ctv, or use --kv-dtype TYPE to set both\n",
                ggml_type_name(st.cache_type_k), ggml_type_name(st.cache_type_v));
        return 1;
    }
    if (options.list_devices) {
        common_print_available_devices();
        return 0;
    }
    if (model_path.empty()) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR missing model; set --model PATH or LLAMA_ARG_MODEL\n");
        print_usage(argv[0]);
        return 1;
    }
    {
        std::string err;
        if (!kvmem_chat_template_override(template_defaults, st.enable_thinking_default, st.template_kwargs, err)) {
            fprintf(stderr, "invalid template defaults: %s\n", err.c_str());
            return 1;
        }
    }
    {
        const auto slash = model_path.find_last_of("/\\");
        st.model_name = slash == std::string::npos ? model_path : model_path.substr(slash + 1);
    }

    if (!options.alias.empty()) st.model_name = options.alias;

    if (image_min_tokens > 0 && image_max_tokens > 0 && image_min_tokens > image_max_tokens) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR --image-min-tokens exceeds --image-max-tokens\n");
        return 1;
    }
    if (st.kparams.enabled && st.kparams.block_tokens == 0) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR --kvmem-block-tokens must be positive\n");
        return 1;
    }
    if (st.spec_mtp && st.kparams.enabled && st.kparams.mtp_state == 2 &&
            (st.spec_n_max < 1 || st.spec_n_max > 5)) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR replay MTP requires --spec-draft-n-max in 1..5\n");
        return 1;
    }

    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);
    ggml_backend_load_all();

    // No speculative rollback state is needed without MTP.
    if (!st.spec_mtp) st.kparams.mtp_state = 0;
    if (st.kparams.enabled) {
        if (!nvme_dir.empty()) {
            st.kparams.nvme_dir = nvme_dir.c_str();
        }
        llama_kvmem_set_params(&st.kparams);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    mparams.load_mtp = st.spec_mtp;
    mparams.load_mode = options.load_mode;
    try {
        device_config.apply(options, mparams);
    } catch (const std::exception & e) {
        fprintf(stderr, "invalid GPU configuration: %s\n", e.what());
        return 1;
    }
    // Check resources before spending time/VRAM on loading model weights.
    auto readable = [](const std::string & path) {
        std::error_code ec;
        return std::filesystem::is_regular_file(path, ec) && std::ifstream(path, std::ios::binary).good();
    };
    if (!readable(model_path)) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR failed to load model: --model file is missing or unreadable: %s\n", model_path.c_str());
        return 1;
    }
    if (!mmproj_path.empty() && !readable(mmproj_path)) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR --mmproj file is missing or unreadable: %s\n", mmproj_path.c_str());
        return 1;
    }
    if (!no_ui && !ui_dir.empty() && !readable((std::filesystem::path(ui_dir) / "index.html").string())) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR --ui-dir/--path must contain a readable index.html\n");
        return 1;
    }
    json startup = {
        {"model", model_path}, {"alias", st.model_name},
        {"gpu", {{"device_requested", options.device_names.empty() ? "auto" : options.device_names},
                  {"main_gpu", mparams.main_gpu}, {"split_mode", mparams.split_mode == LLAMA_SPLIT_MODE_NONE ? "none" : "layer"},
                  {"layers_requested", ngl}}},
        {"context_requested", n_ctx}, {"batch_requested", st.n_batch},
        {"n_predict", st.n_predict_default},
        {"kv", {{"k", ggml_type_name(st.cache_type_k)}, {"v", ggml_type_name(st.cache_type_v)}}},
        {"kvmem", {{"enabled", st.kparams.enabled}, {"budget", st.kparams.budget}, {"gen_reserve", st.kparams.gen_reserve},
                   {"sink_tokens", st.kparams.sink_tokens}, {"block_tokens", st.kparams.block_tokens}}},
        {"spec_type", st.spec_mtp ? "draft-mtp" : "none"},
        {"vision", {{"enabled", !mmproj_path.empty()}, {"projector", mmproj_path}, {"gpu", mmproj_gpu}}},
        {"http", {{"host", host}, {"port", port}, {"timeout", options.timeout}, {"slots", 1}}},
        {"auth", {{"enabled", !options.api_keys.empty()}, {"key_count", options.api_keys.size()}}},
        {"sources", config_sources}, {"unlisted_sources", "default"}
    };
    kvmem_diag("KVMEM_STARTUP requested=%s\n", startup.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    LOG_INF("srv    loading model %s\n", model_path.c_str());
    st.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!st.model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    st.vocab = llama_model_get_vocab(st.model);
    try {
        st.tmpls = common_chat_templates_init(st.model, chat_template);
    } catch (const std::exception & e) {
        fprintf(stderr, "invalid chat template: %s\n", e.what());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) st.n_batch;
    cparams.n_ubatch = options.ubatch > 0 ? options.ubatch : st.n_batch;
    if (options.threads > 0) cparams.n_threads = options.threads;
    if (options.threads_batch > 0) cparams.n_threads_batch = options.threads_batch;
    else if (options.threads > 0) cparams.n_threads_batch = options.threads;
    if (options.flash_attn_set) cparams.flash_attn_type = options.flash_attn;
    cparams.n_seq_max = 1;
    cparams.type_k = st.cache_type_k;
    cparams.type_v = st.cache_type_v;
    if (st.spec_mtp) {
        const uint32_t n_out = (uint32_t) (1 + std::max(0, st.spec_n_max));
        cparams.n_outputs_max = n_out;
        cparams.n_outputs_max_per_seq = n_out;
        cparams.n_rs_seq = (uint32_t) std::max(0, st.spec_n_max);
    }
    st.ctx = llama_init_from_model(st.model, cparams);
    if (!st.ctx) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR failed to create context; check --ctx-size, KV types, --flash-attn and available memory\n");
        return 1;
    }
    kvmem_diag("KVMEM_CONTEXT target threads=%d threads_batch=%d ubatch=%u flash_attn_requested=%s\n",
            llama_n_threads(st.ctx), llama_n_threads_batch(st.ctx), llama_n_ubatch(st.ctx),
            llama_flash_attn_type_name(cparams.flash_attn_type));
    if (st.spec_mtp) {
        kvmem_spec_opts sopts;
        sopts.n_max = st.spec_n_max;
        sopts.p_min = st.spec_p_min;
        sopts.n_gpu_layers = ngl;
        sopts.n_ctx = n_ctx;
        sopts.n_batch = st.n_batch;
        sopts.n_ubatch = cparams.n_ubatch;
        sopts.n_threads = options.threads;
        sopts.n_threads_batch = options.threads_batch > 0 ? options.threads_batch : options.threads;
        if (options.flash_attn_set) sopts.flash_attn = options.flash_attn;
        sopts.kvmem_enabled = st.kparams.enabled;
        sopts.type_k = st.cache_type_k;
        sopts.type_v = st.cache_type_v;
        sopts.draft_type = st.spec_cache_type;
        if (!kvmem_spec_start(st.spec, st.model, st.ctx, sopts)) {
            return 1;
        }
    }

    if (!mmproj_path.empty()) {
        try {
            if (image_min_tokens > 0 && image_max_tokens > 0 && image_min_tokens > image_max_tokens)
                throw std::invalid_argument("image-min-tokens exceeds image-max-tokens");
            st.vision = std::make_unique<kvmem_vision>(st.model, mmproj_path, mmproj_gpu, image_min_tokens, image_max_tokens);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }

    httplib::Server svr;
    svr.set_read_timeout(options.timeout, 0);
    svr.set_write_timeout(options.timeout, 0);
    if (options.threads_http_set) {
        const int n = options.threads_http > 0 ? options.threads_http :
            std::max(5, static_cast<int>(std::thread::hardware_concurrency()) - 1);
        svr.new_task_queue = [n] { return new httplib::ThreadPool(n, static_cast<size_t>(n) + 1024); };
        kvmem_diag("KVMEM_HTTP threads=%d timeout=%d\n", n, options.timeout);
    }
    svr.set_idle_interval(0, 100000);
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
    });
    svr.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    std::unordered_set<std::string> ui_paths;
    if (!kvmem_mount_ui(svr, ui_dir, no_ui, argv[0], &ui_paths)) return 1;
    kvmem_install_auth(svr, options.api_keys, std::move(ui_paths));
    const int generation_limit = st.kparams.enabled && st.kparams.gen_reserve > 0 ?
        std::min(n_ctx, (int) st.kparams.gen_reserve) : n_ctx;
    startup["context_actual"] = llama_n_ctx(st.ctx);
    startup["batch_actual"] = llama_n_batch(st.ctx);
    startup["ubatch_actual"] = llama_n_ubatch(st.ctx);
    startup["threads_actual"] = llama_n_threads(st.ctx);
    startup["threads_batch_actual"] = llama_n_threads_batch(st.ctx);
    startup["flash_attn_requested"] = llama_flash_attn_type_name(cparams.flash_attn_type);
    startup["generation_limit"] = generation_limit;
    startup["default_max_tokens"] = std::min(st.n_predict_default > 0 ? st.n_predict_default : generation_limit, generation_limit);
    startup["http"]["threads"] = options.threads_http_set ?
        (options.threads_http > 0 ? options.threads_http : std::max(5, static_cast<int>(std::thread::hardware_concurrency()) - 1)) :
        static_cast<int>(CPPHTTPLIB_THREAD_POOL_COUNT);
    if (!config_sources.contains("--threads-batch") && options.threads > 0)
        startup["sources"]["--threads-batch"] = "inherited:--threads";
    if (!config_sources.contains("--ubatch-size")) startup["sources"]["--ubatch-size"] = "inherited:--batch-size";
    json kwargs = json::object();
    for (const auto & item : st.template_kwargs) kwargs[item.first] = json::parse(item.second);
    const auto thinking_params = kvmem_ui_sampling(true, st.sampling_overrides);
    const auto plain_params = kvmem_ui_sampling(false, st.sampling_overrides);
    auto default_params = st.enable_thinking_default ? thinking_params : plain_params;
    default_params["n_predict"] = std::min(st.n_predict_default > 0 ? st.n_predict_default : generation_limit, generation_limit);
    default_params["max_tokens"] = default_params["n_predict"];
    // 1:1 upstream /props extras (server-context.cpp get_res_props).
    // 中文：除 kvmem 扩展字段外，补齐上游 /props 的标准字段（bos/eos token、模板能力等）
    std::string bos_token_str, eos_token_str;
    if (st.vocab != nullptr) {
        const llama_token bos_id = llama_vocab_bos(st.vocab);
        const llama_token eos_id = llama_vocab_eos(st.vocab);
        if (bos_id >= 0) bos_token_str = token_piece(st.vocab, bos_id);
        if (eos_id >= 0) eos_token_str = token_piece(st.vocab, eos_id);
    }
    json chat_template_caps = json::object();
    if (st.tmpls) {
        for (const auto & cap : common_chat_templates_get_caps(st.tmpls.get())) {
            chat_template_caps[cap.first] = cap.second;
        }
    }
    const json props = {
        // upstream get_res_props fields
        // 中文：上游 /props 返回的标准字段，UI 依赖这些键渲染模型信息
        {"default_generation_settings", {{"params", default_params}, {"n_ctx", n_ctx}}},
        {"total_slots", 1},
        {"model_alias", st.model_name},
        // kvmem's llama.cpp predates llama_model_ftype_name(); keep the field for UI parity.
        // 中文：当前 kvmem 的 llama.cpp 尚无 llama_model_ftype_name()，保留空字段以兼容上游 UI 展示
        {"model_ftype", ""},
        {"model_path", model_path},
        {"modalities", {{"vision", st.vision != nullptr}, {"audio", false}, {"video", false}}},
        {"media_marker", mtmd_default_marker()},
        {"endpoint_slots", true}, {"endpoint_props", false}, {"endpoint_metrics", false},
        {"ui", !no_ui},
        {"ui_settings", json::object()},
        {"chat_template", common_chat_templates_source(st.tmpls.get())},
        {"chat_template_caps", chat_template_caps},
        {"bos_token", bos_token_str},
        {"eos_token", eos_token_str},
        {"build_info", std::string(llama_build_info())},
        {"is_sleeping", false},
        {"cors_proxy_enabled", false},
        // kvmem extensions (kept for existing clients)
        // 中文：kvmem 扩展字段，保留以兼容既有客户端
        {"role", "model"}, {"model_name", st.model_name},
        {"kvmem", {{"generation_limit", generation_limit},
            {"defaults", {{"enable_thinking", st.enable_thinking_default},
                          {"reasoning_budget_tokens", st.reasoning_budget_default}, {"chat_template_kwargs", kwargs}}},
            {"sampling", {{"thinking", thinking_params}, {"non_thinking", plain_params}}}}}
    };
    svr.Get("/props", [props](const httplib::Request &, httplib::Response & res) {
        res.set_header("Cache-Control", "no-store");
        res.set_content(props.dump(), "application/json");
    });
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    // A short-lived status lock keeps polling independent of inference.
    svr.Get("/slots", [&](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Cache-Control", "no-store");
        const auto progress = st.progress.snapshot();
        const bool busy = progress.busy;
        json slot = {
            {"id", 0},
            {"n_ctx", n_ctx},
            {"speculative", st.spec.ok},
            {"is_processing", busy},
            {"id_task", progress.task},
            {"n_prompt_tokens", progress.prompt},
            {"n_prompt_tokens_processed", progress.processed},
            {"n_prompt_tokens_cache", progress.cached},
            {"params", progress.params.empty() ? default_params : progress.params},
            {"next_token", json::array({
                {
                    {"has_next_token", busy},
                    {"has_new_line", false},
                    {"n_remain", busy ? std::max(0, progress.limit - progress.generated) : 0},
                    {"n_decoded", progress.generated},
                }
            })},
            {"prompt", ""},
            {"generated", ""},
        };
        if (busy && req.has_param("fail_on_no_slot")) {
            res.status = 503;
            res.set_content(json{{"error", json{
                {"code", 503},
                {"message", "no slot available"},
                {"type", "unavailable_error"},
            }}}.dump(), "application/json");
            return;
        }
        res.set_content(json::array({slot}).dump(), "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {
            {"object", "list"},
            {"data", json::array({json{{"id", st.model_name}, {"name", st.model_name}, {"object", "model"}, {"status", {{"value", "loaded"}}}}})},
        };
        res.set_content(j.dump(), "application/json");
    });

    auto handle_chat = [&](const httplib::Request & req, httplib::Response & res) {
        const std::time_t created = std::time(nullptr);
        json body;
        std::vector<std::vector<uint8_t>> media_files;
        try {
            body = json::parse(kvmem_parse_media_messages(req.body, st.vision != nullptr, media_files));
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        ChatRequest cr;
        cr.max_tokens = st.n_predict_default;
        cr.enable_thinking = st.enable_thinking_default;
        cr.template_kwargs = st.template_kwargs;
        cr.reasoning_budget_tokens = st.reasoning_budget_default;
        cr.reasoning_budget_message = st.reasoning_budget_message;
        std::string err;
        if (!parse_chat_request(body, cr, err) ||
            !kvmem_output_limit(body, generation_limit, cr.max_tokens, err)) {
            res.status = 400;
            res.set_content(json{{"error", err}}.dump(), "application/json");
            return;
        }
        if (body.contains("cache_reset") && !body["cache_reset"].is_boolean()) {
            res.status = 400;
            res.set_content("{\"error\":\"cache_reset must be a boolean\"}", "application/json");
            return;
        }

        cr.sampling = kvmem_chat_sampling_defaults(cr.enable_thinking);
        if (!kvmem_chat_sampling_override(st.sampling_overrides, cr.sampling, err) ||
            !kvmem_chat_sampling_override(body, cr.sampling, err)) {
            res.status = 400;
            res.set_content(json{{"error", err}}.dump(), "application/json");
            return;
        }

        auto slot = std::make_shared<kvmem_server_slot_guard>(st.mu, st.progress);
        if (req.is_connection_closed && req.is_connection_closed()) return;
        st.mm_reset_requested = body.value("cache_reset", false);

        common_chat_templates_inputs inputs;
        inputs.messages = cr.msgs;
        inputs.tools = cr.tools;
        inputs.tool_choice = cr.tool_choice;
        inputs.grammar = cr.grammar;
        inputs.json_schema = cr.json_schema;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.enable_thinking = cr.enable_thinking;
        inputs.chat_template_kwargs = cr.template_kwargs;
        // llama-server default: extract <think> into delta.reasoning_content.
        // NONE leaves thinking in content, so OpenCode TUI never classifies it.
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (cr.parallel_tool_calls_set) {
            inputs.parallel_tool_calls = cr.parallel_tool_calls;
        } else {
            const auto caps = common_chat_templates_get_caps(st.tmpls.get());
            const auto it = caps.find("supports_parallel_tool_calls");
            inputs.parallel_tool_calls = it != caps.end() && it->second;
        }
        common_chat_params formatted;
        try {
            formatted = common_chat_templates_apply(st.tmpls.get(), inputs);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("chat template: ") + e.what()}}.dump(), "application/json");
            return;
        }
        const std::string & prompt = formatted.prompt;
        std::shared_ptr<kvmem_prompt> parsed_prompt;
        try {
            parsed_prompt = media_files.empty()
                ? std::make_shared<kvmem_prompt>(tokenize_text(st.vocab, prompt, true))
                : st.vision->tokenize(prompt, media_files);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        st.active_prompt = parsed_prompt;
        st.turn_generation_rows = (uint32_t) std::min<uint64_t>(UINT32_MAX,
                (uint64_t) std::max(0, cr.max_tokens) + (st.spec.ok ? std::max(0, st.spec_n_max) + 1u : 0u));
        auto toks = parsed_prompt->tokens;
        if (toks.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"empty prompt\"}", "application/json");
            return;
        }
        if ((int) toks.size() + cr.max_tokens > (int) llama_n_ctx(st.ctx)) {
            res.status = 400;
            res.set_content("{\"error\":\"prompt + max_tokens exceeds n_ctx\"}", "application/json");
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
            st.turn_query_exact = cr.query_begin >= 0 && cr.query_end > cr.query_begin && cr.query_end <= (int) toks.size();
            if (cr.query_begin < 0 && cr.query_end < 0) {
                st.turn_query_exact = derive_native_query_span(st, prompt, inputs, *parsed_prompt, qbegin, qend);
            }
        }
        if (parsed_prompt->has_media() && cr.query_begin < 0 && !st.turn_query_exact) {
            // The final text question follows native visual chunks and their boundaries.
            int last_media_end = 0;
            for (const auto & range : parsed_prompt->media_ranges()) last_media_end = range.second;
            qend = (int) toks.size() - (st.spec.ok ? 1 : 0);
            qbegin = std::max(last_media_end, qend - st.query_max_tokens);
        }
        clamp_query_span(st, qbegin, qend);
        if (st.turn_query_exact && std::find(toks.begin() + qbegin, toks.begin() + qend, LLAMA_TOKEN_NULL) != toks.begin() + qend) {
            st.turn_query_exact = false;
            kvmem_diag("KVMEM_TRACE query_loc fallback=explicit_span_contains_media\n");
        }
        try {
            multimodal_validate_capacity(st, *parsed_prompt, st.query_policy_user ? (int) toks.size() : qbegin, (int) toks.size());
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
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
        for (const auto & m : cr.msgs) {
            if (m.role == "tool" || !m.tool_calls.empty()) {
                n_tool_hist++;
            }
        }
        bool prompt_has_tool = false;
        for (const auto & t : cr.tools) {
            if (!t.name.empty() && prompt.find(t.name) != std::string::npos) {
                prompt_has_tool = true;
                break;
            }
        }
        kvmem_diag("KVMEM_TRACE n_prompt=%d query=[%d,%d) force_pos=%d last_user_chars=%zu\n",
                (int) toks.size(), qbegin, qend, force, cr.last_user.size());
        kvmem_diag("KVMEM_TRACE chat_parse n_msg=%zu n_tools=%zu tool_choice=%s tool_hist=%d "
                "prompt_has_tool=%d grammar_bytes=%zu think=%d reasoning=%s parser_bytes=%zu\n",
                cr.msgs.size(), cr.tools.size(), tool_choice_cstr(cr.tool_choice),
                n_tool_hist, (int) prompt_has_tool, formatted.grammar.size(),
                (int) cr.enable_thinking,
                common_reasoning_format_name(inputs.reasoning_format),
                formatted.parser.size());

        const std::string request_id = kvmem_chat_request_id();
        const std::string cid = "chatcmpl-" + request_id;
        llama_context * ctx = st.ctx;
        const llama_vocab * vocab = st.vocab;

        common_params_sampling sparams = make_chat_sampling(vocab, formatted, cr);
        if (!kvmem_chat_reasoning_budget_supported(sparams, cr.enable_thinking, err)) {
            res.status = 400;
            res.set_content(json{{"error", err}}.dump(), "application/json");
            return;
        }
        kvmem_diag("KVMEM_TRACE sampling thinking=%d temperature=%.6g top_p=%.6g top_k=%d min_p=%.6g "
                "presence_penalty=%.6g frequency_penalty=%.6g repetition_penalty=%.6g seed=%u\n",
                (int) cr.enable_thinking, sparams.temp, sparams.top_p, sparams.top_k, sparams.min_p,
                sparams.penalty_present, sparams.penalty_freq, sparams.penalty_repeat, sparams.seed);
        std::vector<std::string> stops = cr.stop;
        stops.insert(stops.end(), formatted.additional_stops.begin(), formatted.additional_stops.end());
        kvmem_diag("KVMEM_TRACE chat_sample grammar_type=%s lazy=%d n_trig=%zu gen_prompt_bytes=%zu "
                "think_start_bytes=%zu think_end_n=%zu rbudget=%d start_toks=%zu end_seqs=%zu forced_toks=%zu\n",
                grammar_type_cstr(sparams.grammar.type), (int) sparams.grammar_lazy,
                sparams.grammar_triggers.size(), sparams.generation_prompt.size(),
                formatted.thinking_start_tag.size(), formatted.thinking_end_tags.size(),
                sparams.reasoning_budget_tokens,
                sparams.reasoning_budget_start.size(),
                sparams.reasoning_budget_end.size(),
                sparams.reasoning_budget_forced.size());
        const bool parse_tools = !cr.tools.empty() &&
                cr.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

        bool use_spec = st.spec.ok;
        if (use_spec && !sparams.grammar.empty()) {
            try {
                common_params_sampling probe = sparams;
                common_sampler_ptr test(common_sampler_init(st.model, probe));
                if (!test) {
                    use_spec = false;
                }
            } catch (const std::exception & e) {
                fprintf(stderr, "KVMEM_TRACE spec sampler init failed (%s); greedy fallback\n", e.what());
                use_spec = false;
            }
        }

        if ((st.vision || st.query_policy_user) && st.spec.ok && !use_spec) {
            res.status = 400;
            res.set_content("{\"error\":\"MTP sampler could not initialize for this request\"}", "application/json");
            return;
        }

        json request_params = default_params;
        request_params["n_predict"] = cr.max_tokens;
        request_params["max_tokens"] = cr.max_tokens;
        request_params["temperature"] = sparams.temp;
        st.progress.prompt((int) toks.size(), cr.max_tokens, std::move(request_params));
        st.log.start();
        LOG_INF("slot   processing task, n_prompt = %d, n_predict = %d\n", (int) toks.size(), cr.max_tokens);
        auto timings = std::make_shared<json>(json::object());
        // 1:1 upstream timings block (server-context.cpp): prompt/predicted counts, ms, per-token and per-second rates.
        // 中文：对齐上游的 timings 统计块——prompt/predicted 的计数、耗时、每 token 与每秒速率，供 /v1 响应回传
        auto make_emit_gen_wall = [timings, &st, n_prompt = (int) toks.size()](
                std::chrono::steady_clock::time_point t_turn0,
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
                const double prompt_per_second = prefill_ms > 0.0 ? 1000.0 * (double) prompt_n / prefill_ms : 0.0;
                const double predicted_per_second = gen_ms > 0.0 ? 1000.0 * (double) n_gen / gen_ms : 0.0;
                *timings = {
                    {"prompt_n", prompt_n},
                    {"prompt_ms", prefill_ms},
                    {"prompt_per_token_ms", prompt_n > 0 ? prefill_ms / (double) prompt_n : 0.0},
                    {"prompt_per_second", prompt_per_second},
                    {"predicted_n", n_gen},
                    {"predicted_ms", gen_ms},
                    {"predicted_per_token_ms", n_gen > 0 ? gen_ms / (double) n_gen : 0.0},
                    {"predicted_per_second", predicted_per_second},
                    {"cache_n", cache_n}
                };
                if (!verbose) {
                    return;
                }
                LOG_INF("slot   prompt eval time = %10.2f ms / %5d tokens (%8.2f tokens per second), cache = %d\n",
                        prefill_ms, prompt_n, prompt_per_second, cache_n);
                LOG_INF("slot          eval time = %10.2f ms / %5d tokens (%8.2f tokens per second)\n",
                        gen_ms, n_gen, predicted_per_second);
                LOG_INF("slot         total time = %10.2f ms / %5d tokens\n", prefill_ms + gen_ms, prompt_n + n_gen);
                const double wall_ms = std::chrono::duration<double, std::milli>(now - t_turn0).count();
                const double tps = gen_ms > 0.0 ? 1000.0 * (double) n_gen / gen_ms : 0.0;
                kvmem_diag("KVMEM_GEN_WALL n=%d ms=%.2f toks=%.2f\n", n_gen, gen_ms, tps);
                kvmem_diag("KVMEM_CHAT_TURN n_prompt=%d n_gen=%d prefill_ms=%.2f gen_ms=%.2f "
                        "wall_ms=%.2f gen_toks=%.2f\n",
                        n_prompt, n_gen, prefill_ms, gen_ms, wall_ms, tps);
            };
        };

        int n_cache_hit = 0;
        auto emit_json = [&](const std::string & content, int n_gen, bool hit_limit) {
            common_chat_msg msg = parse_assistant_output(content, formatted, parse_tools);
            std::vector<std::string> tc_ids;
            int n_id = 0;
            msg.set_tool_call_ids(tc_ids, [&n_id, &request_id]() {
                return kvmem_chat_tool_id(request_id, ++n_id);
            });
            std::string finish = "stop";
            if (!msg.tool_calls.empty()) {
                finish = "tool_calls";
            } else if (hit_limit) {
                finish = "length";
            }
            json message;
            try {
                message = message_to_nlohmann(msg);
            } catch (const std::exception &) {
                message = json{{"role", "assistant"}, {"content", content}};
            }
            kvmem_diag("KVMEM_TRACE chat_out n_tool_calls=%zu finish=%s content_chars=%zu reasoning_chars=%zu\n",
                    msg.tool_calls.size(), finish.c_str(),
                    msg.content.size(), msg.reasoning_content.size());
            json out = {
                {"id", cid},
                {"object", "chat.completion"},
                {"created", created},
                {"model", st.model_name},
                {"system_fingerprint", std::string(llama_build_info())},
                {"choices", json::array({json{
                    {"index", 0},
                    {"message", message},
                    {"finish_reason", finish},
                }})},
                {"usage", usage_json((int) toks.size(), n_gen, n_cache_hit)},
            };
            out["timings"] = *timings;
            res.set_content(out.dump(), "application/json");
        };

        // ACT tags follow the rendered template, not a hard-coded constant.
        // The request may switch auto-continue off for A/B comparisons.
        const bool act_on = (cr.act_override >= 0) ? (cr.act_override == 1) : st.act.enabled;
        st.act.thinking_on = cr.enable_thinking;
        if (!formatted.thinking_end_tags.empty()) {
            st.act.think_end = formatted.thinking_end_tags.front();
        }
        if (!formatted.thinking_start_tag.empty()) {
            st.act.think_start = formatted.thinking_start_tag;
        }
        // The request flag alone is not the whole story. A client may steer
        // thinking only through chat_template_kwargs, and the server-side
        // --enable-thinking default also opens a thinking block in the rendered
        // prompt. If the prompt ends inside an OPEN thinking block the model will
        // reason no matter what the flag said, so the continuation loop has to
        // stay armed. Otherwise a round stops at --act-round-tokens and closes
        // the block, which looks exactly like "thinking gives up by itself as
        // soon as the round limit is reached".
        if (!st.act.thinking_on && !st.act.think_start.empty()) {
            const auto last_open = formatted.prompt.rfind(st.act.think_start);
            if (last_open != std::string::npos) {
                size_t last_close = std::string::npos;
                for (const auto & tag : formatted.thinking_end_tags) {
                    if (tag.empty()) {
                        continue;
                    }
                    const auto pos = formatted.prompt.rfind(tag);
                    if (pos != std::string::npos && (last_close == std::string::npos || pos > last_close)) {
                        last_close = pos;
                    }
                }
                if (last_close == std::string::npos || last_close < last_open) {
                    st.act.thinking_on = true;
                }
            }
        }
        // One line that explains, without guessing, why the continuation loop
        // will or will not run for this request.
        kvmem_diag("KVMEM_ACT_STATE act_on=%d thinking_on=%d max_rounds=%d round_tokens=%d total_tokens=%d\n",
                (int) act_on, (int) st.act.thinking_on, st.act.max_rounds,
                st.act.round_tokens, st.act.total_tokens);

        if (cr.stream) {
            const int max_tokens = cr.max_tokens;
            const bool spec_stream = use_spec;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider("text/event-stream",
                [slot, &st, &req, toks, cid, request_id, created, max_tokens, sparams, parse_tools, formatted, stops,
                 spec_stream, ctx, vocab, make_emit_gen_wall, timings, act_on](size_t, httplib::DataSink & sink) mutable {
                    StreamIo io;
                    io.sink = &sink;
                    io.req = &req;
                    auto send = [&](const std::string & payload) -> bool {
                        const std::string line = "data: " + payload + "\n\n";
                        if (!sink.write(line.data(), line.size())) {
                            io.aborted = true;
                            return false;
                        }
                        return true;
                    };
                    // 1:1 upstream initial delta: {role, content:null} (server-task.cpp to_json_oaicompat_chat)
                    // 中文：对齐上游流式首帧 delta——role=assistant 且 content=null，客户端按此初始化
                    send(stream_choice_chunk(cid, st.model_name, created, json{{"role", "assistant"}, {"content", nullptr}}, nullptr).dump());
                    const auto t_turn0 = std::chrono::steady_clock::now();
                    int n_cache_hit = 0;
                    if (!run_prefill_retrieval(st, toks, &io, &n_cache_hit)) {
                        if (!io.aborted) {
                            send(json{{"error", st.mm_error.empty() ? "prefill/retrieval failed" : st.mm_error}}.dump());
                            sink.write("data: [DONE]\n\n", 14);
                        }
                        multimodal_finish_request(st);
                        slot->unlock();
                        sink.done();
                        return true;
                    }
                    const auto t_pf1 = std::chrono::steady_clock::now();
                    const double prefill_ms =
                            std::chrono::duration<double, std::milli>(t_pf1 - t_turn0).count();
                    kvmem_diag("KVMEM_CHAT_PREFILL ms=%.2f n_prompt=%d\n",
                            prefill_ms, (int) toks.size());
                    llama_kvmem_end_prefill_capture();
                    st.mm_live_checkpoint.reset();
                    auto emit_gen_wall = make_emit_gen_wall(t_turn0, t_pf1, prefill_ms, n_cache_hit);

                    // ---- ACT: multi-round thinking loop --------------------------------
                    // A single generation is capped by --kvmem-gen-reserve. When thinking
                    // hits the limit, commit the chain into the context and open the next
                    // round instead of cutting the answer short.
                    std::vector<llama_token> cur      = toks;  // prompt of the current round
                    std::vector<llama_token> gen;              // tokens produced this round
                    std::vector<llama_token> all_gen;          // tokens produced by all rounds
                    std::string content;                       // full text across all rounds
                    StreamChatOut sco(formatted, parse_tools, request_id);
                    bool aborted   = false;
                    bool hit_limit = false;
                    bool fatal     = false;
                    int  act_round = 0;

                    for (;;) {
                        gen.clear();
                        const int budget = act_round_budget(st.act, act_on, max_tokens);
                        bool round_eos   = false;
                        bool round_stop  = false;
                        bool round_limit = false;
                        if (spec_stream) {
                            const auto gst = kvmem_spec_generate(st.ctx, st.model, st.spec, cur, budget, sparams,
                            [&](llama_token id, const std::string & piece, bool) {
                                gen.push_back(id);
                                content += piece;
                                emit_gen_wall((int) all_gen.size() + (int) gen.size(), false);
                                auto deltas = sco.set_text(content, true);
                                for (size_t i = 0; i < deltas.size(); ++i) {
                                    const json * ts = (i + 1 == deltas.size()) ? &*timings : nullptr;
                                    send(stream_choice_chunk(cid, st.model_name, created, deltas[i], nullptr, ts).dump());
                                }
                            },
                            [&]() { return !stream_heartbeat(&io); },
                            st.active_prompt->model_pos(cur.size()) - (llama_pos) cur.size());
                            st.mm_live_row = gst.n_past;
                            if (io.aborted) {
                                aborted = true;
                                break;
                            }
                            if (gst.failed && gen.empty()) {
                                // nothing produced: a real failure, not a truncated generation
                                send(json{{"error", "speculative decode failed"}}.dump());
                                fatal = true;
                                break;
                            }
                            // failed with tokens in hand == the KVMem generation reserve was
                            // exhausted; treat it exactly like hitting the output limit.
                            round_limit = gst.failed || (int) gen.size() >= budget;
                        } else {
                        common_sampler * smpl = nullptr;
                        try {
                            common_params_sampling sp = sparams;
                            smpl = common_sampler_init(st.model, sp);
                        } catch (const std::exception & e) {
                            fprintf(stderr, "sampler init failed: %s\n", e.what());
                            send(json{{"error", std::string("sampler init failed: ") + e.what()}}.dump());
                            sink.write("data: [DONE]\n\n", 14);
                            multimodal_finish_request(st);
                            slot->unlock();
                            sink.done();
                            return true;
                        }
                        if (!smpl) {
                            send(json{{"error", "sampler init failed"}}.dump());
                            sink.write("data: [DONE]\n\n", 14);
                            multimodal_finish_request(st);
                            slot->unlock();
                            sink.done();
                            return true;
                        }
                            bool stopped = false;
                            bool hit_stop = false;
                            while ((int) gen.size() < budget && !stopped) {
                                if (!stream_heartbeat(&io)) {
                                    aborted = true;
                                    break;
                                }
                                llama_token id = common_sampler_sample(smpl, ctx, -1);
                                common_sampler_accept(smpl, id, true);
                                if (llama_vocab_is_eog(vocab, id)) {
                                    stopped = true;
                                    break;
                                }
                                std::string piece = token_piece(vocab, id);
                                if (multimodal_decode_generated(st, id, (int) cur.size() + (int) gen.size()) != 0) {
                                    fprintf(stderr, "llama_decode(gen) failed\n");
                                    send(json{{"error", "decode failed"}}.dump());
                                    fatal = true;
                                    break;
                                }
                                content += piece;
                                gen.push_back(id);
                                hit_stop = strip_stop(content, stops);
                                emit_gen_wall((int) all_gen.size() + (int) gen.size(), false);
                                auto deltas = sco.set_text(content, !hit_stop);
                                for (size_t i = 0; i < deltas.size(); ++i) {
                                    const json * ts = (i + 1 == deltas.size()) ? &*timings : nullptr;
                                    send(stream_choice_chunk(cid, st.model_name, created, deltas[i], nullptr, ts).dump());
                                }
                                if (hit_stop) {
                                    break;
                                }
                            }
                            common_sampler_free(smpl);
                            if (aborted || fatal) {
                                break;
                            }
                            round_eos   = stopped;
                            round_stop  = hit_stop;
                            round_limit = !stopped && !hit_stop && (int) gen.size() >= budget;
                        }

                        all_gen.insert(all_gen.end(), gen.begin(), gen.end());
                        if (aborted || fatal) {
                            break;
                        }

                        // keep going while the model has not finished: it may still
                        // be reasoning, or it may be writing the final answer.
                        // bounded by --act-max-rounds and --act-total-tokens
                        const bool want_more = act_on && st.act.thinking_on && round_limit
                                && !round_eos && !round_stop
                                && (act_round + 1) < st.act.max_rounds
                                && (st.act.total_tokens <= 0 || (int) all_gen.size() < st.act.total_tokens);
                        if (!want_more) {
                            hit_limit = round_limit;
                            break;
                        }

                        // ---- the whole chain stays in context; start the next round ----
                        commit_cached(st, cur, gen);
                        cur.insert(cur.end(), gen.begin(), gen.end());
                        if (!st.act.cue.empty()) {
                            const auto cue_ids = common_tokenize(vocab, st.act.cue, false, true);
                            cur.insert(cur.end(), cue_ids.begin(), cue_ids.end());
                        }
                        act_prepare_next_round(st, vocab, gen, budget);
                        kvmem_diag("KVMEM_ACT_ROUND k=%d n_gen=%d total=%d in_think=1\n",
                                act_round, (int) gen.size(), (int) all_gen.size());
                        act_round++;
                        int round_cache_hit = 0;
                        bool ok_prefill;
                        {
                            const bool saved_policy = st.query_policy_user;
                            if (st.act.prefill_mode == "legacy")  { st.query_policy_user = false; }
                            if (st.act.prefill_mode == "restart") { memory_clear_all(st); }
                            ok_prefill = run_prefill_retrieval(st, cur, &io, &round_cache_hit);
                            st.query_policy_user = saved_policy;
                        }
                        if (!ok_prefill) {
                            if (!io.aborted) {
                                send(json{{"error", "continuation prefill failed"}}.dump());
                            }
                            aborted = true;
                            break;
                        }
                        n_cache_hit += round_cache_hit;
                        llama_kvmem_end_prefill_capture();
                        st.mm_live_checkpoint.reset();
                    }

                    if (aborted || fatal) {
                        if (fatal) {
                            sink.write("data: [DONE]\n\n", 14);
                        }
                        multimodal_finish_request(st);
                        slot->unlock();
                        sink.done();
                        return true;
                    }
                    // stopped with an open thinking block (round cap / budget cap):
                    // close it so the client renders it as reasoning instead of content
                    if (hit_limit && act_close_thinking(content, st.act)) {
                        auto close_deltas = sco.set_text(content, false);
                        for (size_t i = 0; i < close_deltas.size(); ++i) {
                            const json * ts = (i + 1 == close_deltas.size()) ? &*timings : nullptr;
                            send(stream_choice_chunk(cid, st.model_name, created, close_deltas[i], nullptr, ts).dump());
                        }
                    }
                    emit_gen_wall((int) all_gen.size(), false);
                    auto flush_deltas = sco.set_text(content, false);
                    for (size_t i = 0; i < flush_deltas.size(); ++i) {
                        const json * ts = (i + 1 == flush_deltas.size()) ? &*timings : nullptr;
                        send(stream_choice_chunk(cid, st.model_name, created, flush_deltas[i], nullptr, ts).dump());
                    }
                    const char * finish = sco.finish_reason(hit_limit);
                    kvmem_diag("KVMEM_TRACE chat_stream n_tc_delta=%d finish=%s "
                            "content_chars=%zu reasoning_chars=%zu act_rounds=%d\n",
                            sco.n_tc_delta, finish,
                            sco.prev.content.size(), sco.prev.reasoning_content.size(), act_round);
                    llama_kvmem_decode_mean_flush();
                    emit_gen_wall((int) all_gen.size());
                    commit_cached(st, cur, gen);
                    send(stream_choice_chunk(cid, st.model_name, created, json::object(), finish).dump());
                    auto usage = stream_usage_chunk(cid, st.model_name, created, (int) toks.size(), (int) all_gen.size(), n_cache_hit);
                    usage["timings"] = *timings;
                    send(usage.dump());
                    sink.write("data: [DONE]\n\n", 14);
                    multimodal_finish_request(st);
                    slot->unlock();
                    sink.done();
                    return true;
                });
            return;
        }

        struct request_guard {
            ServerState & st;
            ~request_guard() { multimodal_finish_request(st); }
        } guard {st};
        StreamIo io;
        io.req = &req;
        const auto t_turn0 = std::chrono::steady_clock::now();
        if (!run_prefill_retrieval(st, toks, &io, &n_cache_hit)) {
            if (io.aborted) {
                kvmem_diag("KVMEM_TRACE stream_abort phase=prefill n_prompt=%d\n",
                        (int) toks.size());
                return;
            }
            res.status = (st.vision || st.query_policy_user) ? st.mm_error_status : 500;
            res.set_content(json{{"error", st.mm_error.empty() ? "prefill/retrieval failed" : st.mm_error}}.dump(), "application/json");
            return;
        }
        const auto t_pf1 = std::chrono::steady_clock::now();
        const double prefill_ms =
                std::chrono::duration<double, std::milli>(t_pf1 - t_turn0).count();
        kvmem_diag("KVMEM_CHAT_PREFILL ms=%.2f n_prompt=%d\n",
                prefill_ms, (int) toks.size());
        llama_kvmem_end_prefill_capture();
        st.mm_live_checkpoint.reset();
        auto emit_gen_wall = make_emit_gen_wall(t_turn0, t_pf1, prefill_ms, n_cache_hit);

        // ---- ACT: multi-round thinking loop (non-streaming) ----------------------
        std::vector<llama_token> cur      = toks;
        std::vector<llama_token> gen;
        std::vector<llama_token> all_gen;
        std::string content;
        bool hit_limit = false;
        bool fatal     = false;
        int  act_round = 0;

        for (;;) {
            gen.clear();
            const int budget = act_round_budget(st.act, act_on, cr.max_tokens);
            bool round_eos   = false;
            bool round_stop  = false;
            bool round_limit = false;
            if (use_spec) {
                const kvmem_spec_gen_stats gst = kvmem_spec_generate(
                        ctx, st.model, st.spec, cur, budget, sparams,
                        [&](llama_token id, const std::string & piece, bool) {
                            gen.push_back(id);
                            content += piece;
                            st.progress.generated((int) all_gen.size() + (int) gen.size());
                            st.log.generated((int) all_gen.size() + (int) gen.size());
                        }, [&]() { return !stream_heartbeat(&io); },
                        st.active_prompt->model_pos(cur.size()) - (llama_pos) cur.size());
                st.mm_live_row = gst.n_past;
                if (io.aborted) return;
                if (gst.failed && gen.empty()) {
                    res.status = 500;
                    res.set_content("{\"error\":\"speculative decode failed\"}", "application/json");
                    return;
                }
                round_limit = gst.failed || (int) gen.size() >= budget;
            } else {
                common_sampler * smpl = nullptr;
                try {
                    common_params_sampling sp = sparams;
                    smpl = common_sampler_init(st.model, sp);
                } catch (const std::exception & e) {
                    res.status = 500;
                    res.set_content(json{{"error", std::string("sampler init failed: ") + e.what()}}.dump(),
                                    "application/json");
                    return;
                }
                if (!smpl) {
                    res.status = 500;
                    res.set_content("{\"error\":\"sampler init failed\"}", "application/json");
                    return;
                }

                int next_row = (int) cur.size();
                auto gen_one = [ctx, smpl, vocab, &st, &next_row](std::string & piece, bool & stopped, llama_token & id_out) -> bool {
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

                bool stopped = false;
                while ((int) gen.size() < budget && !stopped) {
                    if (!stream_heartbeat(&io)) {
                        common_sampler_free(smpl);
                        return;
                    }
                    std::string piece;
                    llama_token id = 0;
                    if (!gen_one(piece, stopped, id)) {
                        common_sampler_free(smpl);
                        res.status = 500;
                        res.set_content("{\"error\":\"decode failed\"}", "application/json");
                        fatal = true;
                        break;
                    }
                    if (stopped) {
                        break;
                    }
                    content += piece;
                    gen.push_back(id);
                    st.progress.generated((int) all_gen.size() + (int) gen.size());
                    st.log.generated((int) all_gen.size() + (int) gen.size());
                    if (strip_stop(content, stops)) {
                        round_stop = true;
                        break;
                    }
                }
                common_sampler_free(smpl);
                if (fatal) {
                    break;
                }
                round_eos   = stopped;
                round_limit = !stopped && !round_stop && (int) gen.size() >= budget;
            }

            all_gen.insert(all_gen.end(), gen.begin(), gen.end());
            if (fatal) {
                break;
            }

            const bool want_more = act_on && st.act.thinking_on && round_limit
                    && !round_eos && !round_stop
                    && (act_round + 1) < st.act.max_rounds
                    && (st.act.total_tokens <= 0 || (int) all_gen.size() < st.act.total_tokens);
            if (!want_more) {
                hit_limit = round_limit;
                break;
            }

            commit_cached(st, cur, gen);
            cur.insert(cur.end(), gen.begin(), gen.end());
            if (!st.act.cue.empty()) {
                const auto cue_ids = common_tokenize(vocab, st.act.cue, false, true);
                cur.insert(cur.end(), cue_ids.begin(), cue_ids.end());
            }
            act_prepare_next_round(st, vocab, gen, budget);
            kvmem_diag("KVMEM_ACT_ROUND k=%d n_gen=%d total=%d in_think=1\n",
                    act_round, (int) gen.size(), (int) all_gen.size());
            act_round++;
            int round_cache_hit = 0;
            bool ok_prefill;
            {
                const bool saved_policy = st.query_policy_user;
                if (st.act.prefill_mode == "legacy")  { st.query_policy_user = false; }
                if (st.act.prefill_mode == "restart") { memory_clear_all(st); }
                ok_prefill = run_prefill_retrieval(st, cur, &io, &round_cache_hit);
                st.query_policy_user = saved_policy;
            }
            if (!ok_prefill) {
                if (!io.aborted) {
                    res.status = 500;
                    res.set_content(json{{"error", "continuation prefill failed"}}.dump(), "application/json");
                }
                return;
            }
            n_cache_hit += round_cache_hit;
            llama_kvmem_end_prefill_capture();
            st.mm_live_checkpoint.reset();
        }

        if (fatal) {
            return;
        }
        llama_kvmem_decode_mean_flush();
        if (hit_limit) {
            act_close_thinking(content, st.act);
        }
        emit_gen_wall((int) all_gen.size());
        commit_cached(st, cur, gen);
        emit_json(content, (int) all_gen.size(), hit_limit);
    };

    svr.Post("/v1/chat/completions", handle_chat);
    svr.Post("/chat/completions", handle_chat);

    if (!svr.bind_to_port(host, port)) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR cannot bind %s:%d; check --host/--port, permissions and port conflicts\n", host.c_str(), port);
        return 1;
    }
    kvmem_diag("KVMEM_STARTUP ready=%s\n", startup.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    LOG_INF("srv    llama-kvmem-server listening on http://%s:%d  model=%s kvmem=%d method=%s n_ctx=%d spec=%s n_max=%d think=%d rbudget=%d qmax=%d\n",
            host.c_str(), port, st.model_name.c_str(), (int) st.kparams.enabled,
            st.kparams.method == 1 ? "retrieval" : "recency", n_ctx,
            st.spec.ok ? "draft-mtp" : "off", st.spec_n_max, (int) st.enable_thinking_default,
            st.reasoning_budget_default, st.query_max_tokens);
    if (!svr.listen_after_bind()) {
        fprintf(stderr, "listen failed\n");
        return 1;
    }
    st.vision.reset();
    kvmem_spec_stop(st.spec);
    llama_free(st.ctx);
    llama_model_free(st.model);
    return 0;
}
