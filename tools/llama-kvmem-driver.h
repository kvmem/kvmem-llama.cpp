#pragma once

// Engine-private state and lifecycle. Shared KVMem contracts must not include
// this header: llama, recurrent checkpoints and native prompt types stay here.
#include "llama.h"
#include "llama-kvmem-hooks.h"
#include "kvmem-spec.h"
#include "kvmem-vision.h"
#include "kvmem-prefill-policy.h"
#include "kvmem-conversation-store.h"
#include "kvmem-session-transfer.h"
#include "kvmem-execution-scope.h"
#include "kvmem/request_control.hpp"
#include "chat.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct MultimodalCheckpointAccounting {
    std::atomic<size_t> live_bytes {0};
    std::atomic<size_t> peak_bytes {0};
    void add(size_t bytes) {
        const size_t live = live_bytes.fetch_add(bytes) + bytes;
        size_t peak = peak_bytes.load();
        while (peak < live && !peak_bytes.compare_exchange_weak(peak, live)) {}
    }
};

struct MultimodalCheckpointData {
    MultimodalCheckpointData() = default;
    MultimodalCheckpointData(const MultimodalCheckpointData &) = delete;
    MultimodalCheckpointData & operator=(const MultimodalCheckpointData &) = delete;
    ~MultimodalCheckpointData() { if (accounting) accounting->live_bytes -= bytes(); }
    std::vector<uint8_t> recurrent;
    std::vector<uint8_t> draft_carry;
    llama_kvmem_tail_mean_state tail_mean;
    std::shared_ptr<MultimodalCheckpointAccounting> accounting;
    size_t bytes() const { return recurrent.size() + draft_carry.size() + tail_mean.values.size()*sizeof(float); }
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

// N host KV stores, one GPU working set, time-multiplexed
// (--kvmem-conversations). LlamaEngineState keeps holding the ACTIVE conversation's
// payload under the field names it already uses; kvmem_conversation holds the
// payload of every conversation, and the active entry's payload members are
// empty while they are on loan to LlamaEngineState. Metadata (client_id, stored) is
// always authoritative in the entry, never in LlamaEngineState.
//
// conversation_swap() below is the single list of conversation-scoped fields. A
// new one added to LlamaEngineState and forgotten there would leak state across
// conversations, so the struct and the swap belong in view of each other.
struct kvmem_conversation {
    bool cold = false;
    bool disk_gen = false, disk_query = false;
    std::unique_ptr<kvmem_session_payload> payload; // frozen RAM/disk allocation manifest
    std::string client_id;  // bound kvmem.conversation_id; empty = inferred
    uint32_t stored = 0;    // llama_kvmem_store_n_tokens() at the last commit
    std::vector<llama_token> cached_tokens;
    std::shared_ptr<kvmem_prompt> cached_prompt;
    std::vector<MultimodalCheckpoint> mm_checkpoints;
    int mm_live_row = 0;
    std::shared_ptr<const MultimodalCheckpointData> mm_live_checkpoint;
    std::shared_ptr<const MultimodalQuery> mm_query;
    std::vector<uint8_t> gdn_ckpt;
    std::vector<uint8_t> gdn_carry, gdn_query_carry;
    int gdn_ckpt_pos = -1;
    std::vector<uint8_t> gdn_ckpt_query;
    int gdn_ckpt_query_pos = -1;
    int last_query_begin = -1;
    int last_query_end = -1;
    std::string last_user_text;
    int last_n_gen = 0;
};

struct kvmem_conv_counts {
    uint64_t disk_bytes = 0, disk_bytes_max = 0;
    uint64_t spills = 0, restores = 0, disk_errors = 0;
    int count = 1;
    int max = 1;
    int active = 0;
    uint64_t bytes = 0;
    uint64_t bytes_max = 0;
    uint64_t extends = 0;
    uint64_t forks = 0;
    // Requests that planned a different conversation and stayed on the
    // attached one. Neither an extend nor a fork: nothing was forked, parked
    // or created.
    uint64_t refusals = 0;
    uint64_t resets = 0;
    uint64_t evictions = 0;
    uint64_t switches = 0;
};

// /slots never takes the inference lock, and kvmem_server_progress resets its
// payload per task, so these sticky counters are published rather than read.
class kvmem_conv_stats {
public:
    void publish(const kvmem_conv_counts & counts) {
        std::lock_guard<std::mutex> lock(mu_);
        data_ = counts;
    }
    kvmem_conv_counts snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return data_;
    }
private:
    mutable std::mutex mu_;
    kvmem_conv_counts data_;
};

struct LlamaEngineState {
    std::unique_ptr<llama_kvmem_execution_state, decltype(&llama_kvmem_execution_free)> execution{
        llama_kvmem_execution_create(), llama_kvmem_execution_free};
    ~LlamaEngineState() {
        kvmem_execution_scope scope(execution.get());
        vision.reset();
        kvmem_spec_stop(spec);
        if (ctx) llama_free(ctx);
    }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr tmpls;
    llama_kvmem_params kparams {};
    int n_batch = 512;
    int query_last_fallback = 64;
    int query_max_tokens = 512;
    bool query_replay_auto = true;
    bool query_policy_user = true;
    bool recurrent_cache_valid = true; // Positions alone do not identify a conversation.
    uint32_t turn_generation_rows = 0;
    bool turn_query_exact = false;
    kvmem_spec_session spec;
    ggml_type cache_type_k = GGML_TYPE_Q8_0;
    ggml_type cache_type_v = GGML_TYPE_Q8_0;
    ggml_type spec_cache_type = GGML_TYPE_F16;
    bool spec_mtp = false;
    int spec_n_max = 3;
    float spec_p_min = 0.0f;
    std::vector<llama_token> cached_tokens;
    std::shared_ptr<kvmem_vision> vision;
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
    // Multi-conversation host stores. max_stores <= 1 (the default) keeps conv
    // empty, conv_active at -1 and every conversation_* helper an early return.
    kvmem_store_limits conv_limits;
    kvmem_store_table conv_table;
    std::map<int, kvmem_conversation> conv;
    int conv_active = -1;
    uint64_t conv_clock = 0;
    bool conv_budget_warned = false;
    std::string turn_conversation_id;
    kvmem_conv_counts conv_counts;
    kvmem_conv_stats conv_stats;
    std::unique_ptr<kvmem_session_files> session_files;
    uint64_t session_generation = 0;
};

// Call under the lane lock and its kvmem_execution_scope. Transport owns cancellation.
std::vector<llama_token> llama_driver_tokenize(const llama_vocab * vocab, const std::string & text, bool add_special);

void llama_driver_swap_conversation(LlamaEngineState & st, kvmem_conversation & conv);

void llama_driver_drop_conversation(kvmem_conversation & conv);
void llama_driver_conversation_buffers(kvmem_conversation & conv, std::vector<kvmem::SnapshotBuffer> & buffers);

void llama_driver_publish_conversations(LlamaEngineState & st);

void llama_driver_clear(LlamaEngineState & st);

void llama_driver_begin_request(LlamaEngineState & st, const kvmem_prompt & prompt,
                                       const std::string & client_id);

void llama_driver_begin_disk_request(LlamaEngineState & st, const kvmem_prompt & prompt,
                                  const std::string & client_id, int predict);
void llama_driver_check_payload_budget(LlamaEngineState & st, const kvmem_prompt & prompt, int predict);

void llama_driver_commit(LlamaEngineState & st, const std::vector<llama_token> & prompt,
                          const std::vector<llama_token> & gen);

bool llama_driver_prepare(LlamaEngineState & st, const std::vector<llama_token> & prompt,
                                 kvmem::RequestControl * io = nullptr, int * n_cache_hit = nullptr);

void llama_driver_query_span(LlamaEngineState & st, const std::string & prompt, const std::string & last_user,
                              const std::vector<llama_token> & toks, int & qbegin, int & qend);

bool llama_driver_native_query_span(const LlamaEngineState & st, const std::string & formatted,
                                      const common_chat_templates_inputs & inputs, const kvmem_prompt & prompt,
                                      int & begin, int & end);

void llama_driver_clamp_query(const LlamaEngineState & st, int & qbegin, int & qend);

void llama_driver_validate_capacity(const LlamaEngineState & st, const kvmem_prompt & prompt, int end);

void llama_driver_finish(LlamaEngineState & st);

int llama_driver_decode_generated(LlamaEngineState & st, llama_token id, int row);
