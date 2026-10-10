#include "llama-kvmem-diag.h"
#include "llama.h"
#include "llama-kvmem-hooks.h"
#include "llama-kvmem-execution.h"
#include "kvmem-spec.h"
#include "kvmem-chat-sampling.h"
#include "kvmem-chat-template.h"
#include "kvmem-chat-id.h"
#include "kvmem-webui.h"
#include "kvmem-server-options.h"
#include "kvmem-server-auth.h"
#include "kvmem-server-progress.h"
#include "kvmem-lane-pool.h"
#include "kvmem-execution-scope.h"
#include <atomic>
#include "kvmem-server-devices.h"
#include "kvmem-server-env.h"
#include "kvmem-vision.h"
#include "llama-kvmem-driver.h"
#include "kvmem-conversation-store.h"
#include "kvmem-session-files.h"
#include "kvmem-session-transfer.h"
#include "kvmem/session_memory.hpp"

#include "chat.h"
#include "kvmem-responses.h"
#include "kvmem-responses-stream.h"
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
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
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
            "  Independent OpenAI-compatible server with up to two text lanes. Does not patch llama-server.\n"
            "  Endpoints: /v1/chat/completions, /v1/responses (streaming and non-streaming),\n"
            "             /v1/models, /props, /slots, /health, plus the bundled chat UI.\n"
            "\n"
            "  -m, --model PATH           GGUF path\n"
            "  --mmproj PATH              vision projector GGUF\n"
            "  --mmproj-offload           place vision encoder on GPU (default)\n"
            "  --no-mmproj-offload        place vision encoder on CPU\n"
            "  -mmdev, --mmproj-device DEVICE  select vision device, e.g. CUDA1 or Vulkan0 (none = CPU)\n"
            "  --image-min-tokens N       native minimum image token count\n"
            "  --image-max-tokens N       native maximum image token count\n"
            "  --video-fps F              video sampling fps for video input (default 2.0; <=0 uses the video's native fps)\n"
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
            "  -np, --parallel N          1..4 independent inference lanes (default: 1)\n"
            "                            context and KV budgets are per lane\n"
            "  -lm, --load-mode MODE      auto | none | mmap | mlock | mmap+mlock | dio\n"
            "  --mmap / --no-mmap         legacy aliases for load-mode mmap / none\n"
            "  --mlock                   legacy alias for load-mode mlock\n"
            "  --timeout, -to N          HTTP read/write timeout seconds (default 1800)\n"
            "  --threads-http N          HTTP worker threads; <=0 = automatic\n"
            "  --device, -dev NAMES      offload devices, e.g. CUDA0,CUDA1; none = CPU\n"
            "  --list-devices            list available offload devices and exit\n"
            "  --main-gpu, -mg N         main device index (default 0)\n"
            "  --split-mode, -sm MODE    none | layer | tensor (multi-GPU requires layer or tensor)\n"
            "  --tensor-split, -ts N,... proportions, one per selected GPU\n"
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
            "  --kvmem-host-mib MIB      shared native K/V RAM quota (default unlimited)\n"
            "  --kvmem-disk-mib MIB      active/idle native K/V SSD quota (default off)\n"
            "  --kvmem-disk-path PATH    process-local native K/V spill directory\n"
            "  --kvmem-nvme-gb GB         NVMe file in GiB (0 = off)\n"
            "  --kvmem-nvme-dir PATH      NVMe directory (default /tmp/kvmem_nvme)\n"
            "  --kvmem-harvest-v          prefill D2H V with raw-K (default off; RAM until NVMe flush)\n"
            "  --kvmem-raw-k-nvme         store raw-K and V on NVMe (needs --kvmem-nvme-gb)\n"
            "  --kvmem-conversations N    live host KV stores, time-multiplexed on one GPU\n"
            "                            working set (default 1 = today's single store)\n"
            "  --kvmem-conversations-gb GB soft cap on total active + idle session RAM, moving\n"
            "                            idle KV by LRU (0 = unlimited; active may exceed cap)\n"
            "  --kvmem-session-ram-gb GB  alias for the RAM soft cap\n"
            "  --kvmem-session-nvme-gb GB disk quota for inactive sessions (default 0 = off)\n"
            "  --kvmem-session-cache-dir PATH private cache directory on your NVMe/SSD\n"
            "                            LRU RAM -> disk -> discard; active session must fit RAM\n"
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
            "  --webui                    serve bundled chat UI (default on)\n",
            argv0);
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

struct ServerState : LlamaEngineState {
    std::mutex mu;
    kvmem_server_progress progress;
    kvmem_server_log log;
    int n_predict_default = -1;
    json sampling_overrides = json::object();
    std::string model_name = "kvmem";
    bool enable_thinking_default = false;
    std::map<std::string, std::string> template_kwargs;
    int reasoning_budget_default = -1;
    std::string reasoning_budget_message;
};

struct StreamIo;
static bool stream_heartbeat(StreamIo * io);

struct StreamIo : kvmem::RequestControl {
    StreamIo() { keep_alive = [this]() { return stream_heartbeat(this); }; }
    StreamIo(const StreamIo &) = delete;
    StreamIo & operator=(const StreamIo &) = delete;
    httplib::DataSink * sink = nullptr;
    const httplib::Request * req = nullptr;
    std::chrono::steady_clock::time_point last_beat{};
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

#include "kvmem-lane-conversations.h"

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
    std::string conversation_id;
    bool enable_thinking = false;
    std::map<std::string, std::string> template_kwargs;
    int reasoning_budget_tokens = -1;
    std::string reasoning_budget_message;
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

// OpenAI Responses "output" items for one assistant message. Kept as a free
// function so a future streaming path can emit the same items incrementally.
// call_id reuses the chat tool_call id verbatim so the client echoes it back
// unchanged in a function_call_output item.
static std::vector<json> responses_output_items(
        const common_chat_msg & msg,
        const std::string & request_id) {
    std::vector<json> output;
    if (!msg.reasoning_content.empty()) {
        output.push_back(json {
            {"id", "rs_" + request_id},
            {"summary", json::array({json{
                {"text", msg.reasoning_content},
                {"type", "summary_text"},
            }})},
            {"type", "reasoning"},
            {"content", json::array({json{
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status", "completed"},
        });
    }
    if (!msg.content.empty()) {
        output.push_back(json {
            {"content", json::array({json{
                {"type", "output_text"},
                {"annotations", json::array()},
                {"logprobs", json::array()},
                {"text", msg.content},
            }})},
            {"id", "msg_" + request_id},
            {"role", msg.role.empty() ? std::string("assistant") : msg.role},
            {"status", "completed"},
            {"type", "message"},
        });
    }
    for (const common_chat_tool_call & tool_call : msg.tool_calls) {
        output.push_back(json {
            {"id", "fc_" + tool_call.id},
            {"type", "function_call"},
            {"status", "completed"},
            {"arguments", tool_call.arguments},
            {"call_id", tool_call.id},
            {"name", tool_call.name},
        });
    }
    return output;
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

// Adapts Responses SSE events to the Chat Completions helpers below. The
// Responses path reuses StreamChatOut for delta computation and rendering state
// (set_text / finish_reason), so one server loop drives both wire formats.
// `json` here is the server's nlohmann type, not upstream common_json.
struct ResponsesStreamOut {
    common_chat_msg prev;
    std::string acc;
    std::vector<std::string> tc_ids;
    std::string request_id;
    KvMemResponsesStreamState state;
    int n_id = 0;

    common_chat_parser_params pp;

    ResponsesStreamOut(const common_chat_params & chat, bool parse_tools, const std::string & id)
        : request_id(id) {
        pp = common_chat_parser_params(chat);
        pp.parse_tool_calls = parse_tools;
        if (!chat.parser.empty()) {
            pp.parser.load(chat.parser);
        }
    }

    std::vector<std::string> set_text(const std::string & text, bool partial) {
        acc = text;
        std::vector<std::string> events;
        try {
            common_chat_msg msg = common_chat_parse(acc, partial, pp);
            if (msg.empty() && partial) {
                return events;
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
                for (std::string & ev : kvmem_responses_stream_events(state, d, request_id)) {
                    events.push_back(std::move(ev));
                }
            }
        } catch (const std::exception & e) {
            if (!partial) {
                LOG_WRN("srv    KVMEM_TRACE responses_stream_parse_fail %s\n", e.what());
            }
        }
        return events;
    }
};

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
        // Optional, never required, and never a reason to fail a request: this
        // key was unrecognized at v0.16.0-rc3, so a client that sends one must
        // still be served, and kvmem_store_client_id drops a value the policy
        // cannot use instead of rejecting it. Ignored entirely with one host
        // store, where conversation identity is the token prefix itself.
        if (k.contains("conversation_id") && k["conversation_id"].is_string()) {
            out.conversation_id = kvmem_store_client_id(k["conversation_id"].get<std::string>());
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
    return true;
}

int main(int argc, char ** argv) {
    // Flush asynchronous common logs on every exit, including startup errors.
    struct log_flush_guard { ~log_flush_guard() { common_log_flush(common_log_main()); } } flush_logs;
    std::string ui_dir;
    bool no_ui = false;
    std::string model_path;
    std::string mmproj_path;
    std::string mmproj_device_name;
    std::string chat_template;
    bool template_set = false;
    std::string template_source;
    json template_defaults = json::object();
    bool mmproj_gpu = true;
    int image_min_tokens = -1, image_max_tokens = -1;
    float video_fps = 2.0f;
    std::string host = "127.0.0.1";
    std::string nvme_dir;
    int port = 8080;
    int n_ctx = 2048;
    int ngl = 99;
    kvmem_server_devices device_config;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> shared_model(nullptr, llama_model_free);
    ServerState st;
    kvmem_execution_scope startup_execution(st.execution.get());
    kvmem_server_options options;
    int requested_conversations = 1;
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
        } else if (eq(arg, "--mmproj-device") || eq(arg, "-mmdev")) {
            mmproj_device_name = need(arg);
            if (mmproj_device_name == "none") {
                mmproj_device_name.clear();
                mmproj_gpu = false;
            } else if (mmproj_device_name.empty()) {
                throw std::invalid_argument("--mmproj-device requires a device name or none");
            } else {
                mmproj_gpu = true;
            }
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
        } else if (eq(arg, "--video-fps")) {
            const std::string value = need(arg);
            try {
                size_t used = 0;
                const float f = std::stof(value, &used);
                if (used != value.size() || !std::isfinite(f)) throw std::invalid_argument("finite number required");
                video_fps = f;
            } catch (...) {
                fprintf(stderr, "%s requires a finite number (<=0 uses the video's native fps)\n", arg);
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
        } else {
            fprintf(stderr, "unknown flag: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }
    requested_conversations = options.conversations;
    options.conversations = std::max(options.conversations, options.parallel);
    st.kparams.sink_tokens = static_cast<uint32_t>(options.sink_tokens);
    st.kparams.payload_host_bytes = options.payload_host_bytes;
    st.kparams.payload_disk_bytes = options.payload_disk_bytes;
    st.kparams.payload_disk_dir = options.payload_disk_dir.empty() ? nullptr : options.payload_disk_dir.c_str();
    if (options.payload_disk_dir.empty() != (options.payload_disk_bytes == 0))
        throw std::invalid_argument("--kvmem-disk-path and --kvmem-disk-mib must be provided together");
    if (options.payload_disk_bytes && (!options.payload_host_bytes || !st.kparams.enabled))
        throw std::invalid_argument("KVMem SSD requires --kvmem-host-mib and KVMem enabled");
    if (options.payload_disk_bytes && (options.session_disk_bytes || st.kparams.cpu_bytes ||
            st.kparams.nvme_bytes || st.kparams.raw_k_nvme))
        throw std::invalid_argument("shared KV SSD cannot be combined with legacy spill stores");
    // Cross-checks on the two new flags, thrown so they reach the
    // "invalid arguments (source=...)" printer below the way every other
    // rejection here does. Neither can fire without one of the flags, so the
    // default path prints nothing new.
    if (options.conversations > 1 && !st.kparams.enabled) {
        throw std::invalid_argument("--kvmem-conversations > 1 requires KVMem; drop --no-kvmem");
    }
    if (options.conversation_bytes != 0 && options.conversations <= 1) {
        throw std::invalid_argument("--kvmem-conversations-gb caps the host stores that "
                                    "--kvmem-conversations N creates; pass N > 1 or drop the cap");
    }
    if (options.session_disk_bytes) {
        if (options.conversations <= 1 || options.session_cache_dir.empty())
            throw std::invalid_argument("session disk cache requires --kvmem-conversations N > 1, "
                "and --kvmem-session-cache-dir PATH");
        if (st.kparams.cpu_bytes || st.kparams.nvme_bytes || st.kparams.raw_k_nvme)
            throw std::invalid_argument("session disk cache requires --kvmem-cpu-gb 0 and --kvmem-nvme-gb 0; "
                "do not combine it with --kvmem-raw-k-nvme");
    } else if (!options.session_cache_dir.empty()) {
        throw std::invalid_argument("--kvmem-session-cache-dir requires --kvmem-session-nvme-gb > 0");
    }
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
    ggml_backend_dev_t mmproj_device = nullptr;
    if (mmproj_gpu && !mmproj_device_name.empty()) {
        mmproj_device = ggml_backend_dev_by_name(mmproj_device_name.c_str());
        const auto type = mmproj_device ? ggml_backend_dev_type(mmproj_device) : GGML_BACKEND_DEVICE_TYPE_CPU;
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            fprintf(stderr, "KVMEM_STARTUP_ERROR invalid --mmproj-device %s; use --list-devices\n",
                    mmproj_device_name.c_str());
            return 1;
        }
    }

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
    if (mparams.split_mode == LLAMA_SPLIT_MODE_TENSOR && st.spec_mtp && st.kparams.enabled && st.kparams.mtp_state != 0) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR tensor KVMem currently supports MTP snapshots only; use --kvmem-mtp-state snapshots\n");
        return 1;
    }
    if (device_config.devices.size() > 2 && st.spec_mtp && mparams.split_mode != LLAMA_SPLIT_MODE_TENSOR &&
            (!st.kparams.enabled || st.kparams.mtp_state == 1 ||
             (st.kparams.mtp_state == 2 && !config_sources.contains("--kvmem-mtp-state")))) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR multi-GPU MTP requires --kvmem and explicit --kvmem-mtp-state snapshots|replay (auto is not supported yet)\n");
        return 1;
    }
    if (requested_conversations != options.conversations) {
        LOG_INF("srv    KVMEM conversations adjusted requested=%d effective=%d parallel=%d\n",
                requested_conversations, options.conversations, options.parallel);
    }
    if (options.parallel > 1 && (!st.kparams.enabled ||
            options.device_names == "none" || mparams.n_gpu_layers == 0 ||
            options.session_disk_bytes ||
            st.kparams.nvme_bytes || st.kparams.raw_k_nvme ||
            device_config.devices.size() > 2 || mparams.split_mode == LLAMA_SPLIT_MODE_TENSOR ||
            (options.threads_http > 0 && options.threads_http < 2 * options.parallel))) {
        fprintf(stderr, "KVMEM_STARTUP_ERROR --parallel %d requires KVMem, one target GPU, "
                "no NVMe/session disk cache, and at least %d HTTP workers\n",
                options.parallel, 2 * options.parallel);
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
                  {"main_gpu", mparams.main_gpu}, {"split_mode", mparams.split_mode == LLAMA_SPLIT_MODE_NONE ? "none" :
                      mparams.split_mode == LLAMA_SPLIT_MODE_TENSOR ? "tensor" : "layer"},
                  {"layers_requested", ngl}}},
        {"context_requested", n_ctx}, {"batch_requested", st.n_batch},
        {"n_predict", st.n_predict_default},
        {"kv", {{"k", ggml_type_name(st.cache_type_k)}, {"v", ggml_type_name(st.cache_type_v)}}},
        {"kvmem", {{"enabled", st.kparams.enabled}, {"budget", st.kparams.budget}, {"gen_reserve", st.kparams.gen_reserve},
                   {"sink_tokens", st.kparams.sink_tokens}, {"block_tokens", st.kparams.block_tokens}}},
        {"spec_type", st.spec_mtp ? "draft-mtp" : "none"},
        {"vision", {{"enabled", !mmproj_path.empty()}, {"projector", mmproj_path}, {"gpu", mmproj_gpu},
                    {"device", mmproj_gpu ? (mmproj_device_name.empty() ? "auto" : mmproj_device_name) : "CPU"},
                    {"video_fps", video_fps}}},
        {"http", {{"host", host}, {"port", port}, {"timeout", options.timeout}, {"slots", options.parallel}}},
        {"auth", {{"enabled", !options.api_keys.empty()}, {"key_count", options.api_keys.size()}}},
        {"sources", config_sources}, {"unlisted_sources", "default"}
    };
    kvmem_diag("KVMEM_STARTUP requested=%s\n", startup.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    LOG_INF("srv    loading model %s\n", model_path.c_str());
    shared_model.reset(llama_model_load_from_file(model_path.c_str(), mparams));
    st.model = shared_model.get();
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
    std::shared_ptr<kvmem::HostKvStorage> shared_payload_storage;
    auto initialize_lane = [&](ServerState & st) {
        kvmem_execution_scope scope(st.execution.get());
        llama_kvmem_set_params(&st.kparams);
        if (options.payload_disk_bytes) {
            st.execution->payload_domains = options.parallel;
            st.execution->payload_storage = shared_payload_storage;
        }
        st.ctx = llama_init_from_model(st.model, cparams);
        if (!st.ctx) {
            fprintf(stderr, "KVMEM_STARTUP_ERROR failed to create context; check --ctx-size, KV types, --flash-attn and available memory\n");
            return false;
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
                return false;
            }
        }

        if (st.kparams.payload_disk_bytes && !shared_payload_storage)
            shared_payload_storage = st.execution->payload_storage;
        if (st.kparams.payload_disk_bytes && !llama_kvmem_payload_budget()) {
            LOG_ERR("srv    KVMEM SSD RAM quota must hold a native working window including MTP\n");
            return false;
        }
        return true;
    };
    if (!initialize_lane(st)) return 1;
    std::vector<std::unique_ptr<ServerState>> additional_lanes;
    std::vector<ServerState *> lanes{&st};
    for (int lane = 1; lane < options.parallel; ++lane) {
        auto next = std::make_unique<ServerState>();
        next->model = st.model;
        next->vocab = st.vocab;
        next->kparams = st.kparams;
        next->n_batch = st.n_batch;
        next->n_predict_default = st.n_predict_default;
        next->sampling_overrides = st.sampling_overrides;
        next->query_last_fallback = st.query_last_fallback;
        next->query_max_tokens = st.query_max_tokens;
        next->query_replay_auto = st.query_replay_auto;
        next->query_policy_user = st.query_policy_user;
        next->model_name = st.model_name;
        next->cache_type_k = st.cache_type_k;
        next->cache_type_v = st.cache_type_v;
        next->spec_cache_type = st.spec_cache_type;
        next->spec_mtp = st.spec_mtp;
        next->spec_n_max = st.spec_n_max;
        next->spec_p_min = st.spec_p_min;
        next->enable_thinking_default = st.enable_thinking_default;
        next->template_kwargs = st.template_kwargs;
        next->reasoning_budget_default = st.reasoning_budget_default;
        next->reasoning_budget_message = st.reasoning_budget_message;
        next->tmpls = common_chat_templates_init(st.model, chat_template);
        if (!initialize_lane(*next)) return 1;
        lanes.push_back(next.get());
        additional_lanes.push_back(std::move(next));
    }
    kvmem_lane_pool lane_pool(lanes.size());
    LOG_INF("srv    KVMEM lanes=%zu shared_model=1 context_per_lane=%d budget_per_lane=%u "
            "generation_reserve_per_lane=%u\n", lanes.size(), n_ctx, st.kparams.budget, st.kparams.gen_reserve);

    if (!mmproj_path.empty()) {
        try {
            if (image_min_tokens > 0 && image_max_tokens > 0 && image_min_tokens > image_max_tokens)
                throw std::invalid_argument("image-min-tokens exceeds image-max-tokens");
            st.vision = std::make_shared<kvmem_vision>(
                    st.model, mmproj_path, mmproj_gpu, mmproj_device,
                    image_min_tokens, image_max_tokens, options.threads, video_fps);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }

    // Arm the multi-conversation host stores only now: the adapter refuses the
    // swap in configurations it cannot drain back from host RAM, and flash
    // attention is resolved inside llama_init_from_model, not at parse time.
    if (options.parallel == 1 && options.conversations > 1) {
        if (!llama_kvmem_store_swap_supported()) {
            if (options.session_disk_bytes) {
                LOG_ERR("srv    KVMEM session disk cache requires supported session switching (flash attention on)\n");
                return 1;
            }
            LOG_WRN("srv    KVMEM --kvmem-conversations %d unavailable in this configuration; "
                    "using one host store\n", options.conversations);
        } else {
            st.conv_limits.max_stores = options.conversations;
            st.conv_limits.max_bytes = options.conversation_bytes;
            if (options.session_disk_bytes || options.payload_disk_bytes) {
                try {
                    st.session_files = std::make_unique<kvmem_session_files>(
                        std::filesystem::u8path(options.payload_disk_bytes ? options.payload_disk_dir : options.session_cache_dir),
                        options.payload_disk_bytes ? options.payload_disk_bytes : options.session_disk_bytes,
                        [](bool warning, const std::string & message) {
                            if (warning) { LOG_WRN("srv    KVMEM session cache %s\n", message.c_str()); }
                            else { LOG_INF("srv    KVMEM session cache %s\n", message.c_str()); }
                        }, kvmem_execution_spill_file(st.execution.get()));
                } catch (const std::exception & e) {
                    LOG_ERR("srv    KVMEM session cache initialization failed: %s\n", e.what()); return 1;
                }
                LOG_INF("srv    KVMEM session disk cache=%s quota=%llu bytes\n", st.session_files->directory().u8string().c_str(),
                    (unsigned long long)st.session_files->limit());
            }
            st.conv_active = st.conv_table.add(llama_kvmem_store_current());
            st.conv.emplace(st.conv_active, kvmem_conversation{});
            st.conv_table.touch(st.conv_active, ++st.conv_clock);
            llama_driver_publish_conversations(st);
            LOG_INF("srv    KVMEM conversations=%d host_bytes_max=%llu\n",
                    options.conversations, (unsigned long long) st.conv_limits.max_bytes);
            // Every host store builds its own runtime, so the pinned arena
            // --kvmem-cpu-gb sizes and the tier --kvmem-nvme-gb sizes are
            // allocated once per store, not once per process. The server owns
            // the conversation caps and the adapter is never told the store
            // count, so say what those two numbers now mean rather than
            // quietly dividing them.
            if (st.kparams.cpu_bytes || st.kparams.nvme_bytes) {
                const double gib = 1024.0 * 1024.0 * 1024.0;
                LOG_WRN("srv    KVMEM --kvmem-cpu-gb and --kvmem-nvme-gb are per host store: "
                        "%d stores commit up to %.1f GiB pinned (allocated and zero-filled as "
                        "stores are created) and overcommit %.1f GiB NVMe (sparse, unlinked)\n",
                        options.conversations,
                        options.conversations * (double) st.kparams.cpu_bytes / gib,
                        options.conversations * (double) st.kparams.nvme_bytes / gib);
            }
        }
    }

    std::unique_ptr<kvmem_lane_conversations> lane_conversations;
    if (options.parallel > 1) {
        for (auto * lane : lanes) {
            kvmem_execution_scope execution(lane->execution.get());
            if (!llama_kvmem_store_swap_supported()) {
                LOG_ERR("srv    KVMEM multiple lanes require supported host KV switching (flash attention on)\n");
                return 1;
            }
            if (!llama_kvmem_store_compatible_with_execution(lanes[0]->execution.get())) {
                LOG_ERR("srv    KVMEM lanes need identical KV geometry; set an explicit --kvmem-budget\n");
                return 1;
            }
            lane->vision = st.vision;
        }
        lane_conversations = std::make_unique<kvmem_lane_conversations>(lanes, options.conversations, options.conversation_bytes);
        LOG_INF("srv    KVMEM global conversations=%d lanes=%d cpu_budget_per_store=%llu\n",
                options.conversations, options.parallel, (unsigned long long)st.kparams.cpu_bytes);
    }
    auto preparation_templates = common_chat_templates_init(st.model, chat_template);
    std::mutex preparation_templates_mutex;
    httplib::Server svr;
    svr.set_read_timeout(options.timeout, 0);
    svr.set_write_timeout(options.timeout, 0);
    const int http_workers = options.threads_http_set ?
        (options.threads_http > 0 ? options.threads_http :
            std::max(2 * options.parallel, std::max(5, static_cast<int>(std::thread::hardware_concurrency()) - 1))) :
        std::max(2 * options.parallel, static_cast<int>(CPPHTTPLIB_THREAD_POOL_COUNT));
    if (options.threads_http_set || options.parallel > 1) {
        svr.new_task_queue = [http_workers] {
            return new httplib::ThreadPool(http_workers, static_cast<size_t>(http_workers) + 1024);
        };
        kvmem_diag("KVMEM_HTTP threads=%d timeout=%d\n", http_workers, options.timeout);
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
    startup["http"]["threads"] = http_workers;
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
    json props = {
        // upstream get_res_props fields
        // 中文：上游 /props 返回的标准字段，UI 依赖这些键渲染模型信息
        {"default_generation_settings", {{"params", default_params}, {"n_ctx", n_ctx}}},
        {"total_slots", options.parallel},
        {"model_alias", st.model_name},
        // kvmem's llama.cpp predates llama_model_ftype_name(); keep the field for UI parity.
        // 中文：当前 kvmem 的 llama.cpp 尚无 llama_model_ftype_name()，保留空字段以兼容上游 UI 展示
        {"model_ftype", ""},
        {"model_path", model_path},
        {"modalities", {{"vision", st.vision != nullptr}, {"audio", false},
                        {"video", st.vision && options.parallel == 1 && st.vision->supports_video()}}},
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
    props["kvmem"]["vision"] = startup["vision"];
    props["kvmem"]["parallel"] = options.parallel;
    props["kvmem"]["conversations_requested"] = requested_conversations;
    if (options.parallel > 1) props["kvmem"]["conversations"] = options.conversations;
    props["kvmem"]["conversation_cpu_bytes"] = st.kparams.cpu_bytes;
    props["kvmem"]["conversation_bytes_max"] = options.conversation_bytes;
    // The legacy single-lane capability can degrade to one store.
    if (st.conv_limits.max_stores > 1) {
        props["kvmem"]["conversations"] = st.conv_limits.max_stores;
    }
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
        json slots = json::array();
        bool all_busy = true;
        for (size_t lane = 0; lane < lanes.size(); ++lane) {
            const auto & st = *lanes[lane];
            const auto progress = st.progress.snapshot();
            const bool busy = progress.busy;
            json slot = {
                {"id", lane},
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
            // Conversation counters belong to this lane.
            if (st.conv_limits.max_stores > 1) {
                const auto conv = st.conv_stats.snapshot();
                slot["kvmem"] = {{"conversations", {
                    {"count", conv.count},
                    {"max", conv.max},
                    {"active", conv.active},
                    {"bytes", conv.bytes},
                    {"bytes_max", conv.bytes_max},
                    {"extends", conv.extends},
                    {"forks", conv.forks},
                    {"refusals", conv.refusals},
                    {"resets", conv.resets},
                    {"evictions", conv.evictions},
                    {"switches", conv.switches}}}};
                auto & sessions = slot["kvmem"]["conversations"];
                sessions["disk_bytes"] = conv.disk_bytes;
                sessions["disk_bytes_max"] = conv.disk_bytes_max;
                sessions["spills"] = conv.spills;
                sessions["restores"] = conv.restores;
                sessions["disk_errors"] = conv.disk_errors;
            }
            if (lane_conversations) slot["kvmem"]["conversations"] = lane_conversations->status((int)lane);
            const auto storage = llama_kvmem_execution_storage_stats(st.execution.get());
            slot["kvmem"]["storage"] = {
                {"host_payload_bytes", storage.host_bytes}, {"host_payload_budget_bytes", storage.host_capacity},
                {"disk_payload_bytes", storage.disk_bytes}, {"disk_payload_budget_bytes", storage.disk_capacity},
                {"disk_read_bytes", storage.read_bytes}, {"disk_written_bytes", storage.written_bytes},
                {"disk_reads", storage.reads}, {"disk_writes", storage.writes}, {"disk_errors", storage.errors}};
            all_busy = all_busy && busy;
            slots.push_back(std::move(slot));
        }
        if (all_busy && req.has_param("fail_on_no_slot")) {
            res.status = 503;
            res.set_content(json{{"error", json{
                {"code", 503},
                {"message", "no slot available"},
                {"type", "unavailable_error"},
            }}}.dump(), "application/json");
            return;
        }
        res.set_content(slots.dump(), "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {
            {"object", "list"},
            {"data", json::array({json{
                {"id", st.model_name},
                {"name", st.model_name},
                {"object", "model"},
                {"status", {{"value", "loaded"}}},
                {"meta", {
                    {"n_ctx", llama_n_ctx(st.ctx)},
                    {"n_ctx_train", llama_model_n_ctx_train(st.model)},
                    {"n_embd", llama_model_n_embd(st.model)},
                }},
            }})},
        };
        res.set_content(j.dump(), "application/json");
    });

    // Diagnostic: dump each /v1/responses request, and the Chat Completions body
    // it converts to, into $KVMEM_DBG_DIR. A client whose request this server
    // only half-understands is invisible on the wire -- it gets a well-formed but
    // empty response -- so seeing the exact bytes a client sends is the only way
    // to tell which item shape it uses. Every request gets its own numbered pair,
    // so a multi-turn exchange (tool call, then the tool result sent back) reads
    // in order. Unset KVMEM_DBG_DIR = no-op, which is the normal case.
    const auto kvmem_debug_dump = [seq = std::make_shared<std::atomic<int>>(0)](
            const char * name, const std::string & data) {
        const char * dir = std::getenv("KVMEM_DBG_DIR");
        if (dir == nullptr || *dir == '\0') {
            return;
        }
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "%03d-", ++*seq);
        const std::string path = std::string(dir) + "/" + prefix + name;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(data.data(), (std::streamsize) data.size());
        std::fprintf(stderr, "[KVMEM_DBG] wrote %s (%zu bytes)\n", path.c_str(), data.size());
        std::fflush(stderr);
    };

    auto handle_chat = [&](const httplib::Request & req, httplib::Response & res) {
        const std::time_t created = std::time(nullptr);
        // /v1/responses reuses this handler: convert the Responses request to
        // Chat Completions first, then select the Responses output shape below.
        const bool is_responses = req.path == "/v1/responses" || req.path == "/responses";
        std::string body_text = req.body;
        if (is_responses) {
            kvmem_debug_dump("responses-raw.json", req.body);
            try {
                body_text = kvmem_responses_to_chatcmpl(req.body);
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
                return;
            }
            kvmem_debug_dump("responses-converted.json", body_text);
        } else {
            kvmem_debug_dump("chatcmpl-raw.json", req.body);
        }        json body;
        std::vector<std::vector<uint8_t>> media_files;
        try {
            body = json::parse(kvmem_parse_media_messages(body_text, st.vision != nullptr,
                                                        st.vision && options.parallel == 1 && st.vision->supports_video(), media_files));
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

        const auto cancelled = [&req] { return req.is_connection_closed && req.is_connection_closed(); };
        auto ticket = lane_pool.enqueue(cr.conversation_id);
        auto cache_reference = lane_conversations ? lane_conversations->pin(cr.conversation_id) : nullptr;
        if (!lane_pool.wait_turn(ticket, cancelled)) return;

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
            const auto caps = common_chat_templates_get_caps(preparation_templates.get());
            const auto it = caps.find("supports_parallel_tool_calls");
            inputs.parallel_tool_calls = it != caps.end() && it->second;
        }
        common_chat_params formatted;
        try {
            std::lock_guard<std::mutex> preparation(preparation_templates_mutex);
            formatted = common_chat_templates_apply(preparation_templates.get(), inputs);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("chat template: ") + e.what()}}.dump(), "application/json");
            return;
        }
        const std::string & prompt = formatted.prompt;
        std::shared_ptr<kvmem_prompt> parsed_prompt;
        try {
            parsed_prompt = media_files.empty()
                ? std::make_shared<kvmem_prompt>(llama_driver_tokenize(st.vocab, prompt, true))
                : st.vision->tokenize(prompt, media_files);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
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
        try {
            llama_driver_validate_capacity(st, *parsed_prompt, (int) toks.size());
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        // Sampler construction and its probe sit here, above the conversation
        // mapping, because both of their failures answer 400. They read cr,
        // formatted, st.model and st.spec only, none of which the mapping
        // below touches.
        common_params_sampling sparams = make_chat_sampling(st.vocab, formatted, cr);
        if (!kvmem_chat_reasoning_budget_supported(sparams, cr.enable_thinking, err)) {
            res.status = 400;
            res.set_content(json{{"error", err}}.dump(), "application/json");
            return;
        }
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
        if (!use_spec) {
            try {
                common_sampler_ptr probe(common_sampler_init(st.model, sparams));
                if (!probe) throw std::runtime_error("sampler initialization failed");
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json"); return;
            }
        }
        auto operation = std::make_shared<kvmem_lane_conversations::operation>();
        size_t prepare_begin = lane_conversations
            ? lane_conversations->preview(*parsed_prompt, cr.conversation_id, body.value("cache_reset", false), cache_reference) : 0;
        std::shared_ptr<kvmem_lane_pool::lease> lease;
        try {
            for (;;) {
                if (st.vision && parsed_prompt->has_media()) st.vision->prepare(*parsed_prompt, prepare_begin, cancelled);
                lease = lane_pool.acquire(ticket, cancelled, [&](const std::vector<bool> & busy, bool reserve) {
                    if (lane_conversations) return lane_conversations->select(*parsed_prompt, cr.conversation_id,
                            body.value("cache_reset", false), busy, reserve, *operation);
                    for (size_t lane = 0; lane < busy.size(); ++lane) if (!busy[lane]) return (int)lane;
                    return -1;
                });
                if (lease || cancelled()) break;
                prepare_begin = operation->prepare_begin;
            }
        } catch (const std::exception & e) {
            if (cancelled()) return;
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json"); return;
        }
        if (!lease) return;
        ServerState & st = *lanes[lease->index];
        std::shared_ptr<kvmem_server_slot_guard> slot;
        try {
            slot = std::make_shared<kvmem_server_slot_guard>(st.mu, st.progress, [lease, operation, &st, &lane_conversations] {
                kvmem_execution_scope execution(st.execution.get());
                if (lane_conversations) lane_conversations->finish(*operation, st);
                else {
                    llama_driver_finish(st);
                    if (st.active_prompt) st.active_prompt->release_media();
                }
                lease->release();
            });
        } catch (...) {
            if (lane_conversations) lane_conversations->finish(*operation, st);
            throw;
        }
        kvmem_execution_scope execution(st.execution.get());
        res.set_header("X-KVMem-Lane", std::to_string(lease->index));
        if (cancelled()) return;
        st.mm_reset_requested = body.value("cache_reset", false);
        st.turn_conversation_id = cr.conversation_id;
        try {
            if (lane_conversations) lane_conversations->attach(*operation, st, cr.conversation_id);
            else if (st.session_files) llama_driver_begin_disk_request(st, *parsed_prompt, cr.conversation_id, cr.max_tokens);
            else llama_driver_begin_request(st, *parsed_prompt, cr.conversation_id);
            if (lane_conversations) lane_conversations->reserve_payload(*operation, st, *parsed_prompt, cr.max_tokens);
            else llama_driver_check_payload_budget(st, *parsed_prompt, cr.max_tokens);
        } catch (const std::exception & e) {
            res.status = 503;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json"); return;
        }
        st.active_prompt = parsed_prompt;
        st.turn_generation_rows = (uint32_t) std::min<uint64_t>(UINT32_MAX,
                (uint64_t) std::max(0, cr.max_tokens) + (st.spec.ok ? std::max(0, st.spec_n_max) + 1u : 0u));
        int qbegin = cr.query_begin;
        int qend = cr.query_end;
        st.turn_query_exact = false;
        st.turn_last_user = cr.last_user;
        if (qbegin < 0 || qend < 0) {
            llama_driver_query_span(st, prompt, cr.last_user, toks, qbegin, qend);
        }
        if (st.query_policy_user) {
            st.turn_query_exact = cr.query_begin >= 0 && cr.query_end > cr.query_begin && cr.query_end <= (int) toks.size();
            if (cr.query_begin < 0 && cr.query_end < 0) {
                st.turn_query_exact = llama_driver_native_query_span(st, prompt, inputs, *parsed_prompt, qbegin, qend);
            }
        }
        if (parsed_prompt->has_media() && cr.query_begin < 0 && !st.turn_query_exact) {
            // The final text question follows native visual chunks and their boundaries.
            int last_media_end = 0;
            for (const auto & range : parsed_prompt->media_ranges()) last_media_end = range.second;
            qend = (int) toks.size() - (st.spec.ok ? 1 : 0);
            qbegin = std::max(last_media_end, qend - st.query_max_tokens);
        }
        llama_driver_clamp_query(st, qbegin, qend);
        if (st.turn_query_exact && std::find(toks.begin() + qbegin, toks.begin() + qend, LLAMA_TOKEN_NULL) != toks.begin() + qend) {
            st.turn_query_exact = false;
            kvmem_diag("KVMEM_TRACE query_loc fallback=explicit_span_contains_media\n");
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
            if (is_responses) {
                kvmem_diag("KVMEM_TRACE chat_out n_tool_calls=%zu finish=%s content_chars=%zu reasoning_chars=%zu\n",
                        msg.tool_calls.size(), finish.c_str(),
                        msg.content.size(), msg.reasoning_content.size());
                const std::time_t now = std::time(nullptr);
                json out = {
                    {"completed_at", now},
                    {"created_at", now},
                    {"id", "resp_" + request_id},
                    {"model", st.model_name},
                    {"object", "response"},
                    {"output", responses_output_items(msg, request_id)},
                    {"status", "completed"},
                    {"usage", json {
                        {"input_tokens", (int) toks.size()},
                        {"output_tokens", n_gen},
                        {"total_tokens", n_gen + (int) toks.size()},
                        {"input_tokens_details", json{{"cached_tokens", n_cache_hit}}},
                    }},
                };
                out["timings"] = *timings;
                res.set_content(out.dump(), "application/json");
                return;
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

        if (cr.stream) {
            const int max_tokens = cr.max_tokens;
            const bool spec_stream = use_spec;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider("text/event-stream",
                [slot, &st, &req, toks, cid, request_id, created, max_tokens, sparams, parse_tools, formatted, stops,
                 spec_stream, ctx, vocab, make_emit_gen_wall, timings, is_responses](size_t, httplib::DataSink & sink) mutable {
                    kvmem_execution_scope execution(st.execution.get());
                    StreamIo io;
                    io.on_prefill = [&st](int rows, int remaining) { st.log.prefilled(rows, remaining); };
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
                    // Responses events arrive pre-framed ("event: ..\ndata: ..\n\n");
                    // Chat Completions keeps the bare "data: <json>" framing.
                    auto send_raw = [&](const std::string & framed) -> bool {
                        if (!sink.write(framed.data(), framed.size())) {
                            io.aborted = true;
                            return false;
                        }
                        return true;
                    };
                    std::optional<ResponsesStreamOut> responses;
                    if (is_responses) {
                        responses.emplace(formatted, parse_tools, request_id);
                    }
                    const bool stream_open = is_responses
                        ? [&] {
                              for (const std::string & ev : kvmem_responses_stream_created(
                                           responses->state, request_id, st.model_name)) {
                                  if (!send_raw(ev)) {
                                      return false;
                                  }
                              }
                              return true;
                          }()
                        // 1:1 upstream initial delta: {role, content:null} (server-task.cpp to_json_oaicompat_chat)
                        // 中文：对齐上游流式首帧 delta——role=assistant 且 content=null，客户端按此初始化
                        : send(stream_choice_chunk(cid, st.model_name, created, json{{"role", "assistant"}, {"content", nullptr}}, nullptr).dump());
                    if (!stream_open) {
                        llama_driver_finish(st);
                        slot->unlock();
                        sink.done();
                        return true;
                    }
                    const auto t_turn0 = std::chrono::steady_clock::now();
                    int n_cache_hit = 0;
                    if (!llama_driver_prepare(st, toks, &io, &n_cache_hit)) {
                        if (!io.aborted) {
                            send(json{{"error", st.mm_error.empty() ? "prefill/retrieval failed" : st.mm_error}}.dump());
                            sink.write("data: [DONE]\n\n", 14);
                        }
                        llama_driver_finish(st);
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

                    std::vector<llama_token> gen;
                    std::string content;
                    StreamChatOut sco(formatted, parse_tools, request_id);
                    bool aborted = false;
                    if (spec_stream) {
                        const auto gst = kvmem_spec_generate(st.ctx, st.model, st.spec, toks, max_tokens, sparams,
                            [&](llama_token id, const std::string & piece, bool) {
                                gen.push_back(id);
                                content += piece;
                                emit_gen_wall((int) gen.size(), false);
                                if (is_responses) {
                                    for (const std::string & ev : responses->set_text(content, true)) {
                                        if (!send_raw(ev)) {
                                            break;
                                        }
                                    }
                                } else {
                                    auto deltas = sco.set_text(content, true);
                                    for (size_t i = 0; i < deltas.size(); ++i) {
                                        const json * ts = (i + 1 == deltas.size()) ? &*timings : nullptr;
                                        send(stream_choice_chunk(cid, st.model_name, created, deltas[i], nullptr, ts).dump());
                                    }
                                }
                            },
                            [&]() { return !stream_heartbeat(&io); },
                            st.active_prompt->model_pos(toks.size()) - (llama_pos) toks.size());
                        aborted = io.aborted || gst.failed;
                        st.mm_live_row = gst.n_past;
                        if (gst.failed) send(json{{"error", "speculative decode failed"}}.dump());
                    } else {
                        common_sampler * smpl = nullptr;
                        try {
                            common_params_sampling sp = sparams;
                            smpl = common_sampler_init(st.model, sp);
                        } catch (const std::exception & e) {
                            fprintf(stderr, "sampler init failed: %s\n", e.what());
                            send(json{{"error", std::string("sampler init failed: ") + e.what()}}.dump());
                            sink.write("data: [DONE]\n\n", 14);
                            llama_driver_finish(st);
                            slot->unlock();
                            sink.done();
                            return true;
                        }
                        if (!smpl) {
                            send(json{{"error", "sampler init failed"}}.dump());
                            sink.write("data: [DONE]\n\n", 14);
                            llama_driver_finish(st);
                            slot->unlock();
                            sink.done();
                            return true;
                        }
                        bool stopped = false;
                        bool hit_stop = false;
                        while ((int) gen.size() < max_tokens && !stopped) {
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
                            if (llama_driver_decode_generated(st, id, (int) toks.size() + (int) gen.size()) != 0) {
                                fprintf(stderr, "llama_decode(gen) failed\n");
                                aborted = true;
                                send(json{{"error", "decode failed"}}.dump());
                                break;
                            }
                            content += piece;
                            gen.push_back(id);
                            hit_stop = strip_stop(content, stops);
                            emit_gen_wall((int) gen.size(), false);
                            if (is_responses) {
                                for (const std::string & ev : responses->set_text(content, !hit_stop)) {
                                    if (!send_raw(ev)) {
                                        break;
                                    }
                                }
                            } else {
                                auto deltas = sco.set_text(content, !hit_stop);
                                for (size_t i = 0; i < deltas.size(); ++i) {
                                    const json * ts = (i + 1 == deltas.size()) ? &*timings : nullptr;
                                    send(stream_choice_chunk(cid, st.model_name, created, deltas[i], nullptr, ts).dump());
                                }
                            }
                            if (hit_stop) {
                                break;
                            }
                        }
                        common_sampler_free(smpl);
                        if (aborted) {
                            llama_driver_finish(st);
                            slot->unlock();
                            sink.done();
                            return true;
                        }
                        emit_gen_wall((int) gen.size(), false);
                        if (is_responses) {
                            // The non-partial parse turns accumulated text into the final diffs;
                            // the *.done sequence then closes every block the stream opened.
                            for (const std::string & ev : responses->set_text(content, false)) {
                                if (!send_raw(ev)) {
                                    break;
                                }
                            }
                            emit_gen_wall((int) gen.size());
                            llama_driver_commit(st, toks, gen);
                            for (const std::string & ev : kvmem_responses_stream_done(
                                         responses->state, responses->prev, request_id, st.model_name,
                                         (int) toks.size(), (int) gen.size(), n_cache_hit)) {
                                if (!send_raw(ev)) {
                                    break;
                                }
                            }
                            llama_driver_finish(st);
                            slot->unlock();
                            sink.done();
                            return true;
                        }
                        const bool hit_limit = (int) gen.size() >= max_tokens;
                        auto flush_deltas = sco.set_text(content, false);
                        for (size_t i = 0; i < flush_deltas.size(); ++i) {
                            const json * ts = (i + 1 == flush_deltas.size()) ? &*timings : nullptr;
                            send(stream_choice_chunk(cid, st.model_name, created, flush_deltas[i], nullptr, ts).dump());
                        }
                        const char * finish = sco.finish_reason(hit_limit);
                        kvmem_diag("KVMEM_TRACE chat_stream n_tc_delta=%d finish=%s "
                                "content_chars=%zu reasoning_chars=%zu\n",
                                sco.n_tc_delta, finish,
                                sco.prev.content.size(), sco.prev.reasoning_content.size());
                        llama_kvmem_decode_mean_flush();
                        emit_gen_wall((int) gen.size());
                        llama_driver_commit(st, toks, gen);
                        send(stream_choice_chunk(cid, st.model_name, created, json::object(), finish).dump());
                        auto usage = stream_usage_chunk(cid, st.model_name, created, (int) toks.size(), (int) gen.size(), n_cache_hit);
                        usage["timings"] = *timings;
                        send(usage.dump());
                        sink.write("data: [DONE]\n\n", 14);
                        llama_driver_finish(st);
                        slot->unlock();
                        sink.done();
                        return true;
                    }
                    if (aborted) {
                        llama_driver_finish(st);
                        slot->unlock();
                        sink.done();
                        return true;
                    }
                    emit_gen_wall((int) gen.size(), false);
                    if (is_responses) {
                        // The non-partial parse turns accumulated text into the final diffs;
                        // the *.done sequence then closes every block the stream opened.
                        for (const std::string & ev : responses->set_text(content, false)) {
                            if (!send_raw(ev)) {
                                break;
                            }
                        }
                        emit_gen_wall((int) gen.size());
                        llama_driver_commit(st, toks, gen);
                        for (const std::string & ev : kvmem_responses_stream_done(
                                     responses->state, responses->prev, request_id, st.model_name,
                                     (int) toks.size(), (int) gen.size(), n_cache_hit)) {
                            if (!send_raw(ev)) {
                                break;
                            }
                        }
                        llama_driver_finish(st);
                        slot->unlock();
                        sink.done();
                        return true;
                    }
                    const bool hit_limit = (int) gen.size() >= max_tokens;
                    auto flush_deltas = sco.set_text(content, false);
                    for (size_t i = 0; i < flush_deltas.size(); ++i) {
                        const json * ts = (i + 1 == flush_deltas.size()) ? &*timings : nullptr;
                        send(stream_choice_chunk(cid, st.model_name, created, flush_deltas[i], nullptr, ts).dump());
                    }
                    const char * finish = sco.finish_reason(hit_limit);
                    kvmem_diag("KVMEM_TRACE chat_stream n_tc_delta=%d finish=%s "
                            "content_chars=%zu reasoning_chars=%zu\n",
                            sco.n_tc_delta, finish,
                            sco.prev.content.size(), sco.prev.reasoning_content.size());
                    emit_gen_wall((int) gen.size());
                    llama_driver_commit(st, toks, gen);
                    send(stream_choice_chunk(cid, st.model_name, created, json::object(), finish).dump());
                    auto usage = stream_usage_chunk(cid, st.model_name, created, (int) toks.size(), (int) gen.size(), n_cache_hit);
                    usage["timings"] = *timings;
                    send(usage.dump());
                    sink.write("data: [DONE]\n\n", 14);
                    llama_driver_finish(st);
                    slot->unlock();
                    sink.done();
                    return true;
                });
            return;
        }

        struct request_guard {
            ServerState & st;
            ~request_guard() { llama_driver_finish(st); }
        } guard {st};
        StreamIo io;
        io.on_prefill = [&st](int rows, int remaining) { st.log.prefilled(rows, remaining); };
        io.req = &req;
        const auto t_turn0 = std::chrono::steady_clock::now();
        if (!llama_driver_prepare(st, toks, &io, &n_cache_hit)) {
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

        if (use_spec) {
            std::string content;
            std::vector<llama_token> gen;
            const kvmem_spec_gen_stats gst = kvmem_spec_generate(
                    ctx, st.model, st.spec, toks, cr.max_tokens, sparams,
                    [&](llama_token id, const std::string & piece, bool) {
                        gen.push_back(id);
                        content += piece;
                        st.progress.generated((int) gen.size());
                        st.log.generated((int) gen.size());
                    }, [&]() { return !stream_heartbeat(&io); },
                    st.active_prompt->model_pos(toks.size()) - (llama_pos) toks.size());
            if (gst.failed) {
                res.status = 500;
                res.set_content("{\"error\":\"speculative decode failed\"}", "application/json");
                return;
            }
            st.mm_live_row = gst.n_past;
            if (io.aborted) return;
            emit_gen_wall((int) gen.size());
            llama_driver_commit(st, toks, gen);
            emit_json(content, (int) gen.size(), (int) gen.size() >= cr.max_tokens);
            return;
        }

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

        int next_row = (int) toks.size();
        auto gen_one = [ctx, smpl, vocab, &st, &next_row](std::string & piece, bool & stopped, llama_token & id_out) -> bool {
            llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);
            if (llama_vocab_is_eog(vocab, id)) {
                stopped = true;
                return true;
            }
            id_out = id;
            piece = token_piece(vocab, id);
            if (llama_driver_decode_generated(st, id, next_row++) != 0) {
                fprintf(stderr, "llama_decode(gen) failed\n");
                return false;
            }
            return true;
        };

        std::string content;
        std::vector<llama_token> gen;
        bool stopped = false;
        while ((int) gen.size() < cr.max_tokens && !stopped) {
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
                return;
            }
            if (stopped) {
                break;
            }
            content += piece;
            gen.push_back(id);
            st.progress.generated((int) gen.size());
            st.log.generated((int) gen.size());
            if (strip_stop(content, stops)) {
                break;
            }
        }
        llama_kvmem_decode_mean_flush();
        emit_gen_wall((int) gen.size());
        llama_driver_commit(st, toks, gen);
        common_sampler_free(smpl);
        emit_json(content, (int) gen.size(), !stopped && (int) gen.size() >= cr.max_tokens);
    };

    svr.Post("/v1/chat/completions", handle_chat);
    svr.Post("/chat/completions", handle_chat);
    svr.Post("/v1/responses", handle_chat);
    svr.Post("/responses", handle_chat);

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
    return 0;
}
