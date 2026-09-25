#pragma once

#include "chat.h"

#include <cstdint>
#include <string>
#include <vector>

// Rendering of the OpenAI Responses streaming event sequence
// (response.created / output_item.added / *_text.delta / output_item.done /
// response.completed ...).
//
// This mirrors llama.cpp's server_task_result_cmpl_partial/final
// to_json_oaicompat_resp() in llama.cpp/tools/server/server-task.cpp. That
// emitter is a member of the server task classes, and the KVMem server does not
// compile server-task.cpp (it owns its own single-slot request path), so the
// sequence is reproduced here instead of called. Keep the two in sync: clients
// drive a state machine off these event and field names.
//
// Free functions over the same common_chat_msg_diff values the Chat Completions
// stream already consumes, so this stays GPU-free and unit-testable.

struct KvMemResponsesStreamState {
    // Block state carried across deltas, exactly as upstream tracks it.
    bool reasoning_started = false;
    bool text_started      = false;
    bool fc_started        = false;
    // Item id of the function_call currently accumulating arguments.
    std::string fc_item_id;
    // Index of the item each block occupies in `response.output`, in the order
    // the items were opened. OpenAI puts this `output_index` on every event and
    // content_index on the text-bearing ones; a client that indexes its own
    // item array by it drops events whose index it never saw.
    int next_output_index  = 0;
    int reasoning_index    = -1;
    int text_index         = -1;
    int fc_index           = -1;
    // Monotonic counter, one per emitted event, as `sequence_number`.
    int64_t next_sequence  = 0;
};

// Events for one parsed delta. Returns an empty vector when the delta carries
// nothing renderable (common in partial mode before a block has settled).
// `state` is updated: it records which blocks have opened.
std::vector<std::string> kvmem_responses_stream_events(
    KvMemResponsesStreamState & state,
    const common_chat_msg_diff & diff,
    const std::string & response_id);

// Events that close the stream: the block-specific *.done events, one
// output_item.done per emitted item, then response.completed with `output` and
// `usage`. `n_prompt_tokens`/`n_cache_hit` feed usage.input_tokens_details.
std::vector<std::string> kvmem_responses_stream_done(
    KvMemResponsesStreamState & state,
    const common_chat_msg & msg,
    const std::string & response_id,
    const std::string & model,
    int n_prompt_tokens,
    int n_gen_tokens,
    int n_cache_hit);

// The opening events, emitted once before the first token.
std::vector<std::string> kvmem_responses_stream_created(
    KvMemResponsesStreamState & state,
    const std::string & response_id,
    const std::string & model);
