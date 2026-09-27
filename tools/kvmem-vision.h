#pragma once

#include "llama.h"
#include "mtmd.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct server_tokens;
struct common_chat_msg_delimiters;
struct common_chat_msg_spans;

// Native media chunks and prefix matching, with a row view for the existing sampler.
class kvmem_prompt {
public:
    explicit kvmem_prompt(const std::vector<llama_token> & tokens);
    explicit kvmem_prompt(std::shared_ptr<server_tokens> native);
    std::vector<llama_token> tokens;
    bool has_media() const;
    size_t common_prefix(const kvmem_prompt & other) const;
    llama_pos model_pos(size_t row) const;
    size_t media_end(size_t row) const;
    const mtmd_input_chunk * chunk(size_t row) const;
    std::vector<std::pair<uint32_t, uint32_t>> media_ranges() const;
    common_chat_msg_spans message_spans(const common_chat_msg_delimiters & delimiters) const;
    std::vector<std::pair<uint32_t, std::string>> media_identity() const;
    std::shared_ptr<kvmem_prompt> with_generated(const std::vector<llama_token> & gen) const;
    std::shared_ptr<kvmem_prompt> prefix(size_t rows) const;
private:
    std::shared_ptr<server_tokens> native_;
    std::map<size_t, llama_pos> position_offsets_ {{0, 0}};
};

class kvmem_vision {
public:
    kvmem_vision(llama_model * model, const std::string & path, bool gpu,
                 int min_tokens, int max_tokens, int n_threads);
    ~kvmem_vision();
    std::shared_ptr<kvmem_prompt> tokenize(const std::string & prompt, const std::vector<std::vector<uint8_t>> & files);
    int decode(llama_context * ctx, const kvmem_prompt & prompt, size_t row, int n_batch,
               const std::function<int(llama_batch)> & dispatch);
    void reset_stats() { encode_calls = 0; encode_ms = 0; }
    uint32_t encode_calls = 0;
    double encode_ms = 0;
    size_t cache_bytes() const { return cache_bytes_; }
private:
    mtmd_context * ctx_ = nullptr;
    int n_embd_ = 0;
    size_t cache_bytes_ = 0;
    struct entry { std::vector<float> embd; uint64_t used = 0; };
    std::map<std::string, entry> cache_;
    uint64_t clock_ = 0;
};

// Also rejects images explicitly when no projector is configured.
std::string kvmem_parse_media_messages(const std::string & body, bool allow_images,
                                      std::vector<std::vector<uint8_t>> & files);
