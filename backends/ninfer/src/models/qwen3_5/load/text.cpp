#include "models/qwen3_5/load/bindings.h"

#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>

namespace ninfer::models::qwen3_5::loading {

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query      = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key        = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    out.gate       = b.parameter(p + "attention/gate", {q, h}, {p + "mixer_input"});
    out.value      = b.parameter(p + "attention/value", {k, h}, {p + "mixer_input"});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    return out;
}

DenseWeights bind_dense(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                        const std::string& p, bool draft) {
    const auto input = p + (draft ? "mlp_input" : "ffn_input");
    return {b.parameter(p + "mlp/gate", {intermediate, h}, {input}),
            b.parameter(p + "mlp/up", {intermediate, h}, {input}),
            b.parameter(p + "mlp/down", {h, intermediate},
                        {p + (draft ? "mlp_product" : "mlp/product")})};
}

namespace {

GdnWeights bind_gdn(Bindings& b, const TextConfig& text, const std::string& p) {
    const auto& g    = text.gdn.value();
    const auto h     = text.hidden_size;
    const auto k     = g.key_width();
    const auto v     = g.value_width();
    const auto heads = g.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.parameter(p + "gdn/query", {k, h}, {p + "mixer_input"});
    out.key          = b.parameter(p + "gdn/key", {k, h}, {p + "mixer_input"});
    out.value        = b.parameter(p + "gdn/value", {v, h}, {p + "mixer_input"});
    out.z            = b.parameter(p + "gdn/z", {v, h}, {p + "mixer_input"});
    out.a_projection = b.parameter(p + "gdn/a_projection", {heads, h}, {p + "mixer_input"});
    out.b_projection = b.parameter(p + "gdn/b_projection", {heads, h}, {p + "mixer_input"});
    out.a_log        = b.direct(p + "gdn/a_log", {heads}, QType::FP32);
    out.dt_bias      = b.direct(p + "gdn/dt_bias", {heads}, QType::FP32);
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, v}, {p + "gdn/gated_output"});
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& moe   = std::get<MoeConfig>(config.ffn);
    const auto p      = prefix + "moe/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_score = b.parameter(p + "shared_score", {1, h}, {input});
    out.experts.reserve(moe.num_experts);
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        out.experts.push_back({b.parameter(ep + "gate", {ir, h}, {input}),
                               b.parameter(ep + "up", {ir, h}, {input}),
                               b.parameter(ep + "down", {h, ir}, {ep + "product"})});
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

// A stage owns its layers whole: norms, the attention or GDN projections, and the FFN or experts all
// live on the stage's device, next to the KV cache and recurrent state that layer reads and writes.
// That is what lets the stage run its layers with the same kernels as a single GPU.
void place_layer(Bindings& b, const BlockWeights& block, std::size_t rank) {
    for (const WeightId id : layer_weights(block)) { b.place(id, rank); }
}

} // namespace

std::vector<WeightId> layer_weights(const BlockWeights& block) {
    std::vector<WeightId> out{block.input_norm, block.post_attention_norm};
    if (const auto* attention = std::get_if<AttentionWeights>(&block.mixer)) {
        out.insert(out.end(), {attention->query, attention->key, attention->gate, attention->value,
                               attention->query_norm, attention->key_norm, attention->output});
    } else {
        const auto& gdn = std::get<GdnWeights>(block.mixer);
        out.insert(out.end(), {gdn.query, gdn.key, gdn.value, gdn.z, gdn.a_projection,
                               gdn.b_projection, gdn.a_log, gdn.dt_bias, gdn.convolution, gdn.norm,
                               gdn.output});
    }
    const auto add_dense = [&](const DenseWeights& dense) {
        out.insert(out.end(), {dense.gate, dense.up, dense.down});
    };
    if (const auto* dense = std::get_if<DenseWeights>(&block.ffn)) {
        add_dense(*dense);
        return out;
    }
    const auto& moe = std::get<MoeWeights>(block.ffn);
    out.push_back(moe.router);
    out.push_back(moe.shared_score);
    for (const auto& expert : moe.experts) { add_dense(expert); }
    add_dense(moe.shared);
    return out;
}

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer) {
    BlockWeights out;
    out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
    out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
        out.ffn = bind_dense(b, config.hidden_size, dense->intermediate_size, p);
    } else {
        out.ffn = bind_moe(b, config, p);
    }
    return out;
}

StagePlan plan_stage_plan(std::uint32_t layers, const LoadOptions& options) {
    if (options.ranks <= 1) {
        if (!options.stage_layers.empty()) {
            throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
        }
        return StagePlan(layers);
    }
    if (options.ranks > layers) {
        throw std::invalid_argument("--devices asks for more devices than the model has layers");
    }
    // The layer counts come from the caller when it has measured device memory (the engine solves
    // them against each card's free bytes), or from `--stage-layers`. Without either, equal counts:
    // the sensible default while layers are near-uniform in size.
    if (!options.stage_layers.empty()) {
        if (options.stage_layers.size() != options.ranks) {
            throw std::invalid_argument("--stage-layers lists " +
                                        std::to_string(options.stage_layers.size()) +
                                        " counts but --devices names " +
                                        std::to_string(options.ranks) + " devices");
        }
        return StagePlan::from_layer_counts(layers, options.stage_layers);
    }
    return StagePlan::even(layers, options.ranks);
}

TextWeights bind_text(Bindings& b, const TextConfig& config, const LoadOptions& options,
                      StagePlan stages) {
    TextWeights out;
    out.stages = std::move(stages);
    out.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    std::vector<std::string> head_inputs{"text/final_hidden"};
    if (options.speculative != SpeculativeBackend::None && !options.proposal_enabled()) {
        head_inputs.push_back(std::string(options.speculative_component()) + "/final_hidden");
    }
    out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                  std::move(head_inputs));
    out.final_norm  = b.direct("text/final_norm", {config.hidden_size});
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        out.layers.push_back(
            bind_block(b, config, "text/layers/" + std::to_string(i) + "/", config.layer_types[i]));
        // A layer lives on its stage. The embedding, final norm and head stay on rank 0 with the
        // round state, sampling and prefill buffers that reach them directly: the last stage sends
        // the residual back, one extra hop per forward pass.
        // One stage is the single-GPU route and binds exactly what it always bound.
        if (!out.stages.single_stage()) {
            place_layer(b, out.layers.back(), out.stages.placement(i).stage);
        }
    }
    return out;
}

ProposalWeights bind_proposal(Bindings& b, const artifact::Proposal& proposal,
                              const TextConfig& target, const LoadOptions& options,
                              std::uint32_t public_tokens) {
    const auto rows = proposal.indexed ? proposal.rows : target.vocab_size;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        (proposal.indexed && rows > public_tokens)) {
        throw artifact::ArtifactError("proposal rows exceed the output domain");
    }
    ProposalWeights out;
    out.rows = static_cast<std::uint32_t>(rows);
    out.head = b.parameter("proposal/head", {rows, target.hidden_size},
                           {std::string(options.speculative_component()) + "/final_hidden"});
    if (proposal.indexed) {
        out.token_ids = b.direct("proposal/token_ids", {rows}, QType::INT32);
        out.global_token_ids =
            b.binder.values(b.at(*out.token_ids).reference.binding, QType::INT32).integers();
        std::set<std::int32_t> unique;
        for (const auto id : out.global_token_ids) {
            if (id < 0 || std::uint32_t(id) >= public_tokens || !unique.insert(id).second) {
                throw artifact::ArtifactError(
                    "proposal token IDs must be unique and in the public domain");
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
