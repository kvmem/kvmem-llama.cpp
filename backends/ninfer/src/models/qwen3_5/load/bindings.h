#pragma once

#include "artifact/binder.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"

#include <map>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::loading {

[[nodiscard]] FrontendResources bind_resources(artifact::Binder& binder, const Config& config);

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {},
                                     artifact::Residency residency = artifact::Residency::Device);
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format                  = QType::BF16,
                                  artifact::Residency residency = artifact::Residency::Device);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    // Requests that every stored object behind a parameter be materialized in `target` instead
    // (see artifact/transcode.h). Returns false, leaving the objects alone, when they already
    // store `target`; throws std::invalid_argument when they are not transcodable Q8.
    bool transcode(WeightId id, QType target, std::string_view option);
    // Places every stored object behind a device parameter in the evictable device tail.
    void evict(WeightId id, std::uint32_t rank);
    // Materializes every stored object behind a device parameter on another pipeline rank's
    // device. Rank 0 is the default and this is then a no-op, so the single-device path is
    // untouched.
    void place(WeightId id, std::size_t rank);

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;

private:
    // One device weight per distinct sign binding, however many Uses name it.
    WeightId hadamard_signs(const artifact::Binding& binding, std::uint64_t width,
                            const std::string& use);
    // One device weight per distinct input-gather binding, the same way.
    WeightId input_columns(const artifact::Binding& binding, std::uint64_t width,
                           const std::string& use);

    std::map<std::string, WeightId, std::less<>> parameters_;
    std::map<std::string, WeightId, std::less<>> signs_;
    std::map<std::string, WeightId, std::less<>> columns_;
};

[[nodiscard]] AttentionWeights bind_attention(Bindings& bindings, const TextConfig& config,
                                              const std::string& prefix);
[[nodiscard]] DenseWeights bind_dense(Bindings& bindings, std::uint64_t hidden,
                                      std::uint64_t intermediate, const std::string& prefix,
                                      bool draft = false);
[[nodiscard]] BlockWeights bind_block(Bindings& bindings, const TextConfig& config,
                                      const std::string& prefix, MixerKind mixer);
// `stages` decides which device holds each whole layer; its identity form (one stage) binds exactly
// what a single-GPU load always bound.
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config,
                                    const LoadOptions& options, StagePlan stages);
// The stage plan `options.ranks` devices ask for over `layers` layers: `options.stage_layers` when
// given, equal counts otherwise. Throws when the model cannot be served by that many devices.
[[nodiscard]] StagePlan plan_stage_plan(std::uint32_t layers, const LoadOptions& options);
// Every parameter a whole layer owns: what moves with it when the layer is placed on a stage.
[[nodiscard]] std::vector<WeightId> layer_weights(const BlockWeights& block);
// Pinned residency keeps the tower in the page-locked Host block, one contiguous group per stage
// (patch/position embedding, each layer, merger) in binding order.
[[nodiscard]] VisionWeights bind_vision(Bindings& bindings, const VisionConfig& config,
                                        const TextConfig& target, artifact::Residency residency);
[[nodiscard]] MtpWeights bind_mtp(Bindings& bindings, const TextConfig& config,
                                  const TextWeights& target);
[[nodiscard]] DraftWeights bind_draft(Bindings& bindings, const DraftConfig& config,
                                      const TextConfig& target, const TextWeights& weights,
                                      const std::string& component);
void bind_dflash2(Bindings& bindings, DraftWeights& weights, const DraftConfig& config,
                  const TextConfig& target);
[[nodiscard]] ProposalWeights bind_proposal(Bindings& bindings, const artifact::Proposal& proposal,
                                            const TextConfig& target, const LoadOptions& options,
                                            std::uint32_t public_tokens);
// --lm-head-q4/q6, --embedding-q4/q6 and --mtp-experts-q4: validate the combination and request
// the load-time transcodes on the selected parameters.
void apply_storage_trades(Bindings& bindings, const Config& config, const ModelWeights& weights,
                          const LoadOptions& options);
// --vision-residency overlay: rank the device weights an exclusive Vision window may borrow.
// `mtp_parameters` and `draft_parameters` are the half-open WeightId index ranges registered by
// bind_mtp and bind_draft.
void apply_vision_overlay_placement(Bindings& bindings, const ModelWeights& weights,
                                    std::pair<std::size_t, std::size_t> mtp_parameters,
                                    std::pair<std::size_t, std::size_t> draft_parameters);
// Byte ranges of the Vision groups inside the materialized pinned block.
[[nodiscard]] VisionOverlayLayout vision_overlay_layout(const VisionWeights& weights,
                                                        std::span<const BoundWeight> bound,
                                                        std::span<const std::byte> pinned_block);
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized);

} // namespace ninfer::models::qwen3_5::loading
