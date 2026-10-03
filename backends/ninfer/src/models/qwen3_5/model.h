#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <optional>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct CpuVisionWeights;

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    // A Use whose matrix is Hadamard-rotated is only admitted where the execution layer rotates
    // the activation: input() refuses it, rotated_input() carries its sign vector.
    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] ops::WeightInput rotated_input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput rotated_input(WeightId id) const;

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // Overlay Vision residency only: the pinned tower groups, the pinned block that holds them and
    // the eviction pool behind the weight arena. The pool is shared mutable device state; its
    // single window is arbitrated by the one Program that executes this Model.
    [[nodiscard]] const std::optional<VisionOverlayLayout>& vision_overlay() const noexcept {
        return vision_overlay_;
    }

    [[nodiscard]] std::span<const std::byte> pinned_weights() const noexcept {
        return backing_.pinned_block();
    }

    [[nodiscard]] EvictableWeightPool* weight_pool() const noexcept {
        return backing_.weight_pool();
    }

    // CPU Vision residency only: the tower decoded to host FP32, shared with every encode session.
    [[nodiscard]] const std::shared_ptr<const CpuVisionWeights>& cpu_vision() const noexcept {
        return cpu_vision_;
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing,
          std::optional<VisionOverlayLayout> vision_overlay,
          std::shared_ptr<const CpuVisionWeights> cpu_vision);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<VisionOverlayLayout> vision_overlay_;
    std::shared_ptr<const CpuVisionWeights> cpu_vision_;
};

} // namespace ninfer::models::qwen3_5
