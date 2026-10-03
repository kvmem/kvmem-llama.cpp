#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing,
             std::optional<VisionOverlayLayout> vision_overlay,
             std::shared_ptr<const CpuVisionWeights> cpu_vision)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)),
      info_(std::move(info)), vision_overlay_(std::move(vision_overlay)),
      cpu_vision_(std::move(cpu_vision)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    if (use.hadamard_signs) {
        throw std::invalid_argument(parameter.name + "@" + use.input +
                                    ": a Hadamard-rotated matrix is not supported at this input");
    }
    ops::WeightInput result{parameter.view, use.policy, use.activation_input_divisor};
    if (use.input_columns) {
        const auto& columns  = weight(*use.input_columns).view;
        result.input_columns = weight_tensor(columns, {static_cast<std::int32_t>(columns.shape[0])});
    }
    return result;
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

ops::WeightInput Model::rotated_input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    ops::WeightInput result{parameter.view, use.policy, use.activation_input_divisor};
    if (use.hadamard_signs) {
        const auto& signs = weight(*use.hadamard_signs).view;
        if (signs.shape.size() != 1 || parameter.view.shape.size() != 2 ||
            signs.shape[0] != parameter.view.shape[1]) {
            throw std::invalid_argument(parameter.name + "@" + use.input +
                                        ": Hadamard signs must cover the input width");
        }
        result.hadamard_signs = weight_tensor(signs, {static_cast<std::int32_t>(signs.shape[0])});
    }
    if (use.input_columns) {
        const auto& columns  = weight(*use.input_columns).view;
        result.input_columns = weight_tensor(columns, {static_cast<std::int32_t>(columns.shape[0])});
    }
    return result;
}

ops::WeightInput Model::rotated_input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return rotated_input(WeightUseId{id, 0});
}

} // namespace ninfer::models::qwen3_5
