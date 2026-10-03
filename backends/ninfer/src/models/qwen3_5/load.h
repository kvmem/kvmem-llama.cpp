#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/types.h"

#include <filesystem>
#include <memory>
#include <span>

namespace ninfer::artifact {
class Reader;
struct ParameterReference;
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_5 {

// Cold load plan borrows its Reader until materialization. Its selected Host bytes and parsed
// resources already have owners; no Program or device allocation is needed for plan_load.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
// How many layers each of `free_bytes.size()` pipeline stages should own so that the most KV cache
// fits on every device at once, given the bytes free on each. Sized from the artifact's stored
// layers; the head stage (rank 0) also carries the embedding, head and round buffers. The default
// when `--stage-layers` is not given.
//
// The per-device KV and recurrent-state costs use the same layouts startup planning reserves, so
// they follow the runtime choices that size them.
struct StageSizing {
    // Stored K/V format of the Paged KV cache.
    KvCacheStorage kv_storage = KvCacheStorage::BFloat16;
    // Recurrent-state slots each Linear Attention layer holds: concurrent requests plus the device
    // state slots the context cache keeps.
    std::uint32_t state_slots = 1;
};
[[nodiscard]] std::vector<std::uint32_t> default_stage_layers(const artifact::Reader& reader,
                                                              LoadOptions options,
                                                              const StageSizing& sizing,
                                                              std::span<const std::uint64_t> free_bytes);
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::qwen3_5
