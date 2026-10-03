#include "runtime/engine/model_instance.h"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    ninfer::EngineOptions base;
    base.max_context = 2048;
    base.kvmem = {128, 128, 1024ULL * 1024ULL * 1024ULL};
    base.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
    base.prefill_chunk = 128;
    base.context_cache.enabled = false;
    try {
        const auto valid = ninfer::runtime::normalize_engine_options(base);
        if (valid.kvmem.device_tokens() != 256 || valid.max_context != 2048) {
            throw std::runtime_error("normalization changed logical/physical limits");
        }
        for (const auto drafts : {1U, 2U, 3U, 4U}) {
            auto mtp = base;
            mtp.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            mtp.speculative.draft_tokens = drafts;
            const auto accepted = ninfer::runtime::normalize_engine_options(mtp);
            if (accepted.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
                accepted.speculative.draft_tokens != drafts) {
                throw std::runtime_error("KVMem normalization silently disabled MTP");
            }
        }
        for (const auto width : {15U, 31U, 63U}) {
            auto copy = base;
            copy.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            copy.speculative.draft_tokens = 3;
            copy.speculative.ngram_draft_tokens = width;
            const auto accepted = ninfer::runtime::normalize_engine_options(copy);
            if (accepted.speculative.ngram_draft_tokens != width)
                throw std::runtime_error("KVMem silently disabled ngram");
        }
        for (const auto concurrency : {2U, 4U}) {
            auto concurrent = base;
            concurrent.max_concurrency = concurrency;
            concurrent.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * concurrency);
            const auto accepted = ninfer::runtime::normalize_engine_options(concurrent);
            if (accepted.max_concurrency != concurrency || accepted.kvmem.device_tokens() != 256) {
                throw std::runtime_error("KVMem normalization changed per-request working capacity");
            }
        }
        auto vision = base;
        vision.enable_vision = true;
        vision.vision_residency = ninfer::VisionResidency::Resident;
        (void)ninfer::runtime::normalize_engine_options(vision);
        auto rotated = base;
        rotated.kv_cache = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
        (void)ninfer::runtime::normalize_engine_options(rotated);
        auto disk = base;
        disk.device_profile = "off";
        disk.kvmem.disk_path = "snapshot-test";
        disk.kvmem.disk_bytes = 1ULL << 30;
        (void)ninfer::runtime::normalize_engine_options(disk);
        const std::vector<std::function<void(ninfer::EngineOptions&)>> invalid{
            [](auto& o) { o.max_concurrency = 2; },
            [](auto& o) { o.max_concurrency = 5;
                         o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1280); },
            [](auto& o) { o.context_cache.enabled = true; },
            [](auto& o) { o.kvmem.disk_path = "missing-quota"; },
            [](auto& o) { o.kvmem.disk_bytes = 1ULL << 30; },
            [](auto& o) { o.kvmem.disk_path = "unfixed-profile"; o.kvmem.disk_bytes = 1ULL << 30; },
            [](auto& o) { o.kvmem.disk_path = "concurrent"; o.kvmem.disk_bytes = 1ULL << 30;
                         o.device_profile = "off"; o.max_concurrency = 2;
                         o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.kv_cache = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
                         o.max_concurrency = 2; o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.enable_vision = true; o.vision_residency = ninfer::VisionResidency::Overlay; },
            [](auto& o) { o.enable_vision = true; o.max_concurrency = 2;
                         o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.concurrent_prefill = true; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp; },
            [](auto& o) { o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.kvmem.selected_tokens = 127; },
            [](auto& o) { o.prefill_chunk = 256; },
            [](auto& o) { o.devices = {0, 1}; },
            [](auto& o) { o.purpose = ninfer::EnginePurpose::CausalScoring; },
            [](auto& o) { o.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256; },
            [](auto& o) { o.cuda_memory_policy = ninfer::CudaMemoryPolicy::Mixed; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 5; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3;
                         o.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.ngram_draft_tokens = 64; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.ngram_draft_tokens = 15;
                         o.enable_vision = true; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.ngram_draft_tokens = 15;
                         o.max_concurrency = 2; o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.ngram_draft_tokens = 63;
                         o.kvmem.selected_tokens = 192; o.kvmem.reserve_tokens = 64; o.prefill_chunk = 64; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.mtp_attention_window = 128; },
        };
        for (const auto& change : invalid) {
            auto bad = base;
            change(bad);
            bool rejected = false;
            try { (void)ninfer::runtime::normalize_engine_options(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) { throw std::runtime_error("unsupported KVMem combination accepted"); }
        }
        std::cout << "P1_OPTIONS PASS: ordinary/MTP1..4 and " << invalid.size()
                  << " rejected combinations\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "P1_OPTIONS FAIL " << e.what() << '\n';
        return 1;
    }
}
