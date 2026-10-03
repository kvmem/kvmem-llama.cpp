#include "ninfer/engine.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// No HTTP shim: every case enters the public raw-token preparation, scheduler,
// admission, Program prefill, ordinary decode, and request retirement paths.
int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7) {
            std::cerr << "artifact bf16|int8 native|window graph|eager output-file [host-failure|long]\n";
            return 2;
        }
        ninfer::EngineOptions o;
        o.artifact_path = argv[1];
        o.device = 0;
        const std::string scenario = argc == 7 ? argv[6] : "boundaries";
        o.max_context = scenario == "long" ? 32768 + 256 : 2048;
        const bool window = std::string(argv[3]) == "window";
        if (window) {
            o.kvmem.selected_tokens = 128;
            o.kvmem.reserve_tokens = 128;
            o.kvmem.host_bytes = (scenario == "long" ? 4ULL : 1ULL) * 1024ULL * 1024ULL * 1024ULL;
            o.kvmem.verify_transfers = true;
            if (scenario == "host-failure") { o.kvmem.host_bytes = 1; }
        }
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(window ? 256 : o.max_context);
        o.max_concurrency = 1;
        o.prefill_chunk = 128;
        o.kv_cache = std::string(argv[2]) == "bf16"
            ? ninfer::KvCacheStorage::BFloat16 : ninfer::KvCacheStorage::Int8Group64;
        o.context_cache.enabled = false;
        o.use_cuda_graph = std::string(argv[4]) == "graph";
        o.device_profile = "off";
        ninfer::Engine engine(o);
        const auto memory = engine.memory_summary();
        if (memory.kv_capacity != (window ? 256U : o.max_context)) {
            throw std::runtime_error("Engine resolved an unexpected physical KV capacity");
        }
        std::cout << "P1_MEMORY kv_tokens=" << memory.kv_capacity
                  << " kv_pages=" << memory.kv_capacity_page_groups
                  << " workspace=" << memory.workspace.capacity_bytes << '\n';
        const auto seed = engine.tokenize_text("The archive contains rivers, mountains, and engineering. ");
        if (seed.empty()) { throw std::runtime_error("empty tokenizer result"); }
        std::ofstream out(argv[5]);
        const std::vector<int> lengths = scenario == "long" ? std::vector<int>{8192, 32768}
            : scenario == "host-failure" ? std::vector<int>{1025, 63, 1025, 65}
            : std::vector<int>{63, 64, 65, 191, 255, 256, 257, 1023, 1024, 1025};
        for (int length : lengths) {
            const int outputs = length > 256 ? 130 : 8;
            std::vector<ninfer::TokenId> prompt;
            for (int i = 0; i < length; ++i) { prompt.push_back(seed[i % seed.size()]); }
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = outputs;
            request.execution.sampling.temperature = 0.0F;
            request.execution.sampling.seed = 42;
            request.execution.allow_prefix_reuse = false;
            request.stop.include_model_defaults = false;
            const auto start = std::chrono::steady_clock::now();
            auto handle = engine.submit(engine.prepare_tokens(prompt, false), request);
            ninfer::GenerationResult result;
            bool budget_failure = false;
            try { result = handle.wait(); }
            catch (const ninfer::RequestError& error) {
                budget_failure = scenario == "host-failure" && length > 256 &&
                    error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded &&
                    std::string(error.what()).find("budget") != std::string::npos;
                if (!budget_failure) { throw; }
            }
            if (scenario == "host-failure" && length > 256) {
                if (!budget_failure || !engine.is_available()) {
                    throw std::runtime_error("Host exhaustion did not fail cleanly");
                }
                std::cout << "P4_HOST_LIMIT rejected; engine remains available\n";
                continue;
            }
            if (result.generated_token_ids.size() != static_cast<std::size_t>(outputs)) {
                throw std::runtime_error("request did not reach its output budget");
            }
            out << length;
            for (auto id : result.generated_token_ids) { out << ' ' << id; }
            out << '\n';
            out.flush();
            std::cout << "P1_ENGINE length=" << length << " output=" << outputs
                      << " seconds=" << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start).count() << '\n';
        }
        std::cout << "P1_ENGINE PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "P1_ENGINE FAIL " << e.what() << '\n';
        return 1;
    }
}
