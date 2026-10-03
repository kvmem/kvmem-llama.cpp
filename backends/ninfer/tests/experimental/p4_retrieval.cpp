#include "ninfer/engine.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("artifact bf16|int8 eager|graph output");
        ninfer::EngineOptions o;
        o.artifact_path = argv[1];
        o.max_context = 16384;
        o.device = 0;
        o.max_concurrency = 1;
        o.prefill_chunk = 128;
        o.kvmem.selected_tokens = 384;
        o.kvmem.reserve_tokens = 128;
        o.kvmem.host_bytes = 2ULL << 30;
        o.kvmem.verify_transfers = true;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        o.kv_cache = std::string(argv[2]) == "bf16"
            ? ninfer::KvCacheStorage::BFloat16 : ninfer::KvCacheStorage::Int8Group64;
        o.context_cache.enabled = false;
        o.use_cuda_graph = std::string(argv[3]) == "graph";
        o.device_profile = "off";
        ninfer::Engine engine(o);
        std::ofstream output(argv[4]);
        for (int turn = 0; turn < 3; ++turn) {
            ninfer::PromptInput prompt;
            const auto add = [&](ninfer::ChatRole role, const std::string& text) {
                ninfer::ChatMessage message;
                message.role = role;
                message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = text});
                prompt.messages.push_back(std::move(message));
            };
            add(ninfer::ChatRole::System, "Use the archive to answer the final question.");
            for (int record = 0; record < 24; ++record) {
                std::string paragraph = "Archive entry " + std::to_string(record) + ". ";
                for (int repeat = 0; repeat < 8; ++repeat) {
                    paragraph += record < 4
                        ? "The violet observatory studies pulsars, cosmic rays and distant galaxies. "
                        : "The coastal office files invoices, transport schedules and warehouse inventories. ";
                }
                add(ninfer::ChatRole::User, paragraph);
                add(ninfer::ChatRole::Assistant, "The archive entry is recorded.");
            }
            const std::string query = turn == 1
                ? "Explain coastal warehouse inventories and transport schedules."
                : "Explain the violet observatory, its pulsars, cosmic rays and distant galaxies.";
            add(ninfer::ChatRole::User, query);
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 130;
            request.execution.sampling.temperature = 0;
            request.execution.sampling.seed = 42;
            request.execution.allow_prefix_reuse = false;
            request.stop.include_model_defaults = false;
            const auto result = engine.submit(engine.prepare(std::move(prompt)), request).wait();
            if (result.generated_token_ids.size() != 130 || !engine.is_available())
                throw std::runtime_error("retrieval request did not complete");
            output << turn;
            for (auto token : result.generated_token_ids) output << ' ' << token;
            output << '\n';
            std::cout << "P4_RETRIEVAL turn=" << turn << " output=" << result.generated_token_ids.size() << '\n';
        }
        std::cout << "P4_RETRIEVAL PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
