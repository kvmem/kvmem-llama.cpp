#include "ninfer/engine.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void add(ninfer::PromptInput& p, ninfer::ChatRole role, const std::string& text) {
    ninfer::ChatMessage m; m.role = role;
    m.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = text});
    p.messages.push_back(std::move(m));
}
ninfer::PromptInput archive(int repeats) {
    ninfer::PromptInput p; p.options.enable_thinking = false;
    add(p, ninfer::ChatRole::System, "Answer questions using the archive. Return only the requested code.");
    std::string body;
    const std::string filler = "The coastal office files invoices, transport schedules and warehouse inventories. ";
    for (int i = 0; i < 80; ++i) body += filler;
    for (int i = 0; i < 4; ++i) body +=
        "The violet observatory has primary access code AZURE-7319 and backup access code COPPER-6248. ";
    for (int i = 0; i < repeats; ++i) body += filler;
    add(p, ninfer::ChatRole::User, body);
    add(p, ninfer::ChatRole::Assistant, "The archive is recorded.");
    add(p, ninfer::ChatRole::User, "What is the primary access code of the violet observatory?");
    return p;
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6) throw std::invalid_argument("artifact context-target output [drafts] [int8|rk8v4]");
        const auto drafts = argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 0U;
        const auto target = static_cast<std::uint32_t>(std::stoul(argv[2]));
        require(target >= 8192 && target <= 262144, "unsupported context target");
        ninfer::EngineOptions o;
        o.artifact_path = argv[1]; o.device = 0; o.max_concurrency = 1;
        o.max_context = 262144; o.prefill_chunk = 512; o.device_profile = "off";
        o.kvmem.selected_tokens = 2048; o.kvmem.reserve_tokens = 512;
        o.kvmem.host_bytes = 12ULL << 30; o.kvmem.verify_transfers = true;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2560);
        o.kv_cache = ninfer::KvCacheStorage::Int8Group64; o.context_cache.enabled = false;
        if (argc > 5) {
            if (std::string(argv[5]) == "rk8v4") o.kv_cache = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
            else require(std::string(argv[5]) == "int8", "unknown KV format");
        }
        if (drafts) {
            o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            o.speculative.draft_tokens = drafts;
        }
        ninfer::Engine engine(o);
        const auto base = engine.prepare(archive(0)).summary().prompt_tokens;
        const auto hundred = engine.prepare(archive(100)).summary().prompt_tokens;
        const auto unit = (hundred - base) / 100;
        require(unit > 0, "corpus calibration failed");
        const auto goal = target - 512;
        int repeats = static_cast<int>((goal - base) / unit);
        auto prompt = archive(repeats);
        auto prepared = engine.prepare(prompt);
        require(prepared.summary().prompt_tokens <= goal &&
                prepared.summary().prompt_tokens + 64 >= goal, "corpus missed target length");
        std::ofstream output(argv[3]);
        const auto run = [&](const char* name, const ninfer::PromptInput& input,
                             ninfer::RequestOptions options) {
            const auto start = std::chrono::steady_clock::now();
            const auto result = engine.submit(engine.prepare(input), options).wait();
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            require(engine.is_available(), "Engine unavailable after request");
            if (drafts && result.generated_token_ids.size() > 1)
                require(result.speculative.drafted_tokens > 0, "MTP did not execute");
            output << name << " prompt=" << result.prompt.prompt_tokens << " reused="
                   << result.reused_prompt_tokens << " generated=" << result.generated_token_ids.size()
                   << " seconds=" << seconds << " content=" << result.content << '\n';
            output << "tokens"; for (auto id : result.generated_token_ids) output << ' ' << id;
            output << '\n'; output.flush();
            std::cout << "P4_QUALITY case=" << name << " prompt=" << result.prompt.prompt_tokens
                      << " reused=" << result.reused_prompt_tokens << " seconds=" << seconds
                      << " drafted=" << result.speculative.drafted_tokens
                      << " accepted=" << result.speculative.accepted_tokens
                      << " content=" << result.content << std::endl;
            const auto memory = engine.memory_summary();
            std::cout << "P6_RESOURCE host=" << memory.kvmem.host_payload_bytes
                      << " checkpoint=" << memory.kvmem.checkpoint_bytes
                      << " device_allocated=" << memory.cuda_residency.device_allocated_bytes
                      << " workspace_peak=" << memory.workspace_logical_peak_bytes << std::endl;
            require(memory.kvmem.host_payload_bytes <= o.kvmem.host_bytes, "long history exceeded H");
            require(memory.kvmem.resident_pages == 0 && memory.kvmem.mtp_resident_pages == 0,
                    "long request leaked active KV");
            return result;
        };
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 64;
        request.execution.sampling.temperature = 0;
        request.execution.sampling.seed = 42;
        request.execution.allow_prefix_reuse = true;
        const auto first = run("primary", prompt, request);
        require(first.content.find("AZURE-7319") != std::string::npos, "early primary needle missed");
        auto follow = prompt;
        add(follow, ninfer::ChatRole::Assistant, first.content);
        add(follow, ninfer::ChatRole::User, "What is the backup access code of the violet observatory?");
        const auto second = run("backup-followup", follow, request);
        require(second.reused_prompt_tokens > 0, "followup did not restore history");
        require(second.content.find("COPPER-6248") != std::string::npos, "early backup needle missed");
        if (target == 8192) {
            auto fresh_request = request; fresh_request.execution.allow_prefix_reuse = false;
            const auto fresh_follow = run("backup-fresh", follow, fresh_request);
            require(fresh_follow.content.find("COPPER-6248") != std::string::npos,
                    "full-recompute backup needle missed");
            // One output token leaves no decode work to hide an underestimated
            // replay service budget. A long query also crosses chunk boundaries.
            auto long_query = follow;
            for (int i = 0; i < 100; ++i) long_query.messages.back().parts[0].text += " Read the archive carefully.";
            auto one = request; one.execution.requested_output_tokens = 1;
            one.stop.include_model_defaults = false;
            const auto boundary = run("query-boundary-one-output", long_query, one);
            require(boundary.generated_token_ids.size() == 1, "one-token replay failed");
            auto creative = follow;
            creative.messages.front().parts[0].text = "Answer helpfully. Write imaginative stories when asked.";
            creative.messages.back().parts[0].text = "Write a short imaginative story about the violet observatory.";
            auto sampled = request; sampled.execution.requested_output_tokens = 32;
            sampled.stop.include_model_defaults = false;
            sampled.execution.sampling.temperature = 0.85F;
            sampled.execution.sampling.top_k = 20; sampled.execution.sampling.top_p = 0.9F;
            sampled.execution.sampling.presence_penalty = 0.15F;
            sampled.execution.sampling.frequency_penalty = 0.1F;
            const auto a = run("sampling-a", creative, sampled);
            const auto b = run("sampling-repeat", creative, sampled);
            require(a.generated_token_ids == b.generated_token_ids, "sampling replay changed seeded output");
            sampled.execution.sampling.seed = 73;
            const auto c = run("sampling-other-seed", creative, sampled);
            require(a.generated_token_ids != c.generated_token_ids, "sampling seed had no observable effect");
        }
        std::cout << "P4_QUALITY PASS target=" << target << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
