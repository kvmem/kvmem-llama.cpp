#pragma once

#include "kvmem-chat-sampling.h"
#include "httplib.h"
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

inline std::filesystem::path kvmem_executable_path(const char * argv0) {
#ifdef _WIN32
    std::wstring name(32768, L'\0');
    const DWORD size = GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
    if (size > 0 && size < name.size()) {
        name.resize(size);
        return std::filesystem::path(name);
    }
#else
    std::error_code ec;
    auto binary = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return binary;
#endif
    return std::filesystem::absolute(argv0);
}

inline bool kvmem_output_limit(const nlohmann::json & body, int limit, int & value, std::string & error) {
    // An OpenAI-compatible endpoint should be liberal in what it accepts. Clients
    // routinely ask for more output than this deployment can serve, and some send
    // JSON `null` or `0` for "let the server decide" (DeepSeek Harness sends
    // max_completion_tokens). Hard-rejecting those with a 400 breaks clients that
    // work fine against llama-server, which clamps instead -- and the clamp at the
    // bottom of this function is the real authority anyway, so rejecting values it
    // would have clamped was self-contradictory. Only a value we cannot interpret
    // at all (a string) is still an error.
    const char * key = body.contains("max_tokens") ? "max_tokens" : "max_completion_tokens";
    if (body.contains(key)) {
        const auto & n = body[key];
        if (n.is_number()) {
            const double requested = n.get<double>();
            if (requested > (double) limit) {
                fprintf(stderr, "requested %s=%.0f exceeds the generation limit %d; clamping\n",
                        key, requested, limit);
                value = limit;
            } else if (requested >= 1.0) {
                value = (int) requested;
            }
            // -1, 0 and negative values mean "server default": leave `value` at
            // whatever the caller already resolved.
        } else if (!n.is_null()) {
            error = std::string(key) + " must be a number or null";
            return false;
        }
    }
    if (value < 1) value = limit;
    value = std::min(value, limit);
    return true;
}

inline nlohmann::json kvmem_ui_sampling(bool thinking, const nlohmann::json & overrides) {
    auto sp = kvmem_chat_sampling_defaults(thinking);
    std::string error;
    if (!kvmem_chat_sampling_override(overrides, sp, error)) throw std::runtime_error(error);
    return {{"temperature", sp.temp}, {"top_p", sp.top_p}, {"top_k", sp.top_k}, {"min_p", sp.min_p},
            {"presence_penalty", sp.penalty_present}, {"frequency_penalty", sp.penalty_freq},
            {"repeat_penalty", sp.penalty_repeat}};
}

inline bool kvmem_mount_ui(httplib::Server & server, const std::string & requested, bool disabled, const char * argv0) {
    namespace fs = std::filesystem;
    if (disabled) return true;
    const auto binary = kvmem_executable_path(argv0);
    const auto directory = requested.empty() ? binary.parent_path().parent_path() / "share/kvmem/ui" : fs::path(requested);
    if (!fs::is_regular_file(directory / "index.html")) {
        if (requested.empty()) return true;
        fprintf(stderr, "UI directory has no index.html: %s\n", directory.string().c_str());
        return false;
    }
    if (!server.set_mount_point("/", directory.string())) return false;
    server.set_file_request_handler([](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Cache-Control", req.path.find("/_app/immutable/") == 0 ?
                       "public, max-age=31536000, immutable" : "no-cache");
    });
    fprintf(stderr, "KVMEM_UI directory=%s\n", directory.string().c_str());
    return true;
}
