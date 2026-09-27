#pragma once

#include "kvmem-server-options.h"
#include <cstdio>
#include <map>
#include <set>
#ifndef _WIN32
extern char ** environ;
#endif

inline char ** kvmem_process_environment() {
#ifdef _WIN32
    return _environ;
#else
    return environ;
#endif
}

struct kvmem_env_option {
    const char * env;
    const char * flag;
    const char * negative = nullptr;
};

// Names follow the pinned llama.cpp common/arg.cpp, not inferred flag names.
inline const std::vector<kvmem_env_option> & kvmem_env_options() {
    static const std::vector<kvmem_env_option> values = {
        {"LLAMA_ARG_MODEL", "--model"}, {"LLAMA_ARG_THREADS", "--threads"},
        {"LLAMA_ARG_CTX_SIZE", "--ctx-size"}, {"LLAMA_ARG_N_PREDICT", "--n-predict"},
        {"LLAMA_ARG_BATCH", "--batch-size"}, {"LLAMA_ARG_UBATCH", "--ubatch-size"},
        {"LLAMA_ARG_FLASH_ATTN", "--flash-attn"}, {"LLAMA_ARG_TOP_K", "--top-k"},
        {"LLAMA_ARG_CACHE_TYPE_K", "--cache-type-k"}, {"LLAMA_ARG_CACHE_TYPE_V", "--cache-type-v"},
        {"LLAMA_ARG_N_PARALLEL", "--parallel"}, {"LLAMA_ARG_MMPROJ", "--mmproj"},
        {"LLAMA_ARG_MMPROJ_OFFLOAD", "--mmproj-offload", "--no-mmproj-offload"},
        {"MTMD_BACKEND_DEVICE", "--mmproj-device"},
        {"LLAMA_ARG_IMAGE_MIN_TOKENS", "--image-min-tokens"}, {"LLAMA_ARG_IMAGE_MAX_TOKENS", "--image-max-tokens"},
        {"LLAMA_ARG_MLOCK", "--mlock", ""}, {"LLAMA_ARG_MMAP", "--mmap", "--no-mmap"},
        {"LLAMA_ARG_LOAD_MODE", "--load-mode"}, {"LLAMA_ARG_DEVICE", "--device"},
        {"LLAMA_ARG_N_GPU_LAYERS", "--n-gpu-layers"}, {"LLAMA_ARG_SPLIT_MODE", "--split-mode"},
        {"LLAMA_ARG_TENSOR_SPLIT", "--tensor-split"}, {"LLAMA_ARG_MAIN_GPU", "--main-gpu"},
        {"LLAMA_ARG_ALIAS", "--alias"}, {"LLAMA_ARG_HOST", "--host"}, {"LLAMA_ARG_PORT", "--port"},
        {"LLAMA_ARG_STATIC_PATH", "--ui-dir"}, {"LLAMA_ARG_UI", "--ui", "--no-ui"},
        {"LLAMA_API_KEY", "--api-key"}, {"LLAMA_ARG_API_KEY_FILE", "--api-key-file"},
        {"LLAMA_ARG_TIMEOUT", "--timeout"}, {"LLAMA_ARG_THREADS_HTTP", "--threads-http"},
        {"LLAMA_ARG_JINJA", "--jinja", "--no-jinja"},
        {"LLAMA_ARG_REASONING_EFFORT", "--reasoning-effort"},
        {"LLAMA_ARG_THINK_BUDGET", "--reasoning-budget"},
        {"LLAMA_ARG_THINK_BUDGET_MESSAGE", "--reasoning-budget-message"},
        {"LLAMA_ARG_CHAT_TEMPLATE_KWARGS", "--chat-template-kwargs"},
        {"LLAMA_ARG_CHAT_TEMPLATE", "--chat-template"}, {"LLAMA_ARG_CHAT_TEMPLATE_FILE", "--chat-template-file"},
        {"LLAMA_ARG_SPEC_TYPE", "--spec-type"}, {"LLAMA_ARG_SPEC_DRAFT_N_MAX", "--spec-draft-n-max"},
        {"LLAMA_ARG_SPEC_DRAFT_P_MIN", "--spec-draft-p-min"},
    };
    return values;
}

inline bool kvmem_env_bool(std::string value, const char * name) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "1" || value == "true" || value == "on" || value == "yes") return true;
    if (value == "0" || value == "false" || value == "off" || value == "no") return false;
    throw std::invalid_argument(std::string(name) + " requires a boolean (1/0, true/false, on/off, yes/no)");
}

struct kvmem_start_arg { std::string value, source; };

// Resolver injection also lets tests exercise environments without mutating the process.
template<typename Lookup>
inline std::vector<kvmem_start_arg> kvmem_environment_args(const Lookup & lookup) {
    std::vector<kvmem_start_arg> args;
    for (const auto & option : kvmem_env_options()) {
        const char * value = lookup(option.env);
        if (!value) continue;
        const std::string source = std::string("env:") + option.env;
        if (option.negative) {
            const char * flag = kvmem_env_bool(value, option.env) ? option.flag : option.negative;
            if (*flag) args.push_back({flag, source});
        } else {
            args.push_back({option.flag, source});
            args.push_back({value, source});
        }
    }
    return args;
}

inline void kvmem_check_environment(char ** environment) {
    std::set<std::string> supported;
    for (const auto & option : kvmem_env_options()) supported.insert(option.env);
    for (char ** p = environment; p && *p; ++p) {
        const std::string entry(*p);
        const auto eq = entry.find('=');
        auto name = entry.substr(0, eq);
#ifdef _WIN32
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
#endif
        if (supported.count(name)) continue;
        if (name.compare(0, 10, "LLAMA_ARG_") != 0 && name.compare(0, 13, "LLAMA_API_KEY") != 0) continue;
        // Never fall back to unauthenticated/plaintext service after a misplaced auth/TLS setting.
        if (name.find("API_KEY") != std::string::npos || name.find("SSL_") != std::string::npos)
            throw std::invalid_argument("unsupported security environment variable " + name +
                                        "; use LLAMA_API_KEY / LLAMA_ARG_API_KEY_FILE; native TLS is not supported");
        fprintf(stderr, "KVMEM_WARNING unsupported environment variable %s is ignored; use --help for supported options\n", name.c_str());
    }
}

inline std::string kvmem_config_key(const char * raw) {
    std::string arg = kvmem_server_arg_alias(raw);
    static const std::map<std::string, std::string> aliases = {
        {"-m", "--model"}, {"-c", "--ctx-size"}, {"-n", "--n-predict"}, {"-b", "--batch-size"},
        {"-t", "--threads"}, {"-tb", "--threads-batch"}, {"-ub", "--ubatch-size"}, {"-fa", "--flash-attn"},
        {"-a", "--alias"}, {"-np", "--parallel"}, {"-ngl", "--n-gpu-layers"}, {"--gpu-layers", "--n-gpu-layers"},
        {"-dev", "--device"}, {"-mmdev", "--mmproj-device"}, {"-mg", "--main-gpu"},
        {"-sm", "--split-mode"}, {"-ts", "--tensor-split"},
        {"-to", "--timeout"}, {"-lm", "--load-mode"}, {"--mmap", "--load-mode"},
        {"--no-mmap", "--load-mode"}, {"--mlock", "--load-mode"}, {"-ctk", "--cache-type-k"},
        {"-ctv", "--cache-type-v"}, {"--no-ui", "--ui"}, {"--webui", "--ui"},
        {"--no-jinja", "--jinja"}, {"--no-mmproj-offload", "--mmproj-offload"},
        {"--chat-template-file", "--chat-template"}, {"--no-kvmem", "--kvmem"},
    };
    const auto it = aliases.find(arg);
    return it == aliases.end() ? arg : it->second;
}
