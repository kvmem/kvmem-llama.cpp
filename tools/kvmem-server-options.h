#pragma once

#include "llama.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

inline const char * kvmem_server_arg_alias(const char * arg) {
    const std::string name(arg);
    if (name == "--usage") return "--help";
    if (name == "--predict") return "--n-predict";
    if (name == "-s") return "--seed";
    if (name == "-mm") return "--mmproj";
    if (name == "--no-webui") return "--no-ui";
    if (name == "--path") return "--ui-dir";
    return arg;
}

inline int kvmem_cli_int(const char * option, const char * value, int minimum = 0,
                         int maximum = std::numeric_limits<int>::max()) {
    char * end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value, &end, 10);
    if (!*value || end == value || *end || errno == ERANGE || parsed < minimum || parsed > maximum ||
            std::isspace(static_cast<unsigned char>(*value))) {
        throw std::invalid_argument(std::string(option) + " requires an integer in " +
                                    std::to_string(minimum) + ".." + std::to_string(maximum));
    }
    return static_cast<int>(parsed);
}

inline double kvmem_cli_real(const char * option, const char * value, double minimum, double maximum) {
    char * end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (!*value || end == value || *end || errno == ERANGE || !std::isfinite(parsed) ||
            parsed < minimum || parsed > maximum || std::isspace(static_cast<unsigned char>(*value))) {
        throw std::invalid_argument(std::string(option) + " requires a finite number in " +
                                    std::to_string(minimum) + ".." + std::to_string(maximum));
    }
    return parsed;
}

inline int kvmem_cli_gpu_layers(const char * option, const char * value) {
    const std::string text(value);
    if (text == "all" || text == "-2") return -2;
    if (text == "auto" || text == "-1") {
        throw std::invalid_argument(std::string(option) +
            " auto/-1 requires automatic GPU fitting, which is not implemented; use all or an integer >= 0");
    }
    return kvmem_cli_int(option, value);
}

struct kvmem_server_options {
    int sink_tokens = 0; // Zero keeps one block; positive values round down to whole blocks.
    // Live host KV stores, time-multiplexed on one GPU working set. One
    // reproduces the single-store server, where a different conversation
    // discards the previous one.
    int conversations = 1;
    uint64_t conversation_bytes = 0; // Accounted host store bytes; zero = count cap only.
    int verbosity = 3; // Same default and levels as llama-server.
    int trace = -1; // -1 inherits KVMEM_TRACE; CLI overrides only after parsing.
    int threads = -1;
    int threads_batch = -1;
    int ubatch = 0; // Omission preserves the existing batch-size default.
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool flash_attn_set = false;
    llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;
    std::string alias;
    std::vector<std::string> api_keys;
    int timeout = 1800; // Preserve KVMem's long-request default.
    int threads_http = -1;
    bool threads_http_set = false;
    bool list_devices = false;
    std::string device_names;
    int main_gpu = 0;
    bool main_gpu_set = false;
    llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;
    bool split_mode_set = false;
    std::vector<float> tensor_split;

    void parse_tensor_split(const std::string & value) {
        tensor_split.clear();
        size_t start = 0;
        while (start < value.size()) {
            const size_t end = value.find_first_of(",/", start);
            const std::string field = value.substr(start, end == std::string::npos ? end : end - start);
            tensor_split.push_back(static_cast<float>(kvmem_cli_real("--tensor-split", field.c_str(), 0,
                                     std::numeric_limits<float>::max())));
            if (end == std::string::npos) break;
            start = value.find_first_not_of(",/", end);
            if (start == std::string::npos) throw std::invalid_argument("--tensor-split has an empty final proportion");
        }
        if (tensor_split.empty()) throw std::invalid_argument("--tensor-split requires proportions");
        double total = 0;
        for (float part : tensor_split) total += part;
        if (total > std::numeric_limits<float>::max())
            throw std::invalid_argument("--tensor-split sum exceeds float range");
    }

    void add_key(const std::string & key) {
        if (key.empty()) throw std::invalid_argument("API key must not be empty");
        for (unsigned char c : key) {
            if (c < 33 || c > 126) throw std::invalid_argument("API key must contain printable ASCII without whitespace");
        }
        api_keys.push_back(key);
    }

    void add_keys(const std::string & value) {
        const size_t count = api_keys.size();
        std::string field;
        bool quoted = false;
        for (size_t i = 0; i < value.size(); ++i) {
            const char c = value[i];
            if (c == '"') {
                if (quoted && i + 1 < value.size() && value[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else if (quoted || field.empty()) {
                    quoted = !quoted;
                } else {
                    field += c;
                }
            } else if (c == ',' && !quoted) {
                if (!field.empty()) add_key(field);
                field.clear();
            } else {
                field += c;
            }
        }
        if (quoted) throw std::invalid_argument("--api-key contains an unclosed CSV quote");
        if (!field.empty()) add_key(field);
        if (count == api_keys.size()) throw std::invalid_argument("--api-key requires at least one nonempty key");
    }

    void add_key_file(const char * path) {
        std::ifstream file(path);
        if (!file) throw std::invalid_argument("cannot read --api-key-file");
        const size_t count = api_keys.size();
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty() && line.front() != '#') add_key(line);
        }
        if (file.bad()) throw std::invalid_argument("failed to read --api-key-file");
        if (count == api_keys.size()) throw std::invalid_argument("--api-key-file contains no usable keys");
    }

    template<typename Need>
    bool parse(const std::string & arg, const Need & need) {
        if (arg == "--kvmem-sink-tokens") {
            sink_tokens = kvmem_cli_int(arg.c_str(), need(arg.c_str()));
        } else if (arg == "--kvmem-conversations") {
            conversations = kvmem_cli_int(arg.c_str(), need(arg.c_str()), 1);
        } else if (arg == "--kvmem-conversations-gb") {
            const double gb = kvmem_cli_real(arg.c_str(), need(arg.c_str()), 0, 1048576);
            conversation_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (arg == "--kvmem-trace" || arg == "--no-kvmem-trace") {
            trace = arg == "--kvmem-trace" ? 1 : 0;
        } else if (arg == "-lv" || arg == "--verbosity" || arg == "--log-verbosity") {
            verbosity = kvmem_cli_int(arg.c_str(), need(arg.c_str()), 0, 5);
        } else if (arg == "-to" || arg == "--timeout") {
            timeout = kvmem_cli_int(arg.c_str(), need(arg.c_str()));
        } else if (arg == "--threads-http") {
            threads_http = kvmem_cli_int(arg.c_str(), need(arg.c_str()), std::numeric_limits<int>::min(),
                                       std::numeric_limits<int>::max() - 1024);
            threads_http_set = true;
        } else if (arg == "-dev" || arg == "--device") {
            device_names = need(arg.c_str());
            if (device_names.empty()) throw std::invalid_argument("--device requires device names or none");
        } else if (arg == "--list-devices") {
            list_devices = true;
        } else if (arg == "-mg" || arg == "--main-gpu") {
            main_gpu = kvmem_cli_int(arg.c_str(), need(arg.c_str()), -1);
            main_gpu_set = true;
        } else if (arg == "-sm" || arg == "--split-mode") {
            const std::string value = need(arg.c_str());
            if (value == "none") split_mode = LLAMA_SPLIT_MODE_NONE;
            else if (value == "layer") split_mode = LLAMA_SPLIT_MODE_LAYER;
            else if (value == "row") split_mode = LLAMA_SPLIT_MODE_ROW;
            else if (value == "tensor") split_mode = LLAMA_SPLIT_MODE_TENSOR;
            else throw std::invalid_argument("--split-mode requires none, layer, row or tensor");
            split_mode_set = true;
        } else if (arg == "-ts" || arg == "--tensor-split") {
            parse_tensor_split(need(arg.c_str()));
        } else if (arg == "-t" || arg == "--threads" || arg == "-tb" || arg == "--threads-batch") {
            int n = kvmem_cli_int(arg.c_str(), need(arg.c_str()), std::numeric_limits<int>::min());
            if (n <= 0) n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
            (arg == "-t" || arg == "--threads" ? threads : threads_batch) = n;
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            ubatch = kvmem_cli_int(arg.c_str(), need(arg.c_str()), 1);
        } else if (arg == "-fa" || arg == "--flash-attn") {
            const std::string value = need(arg.c_str());
            if (value == "on" || value == "1" || value == "true") flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            else if (value == "off" || value == "0" || value == "false") flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            else if (value == "auto") flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
            else throw std::invalid_argument("--flash-attn requires on, off or auto");
            flash_attn_set = true;
        } else if (arg == "-a" || arg == "--alias") {
            alias = need(arg.c_str());
            if (alias.empty()) throw std::invalid_argument("--alias must not be empty");
        } else if (arg == "--api-key") {
            add_keys(need(arg.c_str()));
        } else if (arg == "--api-key-file") {
            add_key_file(need(arg.c_str()));
        } else if (arg == "-np" || arg == "--parallel") {
            if (kvmem_cli_int(arg.c_str(), need(arg.c_str())) != 1)
                throw std::invalid_argument("KVMem supports --parallel 1 only; automatic/multiple slots are not implemented");
        } else if (arg == "--mmap" || arg == "--no-mmap") {
            load_mode = arg == "--mmap" ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        } else if (arg == "--mlock") {
            load_mode = LLAMA_LOAD_MODE_MLOCK;
        } else if (arg == "-lm" || arg == "--load-mode") {
            const std::string value = need(arg.c_str());
            if (value == "auto") load_mode = LLAMA_LOAD_MODE_AUTO;
            else if (value == "none") load_mode = LLAMA_LOAD_MODE_NONE;
            else if (value == "mmap") load_mode = LLAMA_LOAD_MODE_MMAP;
            else if (value == "mlock") load_mode = LLAMA_LOAD_MODE_MLOCK;
            else if (value == "mmap+mlock") load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
            else if (value == "dio") load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
            else throw std::invalid_argument("--load-mode requires auto, none, mmap, mlock, mmap+mlock or dio");
        } else {
            return false;
        }
        return true;
    }
};
