#include "kvmem-server-options.h"
#include "kvmem-server-env.h"
#include <cstdio>
#include <functional>

static void check(bool ok) {
    if (!ok) throw std::runtime_error("server options test failed");
}
static void rejects(const std::function<void()> & fn) {
    try { fn(); } catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("expected invalid_argument");
}
int main() {
    check(kvmem_cli_gpu_layers("-ngl", "all") == -2);
    check(kvmem_cli_gpu_layers("-ngl", "0") == 0);
    for (const char * value : {"auto", "-1", "1x", "999999999999", "-3", " 1"})
        rejects([&] { kvmem_cli_gpu_layers("-ngl", value); });
    for (const char * value : {"NaN", "inf", "1x", "-0.1", "1.1"})
        rejects([&] { kvmem_cli_real("ratio", value, 0, 1); });
    kvmem_server_options o;
    auto parse = [&](const char * flag, const char * value) {
        check(o.parse(flag, [&](const char *) { return value; }));
    };
    check(o.sink_tokens == 0);
    for (const char * value : {"0", "1", "128", "129", "1024", "2147483647"}) {
        parse("--kvmem-sink-tokens", value);
        check(o.sink_tokens == std::stoi(value));
    }
    for (const char * value : {"-1", "1.5", "128x", "", " 128", "2147483648", "9999999999999999999999"})
        rejects([&] { parse("--kvmem-sink-tokens", value); });
    rejects([&] { o.parse("--kvmem-sink-tokens", [](const char *) -> const char * {
        throw std::invalid_argument("missing value");
    }); });
    for (const char * flag : {"-lv", "--verbosity", "--log-verbosity"}) {
        parse(flag, "2"); check(o.verbosity == 2);
        for (const char * bad : {"-1", "6", "2x", "1.5", "999999999999"})
            rejects([&] { parse(flag, bad); });
    }
    parse("--kvmem-trace", ""); check(o.trace == 1);
    parse("--no-kvmem-trace", ""); check(o.trace == 0);
    parse("-t", "3"); parse("-tb", "5"); parse("-ub", "64");
    check(o.threads == 3 && o.threads_batch == 5 && o.ubatch == 64);
    parse("--flash-attn", "off"); check(o.flash_attn == LLAMA_FLASH_ATTN_TYPE_DISABLED);
    parse("-fa", "on"); check(o.flash_attn == LLAMA_FLASH_ATTN_TYPE_ENABLED);
    parse("-fa", "auto"); check(o.flash_attn == LLAMA_FLASH_ATTN_TYPE_AUTO);
    parse("-np", "1");
    rejects([&] { parse("-np", "2"); });
    rejects([&] { parse("-ub", "0"); });
    rejects([&] { parse("-fa", "invalid"); });
    parse("--load-mode", "mmap+mlock"); check(o.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);
    parse("--no-mmap", ""); check(o.load_mode == LLAMA_LOAD_MODE_NONE);
    parse("--mlock", ""); check(o.load_mode == LLAMA_LOAD_MODE_MLOCK);
    parse("--api-key", "one,two,\"comma,key\",\"escaped\"\"quote\"");
    check(o.api_keys == std::vector<std::string>({"one", "two", "comma,key", "escaped\"quote"}));
    rejects([&] { parse("--api-key", ""); });
    rejects([&] { parse("--api-key", "\"unclosed"); });
    rejects([&] { parse("--api-key", "secret\nvalue"); });
    check(std::string(kvmem_server_arg_alias("--predict")) == "--n-predict");
    check(std::string(kvmem_server_arg_alias("-mm")) == "--mmproj");
    check(std::string(kvmem_server_arg_alias("--path")) == "--ui-dir");
    parse("-to", "60"); parse("--threads-http", "3");
    check(o.timeout == 60 && o.threads_http == 3 && o.threads_http_set);
    rejects([&] { parse("--timeout", "-1"); });
    rejects([&] { parse("--threads-http", "3x"); });
    parse("--device", "CUDA1"); parse("-mg", "0"); parse("-sm", "none");
    check(o.device_names == "CUDA1" && o.main_gpu == 0 && o.split_mode == LLAMA_SPLIT_MODE_NONE);
    parse("-ts", "3/1"); check(o.tensor_split == std::vector<float>({3, 1}));
    parse("-ts", "0,0"); check(o.tensor_split == std::vector<float>({0, 0}));
    for (const char * bad : {"", "1,", "-1,1", "nan,1", "1x,2", "3e38,3e38"})
        rejects([&] { parse("--tensor-split", bad); });
    rejects([&] { parse("--split-mode", "bad"); });
    rejects([&] { parse("--main-gpu", "-2"); });
    std::map<std::string, std::string> environment = {
        {"LLAMA_ARG_MODEL", "model with spaces.gguf"}, {"LLAMA_ARG_UI", "false"},
        {"LLAMA_API_KEY", "env-key"}, {"LLAMA_ARG_MMPROJ_OFFLOAD", "ON"},
    };
    auto lookup = [&](const char * name) -> const char * {
        auto it = environment.find(name);
        return it == environment.end() ? nullptr : it->second.c_str();
    };
    auto args = kvmem_environment_args(lookup);
    check(args.size() == 6 && args[0].value == "--model" && args[1].value == "model with spaces.gguf");
    check(args[0].source == "env:LLAMA_ARG_MODEL");
    check(std::any_of(args.begin(), args.end(), [](const kvmem_start_arg & a) { return a.value == "--no-ui"; }));
    environment["LLAMA_ARG_UI"] = "sometimes";
    rejects([&] { kvmem_environment_args(lookup); });
    check(kvmem_config_key("-ngl") == "--n-gpu-layers");
    check(kvmem_config_key("--path") == "--ui-dir");
    check(kvmem_config_key("--no-mmap") == "--load-mode");
    // Multi-conversation host stores: absent flags must reproduce the
    // single-store server, so the struct defaults are part of the contract.
    check(kvmem_server_options{}.conversations == 1);
    check(kvmem_server_options{}.conversation_bytes == 0);
    check(o.conversations == 1 && o.conversation_bytes == 0);
    for (const char * value : {"1", "2", "8", "64", "2147483647"}) {
        parse("--kvmem-conversations", value);
        check(o.conversations == std::stoi(value));
    }
    for (const char * value : {"0", "-1", "1.5", "2x", "", " 2", "2147483648", "9999999999999999999999"})
        rejects([&] { parse("--kvmem-conversations", value); });
    rejects([&] { o.parse("--kvmem-conversations", [](const char *) -> const char * {
        throw std::invalid_argument("missing value");
    }); });
    parse("--kvmem-conversations-gb", "0");
    check(o.conversation_bytes == 0);
    parse("--kvmem-conversations-gb", "0.5");
    check(o.conversation_bytes == 536870912ull);
    parse("--kvmem-conversations-gb", "24");
    check(o.conversation_bytes == 24ull << 30);
    for (const char * value : {"-1", "nan", "inf", "1x", "", " 1", "1048577"})
        rejects([&] { parse("--kvmem-conversations-gb", value); });
    rejects([&] { o.parse("--kvmem-conversations-gb", [](const char *) -> const char * {
        throw std::invalid_argument("missing value");
    }); });
    check(kvmem_config_key("--kvmem-conversations") == "--kvmem-conversations");
    check(kvmem_config_key("--kvmem-conversations-gb") == "--kvmem-conversations-gb");
    // Request-independent stderr on the default path must stay what
    // v0.16.0-rc3 printed. The -np rejection is the one message this feature
    // was tempted to extend; the pointer to --kvmem-conversations belongs in
    // print_usage, where changed output is expected.
    for (const char * flag : {"-np", "--parallel"}) {
        bool threw = false;
        try {
            o.parse(flag, [](const char *) { return "2"; });
        } catch (const std::invalid_argument & e) {
            threw = true;
            check(std::string(e.what()) ==
                  "KVMem supports --parallel 1 only; automatic/multiple slots are not implemented");
        }
        check(threw);
        parse(flag, "1");
    }
    std::puts("server options: passed");
}
