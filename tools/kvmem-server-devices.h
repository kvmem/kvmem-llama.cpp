#pragma once

#include "kvmem-server-options.h"
#include "ggml-backend.h"

// Own the lists borrowed by llama_model_params until model/context teardown.
struct kvmem_server_devices {
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float> split;

    void apply(const kvmem_server_options & options, llama_model_params & params) {
        if (options.split_mode_set && (options.split_mode == LLAMA_SPLIT_MODE_ROW ||
                                       options.split_mode == LLAMA_SPLIT_MODE_TENSOR))
            throw std::invalid_argument("KVMem supports --split-mode layer for multiple GPUs; tensor is not implemented yet");
        if (!options.device_names.empty()) {
            if (options.device_names != "none") {
                size_t start = 0;
                while (start <= options.device_names.size()) {
                    const auto end = options.device_names.find(',', start);
                    const auto name = options.device_names.substr(start, end == std::string::npos ? end : end - start);
                    auto * dev = ggml_backend_dev_by_name(name.c_str());
                    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)
                        throw std::invalid_argument("invalid --device: " + name + "; see --list-devices");
                    if (std::find(devices.begin(), devices.end(), dev) != devices.end())
                        throw std::invalid_argument("--device contains a duplicate device");
                    devices.push_back(dev);
                    if (end == std::string::npos) break;
                    start = end + 1;
                }
            }
            devices.push_back(nullptr);
            params.devices = devices.data();
        }
        if (options.main_gpu_set) params.main_gpu = options.main_gpu;
        if (options.split_mode_set) params.split_mode = options.split_mode;
        if (!options.tensor_split.empty()) {
            if (options.tensor_split.size() >= llama_max_devices())
                throw std::invalid_argument("too many --tensor-split proportions");
            split = options.tensor_split;
            split.resize(llama_max_devices(), 0);
            params.tensor_split = split.data();
        }
        // Multi-GPU remains opt-in. The current adapter requires every attention
        // layer on a CUDA device and a single shared logical KV window.
        size_t count = devices.empty() ? 0 : devices.size() - 1;
        if (count > 1) {
            if (params.split_mode != LLAMA_SPLIT_MODE_LAYER)
                throw std::invalid_argument("multiple --device entries require --split-mode layer");
            if (params.n_gpu_layers != -2)
                throw std::invalid_argument("multiple GPUs require --gpu-layers all in this release");
            if (!options.tensor_split.empty() && options.tensor_split.size() != count)
                throw std::invalid_argument("--tensor-split must have one proportion per --device");
            for (size_t i = 0; i < count; ++i) {
                auto * reg = ggml_backend_dev_backend_reg(devices[i]);
                if (std::string(ggml_backend_reg_name(reg)) != "CUDA")
                    throw std::invalid_argument("multi-GPU layer currently requires CUDA devices from one backend");
            }
        } else if (options.tensor_split.size() > 1) {
            throw std::invalid_argument("multiple --tensor-split proportions require an explicit multi-GPU --device list");
        }
        if (devices.empty()) {
            size_t discrete = 0, integrated = 0, rpc = 0;
            std::vector<std::string> ids;
            ggml_backend_reg_t integrated_backend = nullptr;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                const auto type = ggml_backend_dev_type(dev);
                auto * reg = ggml_backend_dev_backend_reg(dev);
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    if (std::string(ggml_backend_reg_name(reg)) == "RPC") { ++rpc; continue; }
                    ggml_backend_dev_props props{};
                    ggml_backend_dev_get_props(dev, &props);
                    if (props.device_id) {
                        if (std::find(ids.begin(), ids.end(), props.device_id) != ids.end()) continue;
                        ids.emplace_back(props.device_id);
                    }
                    ++discrete;
                } else if (type == GGML_BACKEND_DEVICE_TYPE_IGPU &&
                           (!integrated_backend || integrated_backend == reg)) {
                    integrated_backend = reg;
                    ++integrated;
                }
            }
            count = rpc + (discrete ? discrete : integrated);
        }
        if (devices.empty() && count > 1 && params.n_gpu_layers != 0 && params.split_mode != LLAMA_SPLIT_MODE_NONE)
            throw std::invalid_argument("select multi-GPU explicitly with --device CUDA0,CUDA1 --split-mode layer --gpu-layers all");
    }
};
