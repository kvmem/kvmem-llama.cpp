# 单独选择视觉头设备

`--mmproj-device DEVICE`（简写 `-mmdev`）只指定视觉投影器和编码器的设备，不改变语言模型的 `--device` / `--split-mode`。例如语言模型使用 `CUDA0,CUDA1` 时，可用 `--mmproj-device CUDA0` 或 `CUDA1` 将视觉头放在其中一张卡上。`--mmproj-device none` 等价于 CPU 视觉头；未指定时保留原来的自动选择。

Windows CUDA 构建可加 `-Vulkan`，使同时支持 Vulkan 的二进制能够选择集成显卡，例如 `--mmproj-device Vulkan0`。构建脚本会检查完整的 `VULKAN_SDK`，需要其中的 `glslc.exe` 和 SPIR-V 头文件。语言模型多卡入口仍只接受同一 CUDA 后端的显卡，Vulkan 仅用于视觉头。

启动时无效设备名或非 GPU 设备会给出明确错误；启动日志的 `vision.device` 和 `KVMEM_TRACE vision_load device=...` 可用于核对选择。视觉头实际设备名可先通过 `--list-devices` 查看。

## 实验工作区的设备对照（2026-09-25）

在 Qwen3.8-27B-UD-Q4_K_M、双 CUDA layer 8:1、Q8_0 视觉头、ReplaySSM MTP 下，以相同的 896×896 图片和问题做了新进程请求。每次为 547 个 prompt token、64 个生成 token，关闭 trace，计时不含启动。Intel Arc 140T 当时显示为 `Vulkan2`；编号随机器和驱动变化。

| 视觉头设备 | 图片预填充，秒 | HTTP 整次请求，秒 | 5050 加载后剩余显存 |
|---|---:|---:|---:|
| CPU | 11.54 / 11.64 | 13.71 / 13.90 | 3,903 MiB |
| Arc 140T (`Vulkan2`) | 7.93 / 6.61 / 5.98 | 10.11 / 8.78 / 8.15 | 3,903 MiB |
| RTX 5050 (`CUDA1`) | 1.92 / 1.79 | 4.03 / 3.86 | 3,301 MiB |

上述 iGPU 三次结果仍在下降，可能受 Vulkan shader/pipeline cache 预热影响，不能把最后一次当作稳定吞吐。移动视觉头到 iGPU 约为 5050 留出 602 MiB；CUDA1 更快。七次请求的 64-token 输出相同。这些是原实验工作区的数据，当前分支的构建和运行验证另行记录。
