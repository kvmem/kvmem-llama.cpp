# ninfer 超过生成预留的输出验证

2026 年 10 月 6 日，RTX 5050 Laptop 上的 ninfer KVMem 功能验证通过：R=128 时，两个 HTTP 请求分别生成了 512 和 1024 个输出 tokens，均以 `finish_reason=length` 结束。Main KV 页池固定为 B+R=384 tokens；生成途中通过重选和搬运继续追加，没有把输出上限截到 R。

## 设备和配置

| 项目 | 本次设置 |
|---|---|
| 设备 | NVIDIA GeForce RTX 5050 Laptop GPU，8151 MiB |
| GPU UUID | `GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29`；通过 CUDA_VISIBLE_DEVICES 绑定 |
| 模型 | `Ternary-Bonsai-2-27B-T2-MTP-r3-Q8.ninfer`；本次只加载主模型，权重 6.70 GiB |
| worker | 本地 `build-backends/ninfer/apps/ninfer-serve-qualified.exe`，CUDA 13.2 / SM120a 构建 |
| KV 和 GDN | 双 NVFP4 KV，FP16 GDN |
| 预算 | B=256、R=128、逻辑 context=4096、Host KV payload=512 MiB |
| 页与预填充 | 每页 64 tokens，Main 页池 6 页；prefill=128 |
| 执行 | 单活动请求，普通 decode；MTP/ngram/lookup/思考/CUDA Graph 均关闭 |
| 设备配置 | device-profile off，cuda-memory-policy default；未启用 WDDM override |
| 诊断 | 开启 `--kvmem-verify-transfers`，检查恢复的 KV 字节与搬运前后的 GDN 状态 |
| 运行方式 | 独立 worker，监听 `127.0.0.1:18426`；测试结束后停止 |

测试前 5050 显存占用为 0 MiB，运行中采样为 7046 MiB。两次请求完成后服务均健康；停止本次 worker 后显存回到 0 MiB。启动日志显示 Main KV capacity 为 384 tokens、页池为 6/6。本次是功能验证，不用于性能比较或证明全部分配的物理驻留。

## 请求结果

两个请求都要求从 1 数到 1000，每个整数占一行，temperature=0、seed=42。每次完整输入为 50 tokens；第二次复用了 49-token 输入 checkpoint，仍实际执行了 1024-token 生成。

| 请求 max_tokens | R | 实际 completion_tokens | 思考 tokens | 结束原因 | 生成期间重选次数 | 请求后健康状态 |
|---:|---:|---:|---:|---|---:|---|
| 512 | 128 | 512 | 0 | length，HTTP 200 | 2 | ok |
| 1024 | 128 | 1024 | 0 | length，HTTP 200 | 6 | ok |

512-token 请求在逻辑 frontier=384、512 时重选。1024-token 请求在 384、512、640、768、896、1024 时重选；另有一次输入 checkpoint 恢复，不计入这六次生成重选。

各次生成重选都把紧凑可见历史缩回 256 tokens，发布时使用 5 个物理页，包括下一步写入所需的页，没有超出 6 页池。每次搬运均记录 `payload_exact=1 state_exact=1`。例如最后一次重选为：

```text
KVMEM_NATIVE_VIEW logical=1024 compact=256 physical_pages=5 host_bytes=18874368 swap=9 payload_exact=1 state_exact=1
```

这说明逻辑生成进度可以越过 R，也可以越过整个 B+R 物理窗口，依靠 Host 归档和工作集重选继续执行。它不保证所有生成历史同时参与注意力，也不是长答案质量或 MTP、Graph、其他 KV 格式的验收。

## 本地证据

原始记录位于工作区 `backend-p0p1-20261002/results/reserve-output-5050-20261006/run-20261006-153501/`：

- `process.json` 和 `props-startup.json`：实际命令、GPU 绑定、预算及能力。
- `output-512-request.json` / `output-1024-request.json` 及对应 response：请求预算、完整输出、usage 和结束原因。
- 对应 `transfer.log`、`server.log`、`requests.jsonl`：重选 frontier、原生页数、KV/GDN 校验与请求完成记录。
- `summary.json`、`gpu-before.json`、`gpu-after.json`：两组结果与测试进程清理状态。

worker SHA-256 为 `d9217b414a7247d01aec4f53a2b10dcf247900d98b504cb545651d978e3c5f87`。本地复现入口为同级上层的 `run.py`，固定使用该 5050 UUID，并在结束时清理本次 worker；原始模型、二进制和日志不随源码文档发布。

参数含义与后端边界见 [两后端实现差异](multi-backend-differences.md)。
