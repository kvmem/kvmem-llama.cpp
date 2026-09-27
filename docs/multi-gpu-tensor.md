# CUDA 双卡 tensor 并行

此模式复用 llama.cpp 的 `LLAMA_SPLIT_MODE_TENSOR` 和 Meta 设备。权重、算子和 attention KV 由上游按物理卡分片；KVMem 通过 Meta buffer 的逻辑行 `tensor_get/set` 归档与回填 KV，不在两卡各存一份完整历史 KV。当前为同步传输的正确性基线，沿用逐张量捕获读回；layer 的 CUDA D2H 合批不会处理 Meta 张量。P2P 与异步 shard 读取尚未实现。

启动时显式指定两张 CUDA 卡、`--split-mode tensor`、`--gpu-layers all`。KVMem 按 Meta 的 KV 分片规则计算每卡每 token 的 KV 字节数，再取两卡可容纳 block 数的较小值；权重比例不一定等于 KV 字节比例。遇到不受支持的 KV buffer 布局或容量不足会在加载阶段报错。ROW、混合后端和非 CUDA 设备会被入口拒绝；超过两张卡尚未验证。

MTP 仅支持 `--kvmem-mtp-state snapshots`。ReplaySSM 的 CUDA fold 目前需要直接访问物理 shard 指针，tensor 模式会在启动时拒绝该选项。视觉头设备选择由另一项改动处理，与本模式无关。

示例：

```powershell
llama-kvmem-server.exe -m <model.gguf> --device CUDA0,CUDA1 --split-mode tensor --tensor-split 4,1 -ngl all --kvmem --kvmem-budget 384 --kvmem-gen-reserve 128 --kv-dtype q8_0
```

实验工作区先前在 Qwen3.8-27B-UD-Q4_K_M 上完成任务一和 256K 压力测试；详细条件与限制见[设计与验证记录](tensor-parallel-design.md)及[256K 压力测试](task2-dual-256k-2026-09-25.md)。这些数据尚不是本分支重新构建后的验收结果。在当前 5060 Ti 16 GiB + 5050 8 GiB 上，短请求的 tensor 预填充比 layer 慢，不能将其设置为默认分卡模式。

当前分支在 CUDA 13.2.86、`120a-real` 上构建，9 项 Windows 仓库测试通过。因 5060 Ti 另有服务占用约 14 GiB 显存，运行回归使用 Qwen3.5-0.8B-Q8_0、CUDA0/CUDA1 tensor 3:1：任务一提示的原生 Meta 与 KVMem 各生成 32 token，token ID 完全相同；超过 160-token 工作池的长提示生成 8 token，检索日志记录 `n_stage_in=1`。KVMem 的捕获日志均为 `mode=legacy`，确认未把 Meta 张量送入 layer 的 CUDA 合批路径。当前分支未重新测 27B 吞吐或 256K 压力。
