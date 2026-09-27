# 自己编译 KVMem + llama.cpp（Windows / ROCm 10 / gfx1030）

> 以下保留 dockylf 在 PR #10 中的原始环境记录和排障经验，其中的版本、目录、
> 默认架构与测量数字属于该次记录。当前整合版的编译入口、自动架构检测、
> `build-hip-win` 目录及 IQ3 启动方式请使用 [ROCm 双平台指南](rocm.md)。

本文记录在 Windows + AMD ROCm/HIP（RDNA2 / gfx1030）上从源码构建 KVMem + llama.cpp
的完整步骤，含两个会**静默**毁掉性能的陷阱。下文把仓库目录记作 `<repo>`。

---

## 0. 前置条件（本机已全部满足）

| 依赖 | 示例位置 | 说明 |
|---|---|---|
| ROCm 10.0.0 | `D:\ROCm\10.0.0` | 提供 AMD clang 23 与运行期 DLL |
| Visual Studio 2022 BuildTools | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` | **工具集必须是 MSVC 14.44**，见 §6 |
| CMake + Ninja | `w64devkit` 的 `bin` 目录（或任意含 cmake/ninja 的目录） | 已在持久 PATH 上 |
| Windows SDK | `C:\Program Files (x86)\Windows Kits\10` | 提供 `rc.exe`（cmake 的 RC 语言需要） |

环境变量 `ROCM_PATH` / `HIP_PATH` / `HIP_DEVICE_LIB_PATH` / `HIP_PLATFORM` 你已设好，无需再动。

---

## 1. 首次编译

打开开始菜单里的 **“x64 Native Tools Command Prompt for VS 2022”**（**不要**用 VS2026 的），然后：

```bat
cd /d <repo>
scripts\build-hip.bat
```

脚本会依次做：加载 MSVC 环境 → 把 ROCm clang 放到 PATH 最前 → 检查 cmake/ninja →
`cmake` 配置（HIP 后端，强制关掉 CUDA 后端）→ 校验优化标志 → `cmake --build`。

**耗时**：本机实测 `-j12` 约 **26 分钟**（578 个编译步骤）。首次编译慢是因为
`GGML_CUDA_FA_ALL_QUANTS` 被强制打开（`--kv-dtype q5_0` 等 KV 类型的必要条件），
会多编 45 个 fattn-vec 实例。

产物在 `build-hip\bin\`，主要是 `llama-kvmem-server.exe` + 一组 DLL。

### 可选的环境变量覆盖

```bat
set JOBS=16                     & rem 并行度，默认 12
set GPU_TARGETS=gfx1030         & rem 默认已是 gfx1030
set ROCM=D:\ROCm\10.0.0         & rem 默认取 ROCM_PATH
set BUILD_DIR=E:\build-kvmem    & rem 换一个构建目录
set CONFIGURE_ONLY=1            & rem 只配置不编译（改 CMake 选项后快速验证）
set FRESH=1                     & rem 先删掉构建目录再配置
```

---

## 2. 日常增量编译（改了源码之后）

改完 `src/` 或 `kvmem/` 下的文件，直接重跑同一个脚本即可——ninja 只重编受影响的文件：

```bat
scripts\build-hip.bat
```

改完 `CMakeLists.txt` 也一样（cmake 会先自动重新配置）。

---

## 3. 编译完先验证（强烈建议）

**这一步别跳过。** 曾经出现过「编译完全成功但推理慢 400 倍」的情况（§6），
最快的判别方式就是跑一次基线：

```bat
set HIP_VISIBLE_DEVICES=0
build-hip\bin\llama-bench.exe -m "<模型目录>\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" -p 64 -n 16 -ngl 99 -r 1
```

期望值（RX 6900 XT / IQ3_S）：

```
pp64  ≈ 300 t/s
tg16  ≈  30 t/s
```

若看到 **0.7 / 0.2** 这种数字，就是构建没开优化 → 按 §6 处理。
另一个快速旁证：`build-hip\bin\ggml-hip.dll` 正常约 **70 MB**，
若变成 **250 MB 以上** 就是未优化。

---

## 4. 启动服务

把 `build-hip\bin` 置顶到 PATH（避免误用 PATH 上既有的 `llama-server`），然后：

```
build-hip\bin\llama-kvmem-server.exe ^
  -m "<模型目录>\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" ^
  --host 127.0.0.1 --port 18200 ^
  -c 262144 -n 12288 --kvmem-budget 32768 --kvmem-gen-reserve 12288 ^
  --kv-dtype q4_0 --spec-type draft-mtp --spec-draft-n-max 1 ^
  --enable-thinking --reasoning-budget 4096
```

可覆盖的变量：`MODEL` / `MMPROJ` / `PORT` / `CTX` / `BUDGET` / `RESERVE` / `KV_DTYPE` /
`EXTRA_ARGS`，以及 `TEXT_ONLY=1`（不加载视觉编码器）。

验证服务：

```bat
python scripts\kvmem_http_check.py --port 18200 --wait 900
python scripts\kvmem_needle_test.py --port 18200 --target-tokens 80000 --budget 32768 --no-think
```

`kvmem_needle_test.py` 是 KVMem 的针对性测试：造一段超出 GPU 预算的长提示、
在早期藏一个唯一事实，检查是否还能取回。

### ⚠️ 必须显式给 `--kvmem-budget`（否则一定起不来）

`--kvmem-budget 0`（**默认值**）的含义是「GPU 常驻集 = 整个 n_ctx」，也就是**完全不虚拟化**。
所以只抄普通 llama-server 的 `-c` 而漏掉 `--kvmem-budget`，KVMem 会按满上下文去申请 KV 池：

```
-c 262144 --kv-dtype q8_0  （漏掉 --kvmem-budget）
  → KVMEM_KV_BYTES bytes=8578662400   （7.99 GiB）
  → 权重 11159.69 MiB + KV 7.99 GiB = 19.2 GiB > 16 GiB 显存
  → ggml_cuda_host_malloc: failed to allocate 260.91 MiB of pinned memory: out of memory
  → failed to initialize the context: failed to allocate compute pp buffers
```

注意症状具有误导性：报的是**主机 pinned 内存**分配失败，不是显存。原因是 Windows ROCm 下
锁页主机缓冲会被算作「共享 GPU 内存」（你的 `hip-build-quickref.txt` 里
`GGML_CUDA_NO_PINNED=1` 那条说的就是它）。真正的根因是显存超配。

`--kvmem-gpu-ratio 0.50` 这个兜底**保护不了你**——它是按**总显存**的 50% 算的，不会扣除
已经占掉的 11.2 GiB 权重。所以一定要自己给预算。

**本模型每 token 的 KV 开销**（16 个注意力层 × 4 个 KV 头 × (256+256) 维）：

| `--kv-dtype` | 字节/token | 1 GiB 可装 | 建议的值（约 ≤2 GiB） |
|---|---|---|---|
| `f16` | 65,536 | ~16 K | `--kvmem-budget 24576` |
| `q8_0` | 34,816 | ~31 K | `--kvmem-budget 49152`（≈1.7 GiB，**已验证可用**） |
| `q5_0` | 22,528 | ~48 K | `--kvmem-budget 81920` |
| `q4_0` | 18,432 | ~58 K | `--kvmem-budget 131072`（≈2.4 GiB） |

本机可用的显存预算参考（16.0 GiB 可用）：

```
权重            11.16 GiB
GDN 循环态       0.31 GiB
BF16 视觉编码器  0.93 GiB   ← --no-mmproj-offload 可省掉
计算图/锁页缓冲  约 0.6 GiB
─────────────────────────
留给 KV 约 2.5–3.0 GiB
```

### ⚠️ 必须加 `--load-mode none`：否则白占 ~10.5 GB 物理内存

同模型、`-ngl 99`、`-c 32768` 的实测对照（脚本 `scripts/proc_mem.py`）：

| 配置 | 提交(private) | **工作集（物理驻留）** |
|---|---|---|
| 主线 llama-server `--load-mode none` | 11.95 GiB | **0.80 GiB** |
| KVMem server 默认（AUTO，mmap 开） | 12.52 GiB | **11.31 GiB** ← 多 10.5 GiB |
| **KVMem server `--load-mode none`** | 12.88 GiB | **0.79 GiB** ← 与主线一致 |

原因：KVMem server 原先**无法关闭 mmap**，默认 `AUTO` 会让 12 GB 模型文件映射进进程
工作集且驻留不去；主线用 `--load-mode none` 走 fread，读完即走。
所以务必加上：

```bat
-lm none    & rem 或 --load-mode none；等价于主线老参数 --no-mmap
```

**注意别把两个数字搞混**：
- **提交(private) ~12.5 GiB 是显存被 WDDM 记在进程提交量上**，主线也一样，关不掉，
  与 `--load-mode` 无关；
- **工作集 11.3 GiB 才是 mmap 造成的**，靠 `--load-mode none` 能压掉。

### ⚠️ 262144 上下文的真正瓶颈：提交额度（commit），不是显存

`--load-mode` 已支持：`auto`（默认）| `none` | `mmap` | `mlock` | `mmap+mlock` | `dio`，
取值与 llama-server 一致。**`none` 就是老 `--no-mmap` 的继承者**（源码佐证：
`llama-model-loader.cpp:559` 的 `use_mmap = ... || LLAMA_LOAD_MODE_AUTO`，
而 arg.cpp 里 `--no-mmap` 的帮助文本写明 "DEPRECATED in favor of `--load-mode`"）。

**但它救不了 OOM。** 实测（本机 30.9 GB 内存）：

| 状态 | 可用物理 | 可提交 | 提交上限 |
|---|---|---|---|
| 服务停止 | 19.5 GB | 17.6 GB | 35.9 GB |
| 服务启动（空上下文） | — | **3.2 GB** | 35.9 GB |
| 服务运行到 ~70K 上下文 | 3.9 GB | **0.7 GB** | 39.0 GB（页面文件自动涨到 8 GB） |

服务进程**在空上下文时就已提交约 14.4 GB**——其中绝大部分是 **WDDM 把显存分配记在进程
提交量上**（权重 11.16 + KV 池 + GDN + MTP ≈ 13 GB），模型在主机侧只占约 388 MiB
（mmap 生效，且 `-ngl 99` 下权重在 GPU）。`--load-mode none` 改的只是读取路径，
对这个数字几乎没有影响（实测无改善）。

于是可用额度只剩约 3 GB，而 KVMem 的 raw-K 主机存储约 **32 KiB/token**：

```
 70,000 token  →  2.3 GB   → 额度耗尽（与你观察到的"60–70K 就停"完全吻合）
262,144 token  →  8.6 GB   → 远高于现有额度，根本装不下
```

**结论：想用 262144，唯一有效的是提高提交上限——也就是放大页面文件。**
本机页面文件仅约 5 GB（提交上限 35.9 = 物理 30.9 + 页面 5）。

控制面板 → 系统 → 高级系统设置 → 性能[设置] → 高级 → 虚拟内存[更改]：
取消「自动管理」，C 盘自定义 **初始 32768 MB / 最大 65536 MB**，重启。

或管理员 PowerShell：

```powershell
$cs = Get-CimInstance Win32_ComputerSystem
$cs.AutomaticManagedPagefile = $false; Set-CimInstance -InputObject $cs
New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name='C:\pagefile.sys'; InitialSize=32768; MaximumSize=65536}
```

改后提交上限 ≈ 63 GB，扣掉服务 14.4 GB 与其它程序，262144 的 8.6 GB 完全装得下。
放 SSD 无妨：raw-K 只在逐出/取回时读，不会持续换页。

自助核对工具（ctypes `GlobalMemoryStatusEx`）：`scripts/mem_status.py`。

### ⚠️ 显存不足的真实症状：桌面先崩，server 无声消失

这块卡同时跑桌面（`dwm.exe` 也住显存里）、浏览器、DeepSeek Harness。余量一旦见底，
**最先崩的往往是 `dwm.exe`** → AMD 驱动复位（TDR）→ **该卡上所有 HIP 上下文被销毁** →
server 下一次 HIP 调用直接终止，**日志戛然而止、没有任何报错**。

确诊方法（事件日志；输出是 GBK，注意解码）：

```bat
wevtutil qe Application /q:*[System[(EventID=1000 or 1001 or 1002 or 2004)]] /c:40 /rd:true /f:text
```

看到 `dwm.exe` + `0xc00001ad` 就是显存耗尽，不是代码 bug。处理：按上表腾显存，并关掉
抢显存的程序（浏览器硬件加速、ComfyUI 等）。

实测同一台机器（iq3_s + q4_0 KV + budget 49152 + mmproj 在 GPU）：

| 场景 | 结果 |
|---|---|
| 单次 75K token 提示（独占 GPU） | 通过 |
| 多轮 20K → 80K（独占 GPU） | 通过 |

**"跑到 60–70K 就停"在独占 GPU 时无法复现——问题在显存余量，不在上下文长度。**

显存紧张时两个额外手段：

- `--no-mmproj-offload`：把 931 MB 的 BF16 视觉编码器放 CPU（官方 IQ4 配方就是这么做的）。
  官方 IQ3 配方敢把它放 GPU，是因为它用的是 `mmproj-Q8_0`，只有你这份 BF16 的一半大。
- `set GGML_CUDA_NO_PINNED=1`：再省约 0.6 GiB 共享显存，代价是主机↔设备拷贝略慢。

### 鉴权：`--api-key`（本仓库已实现）

上游的 `llama-kvmem-server` **没有**鉴权，而 DeepSeek Harness 这类客户端必须有 key 才肯接入，
所以本仓库给它补上了，语义与 llama.cpp 自带的 server 保持一致：

```bat
--api-key dockylf                    & rem 可重复，给多个 key
--api-key-file keys.txt              & rem 每行一个 key（避免 key 出现在命令行/进程列表里）
```

- 读 `Authorization: Bearer <key>`，回退到 Anthropic 风格的 `X-Api-Key`
- **`/health` 与 `/v1/health` 保持公开**，这样启动器/健康检查在拿到 key 之前就能探测就绪
- `OPTIONS` 豁免（CORS 预检不受影响）
- 不带 key 或 key 不对 → `401` + OpenAI 风格的 `authentication_error` JSON
- 不传该参数 = 关闭鉴权（与上游行为一致）

启动日志会打印实际状态，便于确认 key 有没有被读到：

```
listening on http://127.0.0.1:18200 ... auth=api-key
api key authentication enabled (1 key), /health stays public
```

仍然建议绑 `--host 127.0.0.1`：绑 `0.0.0.0` 会把服务暴露到整个局域网。

### 输出长度上限：`--kvmem-gen-reserve` 才是那个数

`-n` / `--kvmem-gen-reserve` 决定单次生成的上限，服务端把它算成
`generation_limit = min(n_ctx, gen_reserve)`。本命令里是 12288，所以：

- 客户端要 `max_completion_tokens` 大于 12288 **不会报错**，会被**截断到 12288**
  并在日志里留一行 `requested ... exceeds the generation limit 12288; clamping`
- `null` / `0` / `-1` / 不传 → 用服务端默认值
- 只有**类型不对**（比如字符串）才返回 400

想让它能生成更长的回答，就调大 `--kvmem-gen-reserve`（同时保证 `--kvmem-budget`
比它大出足够余量）。例如允许 32K 输出：

```bat
--kvmem-gen-reserve 32768 --kvmem-budget 81920
```

回归测试：`python scripts\kvmem_output_limit_test.py --port 18200 --api-key <key>`
覆盖各种客户端会发的形态（超大值 / `null` / `0` / `-1` / 浮点 / 字符串 / `max_tokens`）。

### 日志格式（已与 llama.cpp 主线对齐）

上游的 `llama-kvmem-server` 在 `common_init()` **之后**又调了一次 `llama_log_set()`，
用一个丢弃日志级别的 `fputs` 覆盖掉 llama.cpp 自己装好的处理器。后果有两个：

1. 所有来自 llama.cpp 的行丢掉了 `<时间> <级别> <标签> <函数>:` 前缀；
2. **级别过滤完全失效**——连 DEBUG 行都全量打印（实测同一场景 2776 行，而主线只有 30 行）。

本仓库已删掉那个覆盖。现在 `common_init()` 装的 `common_log_default_callback` 生效，
llama.cpp 的行与 llama-server 格式一致：

```
0.00.011.434 I ggml_cuda_init: found 1 ROCm devices (Total VRAM: 16368 MiB):
0.00.061.816 I llama_model_loader: loaded meta data with 53 key-value pairs ...
0.05.676.386 I srv    llama-kvmem-server listening on http://127.0.0.1:18200 ...
```

还补了主线的 `-lv` 旋钮（别名 `--log-verbosity`）：

| 值 | 含义 |
|---|---|
| 0 / 1 / 2 | 通用输出 / error / warning |
| **3** | **info（默认，与 llama-server 相同）** |
| 4 | trace —— 放行库层 INFO，能看到设备识别、模型加载、层卸载等 |
| 5 | debug |

实测：默认 `-lv 3` 是 7 行；`-lv 4` 是 238 行。想找回原来那种详尽输出就用 `-lv 4`。

**`KVMEM_*` 诊断行默认不显示**。它们是机器可读记录，`scripts/` 下有十几个脚本用正则解析：

```python
re.search(r"KVMEM_KV_BYTES bytes=(\d+) cells=(\d+).*budget=(\d+)", ln)
re.compile(r"KVMEM_GEN_WALL n=(\d+) ms=([\d.]+) toks=([\d.]+)")
```

仓库里本来就有 `KVMEM_TRACE` 这个开关（`long_ctx_vram.py`、`mtp_canary.py` 等十几个脚本
都在设它），但大量打印点漏了门控，导致交互使用时被刷屏（实测 163/535 行）。
现已把 **tools 与 src/adapter 的 117 处打印点全部统一门控**：

| 开关 | 作用 |
|---|---|
| （默认） | 不打印任何 `KVMEM_*`，日志只剩 llama.cpp 自己的输出 |
| `--kvmem-trace` | 打印诊断行；同时把日志级别提到 trace，让库层那些走 `LLAMA_LOG_INFO` 的行也出来 |
| `--no-kvmem-trace` | 即使环境变量设了也强制关闭 |
| `KVMEM_TRACE=1` | 环境变量，等价于 `--kvmem-trace`（脚本用的就是这个） |

实测同一个启动场景：

| 模式 | 总行数 | KVMEM_ 行 |
|---|---|---|
| 默认 | **3** | **0** |
| `--kvmem-trace` | 238 | 6 |

**为什么开关用环境变量做权威**：门控状态在两个模块（server 可执行文件、libllama）里
各有一份内联静态副本，CLI 参数传不进库层；所以 `--kvmem-trace` 会**同时把
`KVMEM_TRACE` 写进进程环境**，库层（`src/adapter`）本来就从环境读它。

#### 还治好了 mtmd/视觉编码器的日志（这是行数暴增的主因）

`mtmd`/`clip` 有自己的一套日志（`clip-impl.h` 的 `LOG_DBG/LOG_INF/...`），默认回调同样是
裸 `fputs(text, stderr)`——**没有前缀，也没有级别过滤**，于是给 334 个投影器张量各打一行。
实测一次会话 535 行里有 **341 行**是它。

主线是在 `server-context.cpp` 里把它接进公共日志的：

```cpp
mtmd_helper_log_set(common_log_default_log_callback, nullptr);
```

本仓库照做了。现在启动日志 **535 → 12 行**，`clip_model_loader: tensor[N]...` 完全消失；
`-lv 4` 仍可把它们调出来。

#### 补齐主线的运行时统计

原来只有 `KVMEM_*` 机器记录，人看不到速率。现在**同时**输出主线那几条（措辞、字段宽度、
触发条件都对齐 llama-server）：

预填进行中，每 3 秒一条，带百分比：

```
1.02.140.880 I srv    prompt processing, n_tokens =   3074, progress = 0.10, t =   7.20 s / 426.71 tokens per second
2.13.783.323 I srv    prompt processing, n_tokens =  29186, progress = 0.97, t =  78.85 s / 370.16 tokens per second
```

解码进行中，每 3 秒一条（累计平均 + 3 秒滑窗），与主线一样在第 100 个 token 之后才开始：

```
0.13.146.824 I srv    n_gen =    279, tg =  46.23 t/s, tg_3s =  48.25 t/s
0.16.169.077 I srv    n_gen =    409, tg =  45.16 t/s, tg_3s =  43.01 t/s
```

每轮结束时的汇总：

```
prompt eval time =     650.16 ms /    46 tokens (   14.13 ms per token,    70.75 tokens per second)
       eval time =   20384.47 ms /   921 tokens (   22.13 ms per token,    45.18 tokens per second)
      total time =   21034.63 ms /   967 tokens
```

进度对象挂在 `ServerState` 上（该 server 只有一个 slot），预填时由
`kvmem-multimodal-server.h` 的分块循环驱动，解码时由 token 回调驱动。

---

## 5. 以后想更新上游代码

KVMem 的补丁是**针对 pin `b81c99b` 的累积 diff**，所以：

- **只想重建**：什么都不用做，直接 §1。
- **想换 llama.cpp 版本**：不能直接 `git pull`。需要先 rebase 补丁，用上游的
  `scripts/rebase-llama.sh`，然后重跑 `scripts/apply-patches.sh`。
- **想确认补丁还在**：

  ```bat
  cd llama.cpp
  git apply --reverse --check ..\patches\llama-kvmem-current.patch && echo PATCH-APPLIED
  ```
  （能反向 apply 成功 = 补丁已应用。`scripts\apply-patches.sh` 是幂等的，重复跑安全。）

### ⚠️ git 可能报 unsafe repository

如果仓库所在分区没有 git 能校验的所有权元数据（仓库在非系统盘、或由其它用户/工具
创建时常见），直接敲 git 会报 `fatal: unsafe repository ... owned by someone else`。
把该目录加入安全列表即可：

```bat
git config --global --add safe.directory <repo>
```

---

## 6. 两个已知陷阱

### (1) 优化标志为空 → 能编译成功但慢约 400 倍

**成因**：构建目录**第一次**配置时 C++ 编译器还没被识别（典型原因：clang++ 不在 PATH 上），
CMake 不会加载 `Platform/Windows-Clang.cmake`，于是把**空的** `CMAKE_CXX_FLAGS_RELEASE`
缓存下来；之后所有配置都沿用，结果连 HIP 设备码都没有 `-O3`。

**判别**：

```bat
findstr CMAKE_CXX_FLAGS_RELEASE build-hip\CMakeCache.txt
```
必须看到 `=-O3 -DNDEBUG`。空的或没有 `-O` 就是中招了。

**处理**：删掉构建目录重新配置（`set FRESH=1` 后重跑脚本）。

```bat
set FRESH=1
scripts\build-hip.bat
```

> 删构建目录时**不要**用带「忽略错误」的删除——静默的残缺删除正是让污染缓存活下来的原因。

现在有两道防线：根 `CMakeLists.txt` 会直接让配置 `FATAL_ERROR`，`build-hip.bat` 在配置后
也会再自查一次。

### (2) MSVC 工具集必须是 14.44，不能用 14.51

AMD clang 会自动探测**最新**的 Visual Studio。你机器上装了 VS2026（MSVC 14.51），
而 14.51 把双参 `<cmath>` builtin（`isgreater` / `isless` / `isunordered` 等）声明成
`constexpr`，clang 会把它当成 `__host__ __device__`，与 ROCm 自己的
`__device__` 声明冲突，HIP 设备码编译直接失败：

```
error: __device__ function 'isgreater' cannot overload __host__ __device__ function 'isgreater'
```

**所以务必用开始菜单的 “x64 Native Tools Command Prompt for VS 2022”**，
它天然就是 14.44。`build-hip.bat` 也会主动去找 VS2022 的 `vcvars64.bat`。
想确认 clang 实际用的是哪套 STL：

```bat
echo | clang++ -v -E -x c++ - 2>&1 | findstr /C:"search starts" /C:"14."
```

正常应只列出 `...\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include`，**不应**出现 `18\Community` 或 `14.51`。

---

## 7. 备选：无法使用 cmd.exe 时（沙箱 / CI）

如果环境不允许调用 `cmd.exe`（`vcvars64.bat` 就没法加载），用这套等价流程：

```bash
cd <repo>
python scripts/msvc_env.py --msvc-version 14.44.35207 > msvc-env.sh
source msvc-env.sh          # 导入 INCLUDE / LIB / LIBPATH
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DGGML_HIP_UMA=OFF -DGGML_VULKAN=OFF \
    -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON
cmake --build build-hip --config Release -j 12
```

`scripts/msvc_env.py` 负责从文件系统里定位 MSVC 14.44 与 Windows SDK 并生成
`INCLUDE`/`LIB`/`LIBPATH`。**它刻意不导出 PATH**——把 Windows 风格的 `;` 分隔 PATH 塞进
bash 会让 shell 丢掉自己的工具（`ls`、`grep` 全部找不到）。
