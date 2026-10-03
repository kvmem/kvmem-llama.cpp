# 轻量 Web UI 接入设计

状态：首版已实现并完成本地验证；当时基于 llama.cpp pin `b81c99b` 的 `tools/ui` 和 KVMem 自定义服务。当前 master 已迁移到 llama.cpp `v0.5.0`。使用和测试命令见 [ui/README.md](../ui/README.md)。

目标是让用户启动 IQ3/IQ4 后直接在浏览器测试文本、思考与图片。复用上游页面组件和静态构建，不移植上游 server 的调度器，也不实现完整 UI 的后端功能。

## 1. 首版范围

保留：

- 文本输入、流式输出、Markdown/代码渲染、思考内容折叠。
- 图片上传/粘贴；只有实际加载视觉头时显示入口。
- 浏览器本地历史、新建会话、切换历史会话。复用上游 IndexedDB，不新增后端数据库。
- 参数小面板：跟随服务器默认、思考开关、模型思考强度、思考预算、最大输出、temperature/top_p/top_k/min_p。
- 停止当前生成，以及生成 token 数、服务端 decode tok/s。

首版隐藏：历史分支编辑/重新生成、续写 assistant、Agent/MCP、服务端文件和终端工具、模型装卸、音视频、PDF、slot 监控、运行中跳过思考、断线续流、PWA 离线模式。

隐藏不只是移除按钮：对应初始化、探测、轮询、自动重连也必须停用。现有 API 客户端的工具调用等能力不受这个 UI 功能范围影响。

## 2. 部署与复用方式

```text
浏览器（上游 UI 的轻量配置）
  ├── GET /、静态资源         → cpp-httplib 静态目录
  ├── GET /props             → 模型能力与默认参数快照
  ├── GET /v1/models         → 现有模型列表
  └── POST /v1/chat/completions → 现有 KVMem 推理与 SSE
```

- 页面和 API 共用 18200 端口，不引入代理或第二个后台进程。
- 复用上游 SvelteKit 静态产物；当前上游配置使用 hash 路由，不需要服务端实现会话页面路由。
- 实现采用 `ui/lightweight/` 中的独立轻量路由入口，复用上游 Markdown/代码、输入和数据库组件，不启动完整页面的生命周期。构建脚本在隔离副本中覆盖入口并禁用 PWA，不修改 llama.cpp 工作树。
- 运行时只需要静态文件，不需要 Node.js。开发和发布构建才执行 `npm ci`、构建与前端检查。
- 发布包放在 `share/kvmem/ui/`；源码构建可放在 `build/share/kvmem/ui/`。服务按可执行文件位置定位，不依赖启动时的工作目录。
- 有随包资源时默认提供页面，无资源时继续正常提供 API。只增加 `--ui-dir PATH` 覆盖目录、`--no-ui` 禁用页面；明确指定的目录无效则报告启动错误。
- 无须在 IQ3/IQ4 启动命令中再堆默认参数；启动器只转发这两个可选设置。
- 复用 cpp-httplib 的目录挂载，静态目录限定为 UI 产物，不把项目根目录映射出去。API 路由优先，未知 API 保持 404，不能回退成 HTML。
- `index.html` 使用重新验证缓存策略；带内容 hash 的资产允许长期缓存。轻量版不注册 service worker，避免升级后继续加载旧 API 客户端。

## 3. 最小接口清单

| 接口 | 处理 |
|---|---|
| `GET /`、静态文件 | 新增静态托管 |
| `GET /props` | 唯一新增的业务 JSON 接口，只读 |
| `GET /health` | 沿用，用于连接检查 |
| `GET /v1/models` | 沿用，必要时追加兼容的模型状态字段 |
| `POST /v1/chat/completions` | 沿用，补必要参数兼容与计时字段 |

不增加 `/slots`、`/tools`、`/models/load`、`/models/unload`、`/models/sse`、`/v1/stream`、`/v1/streams/lookup`、`/v1/chat/completions/control`。也不做返回假成功的占位实现；前端轻量配置不调用它们。

### `/props`

沿用上游页面真正消费的字段，不伪造整个 llama-server 状态对象：

- `role: "model"`、`total_slots: 1`，表示单模型/单个推理状态。
- `default_generation_settings.n_ctx`：逻辑上下文长度，不能填 GPU 检索预算。
- `default_generation_settings.params`：实际生效的采样参数、默认输出上限等。来自同一套服务端默认值合并逻辑，禁止复制 IQ3 的固定数字。
- `modalities.vision`：视觉头实际初始化成功才为 true；audio/video 为 false。
- `chat_template`、`build_info`：按前端实际需要提供；不暴露本机模型绝对路径。
- 小型 `kvmem` 扩展：`generation_limit`（单轮生成上限）和 `defaults`（思考开关、预算、当前模板 kwargs）。UI 不据此创建完整功能发现协议。

思考和非思考采样默认值不同；扩展中分别提供有效默认值，供页面展示。用户选“跟随服务器”时请求省略相关字段，由服务端最终决定。

这些信息在初始化后生成不可变快照。读取 `/props` 不等待持有整个生成阶段的 `st.mu`，不读取正在修改的 KV，也不触发 tokenize/prefill/GPU 同步。

上游 `/v1/models` 客户端有读取 `status.value` 的路径。首版应补充兼容的 `name`、`status: {"value":"loaded"}`，或在轻量适配层做单模型归一化；选前者以减少前端分支。保持原来的 `object`、`data`、`id` 不变，不启动模型状态事件流。

## 4. 请求参数

前端只发送当前支持的参数；不直接沿用上游所有设置项。

| 页面选项 | 请求 |
|---|---|
| 跟随服务器 | 省略该字段，不能把浏览器内置默认值强行发给服务端 |
| 思考开关 | `chat_template_kwargs.enable_thinking` |
| 模型思考强度 | 顶层 `reasoning_effort` |
| 思考 token 预算 | `reasoning_budget_tokens`，含义独立于强度 |
| 最大输出 | 正整数 `max_tokens`，包含思考与正文 |
| 采样 | 明确修改后才发送 temperature/top_p/top_k/min_p |
| 图片 | 现有 `image_url` data URL 格式，保留完整历史图片内容 |

当前上游 UI 把 medium 映射成 2048 token、low 映射成 512 token。轻量版必须移除这个联动。当前 GSQ 模板可提供 low/medium/xhigh 快捷项；未知自定义模板不硬套这些档位，保持默认或输入模板支持的值。UI 一种参数只产生一种字段，不同时写顶层与 kwargs 的 effort。

最大输出不提供“无限”。默认跟随当前服务的输出设置，UI 按 `generation_limit` 校验。服务端兼容规则：省略或 `max_tokens=-1` 使用服务默认值并受生成预留限制；0、其他负数、非整数和显式超过上限的值返回清晰的 400。设置 `max_completion_tokens` 时遵循相同校验，两个别名的优先级保持现有实现。

这项兼容只解决现有上游 UI 的 `-1` 语义，不引入自动修改检索预算、自动扩容或 ring buffer。

不发送 `reasoning_control`、`continue_final_message`、流恢复标识和不支持的采样选项。高级任意 JSON 编辑器不纳入首版。

## 5. 历史与 KV 复用

服务端继续只有一份活动推理状态。浏览器中的多条会话是历史记录，不代表服务端同时驻留多套 KV。

- 正常追加：发送完整 messages，保留消息、思考字段和图片的稳定序列化方式；不发送 `cache_reset`，复用现有前缀/检查点/检索逻辑。
- 新建或切换会话：仍发送所选完整历史，由现有前缀和检查点判断能复用多少。页面切换本身不调用后端，也不清空 KV。
- 当前 `run_prefill_multimodal` 已在找不到 recurrent checkpoint 时作为缓存未命中重建；沿用这一行为，不恢复旧版报错，也不再实现第二套会话归属判断。
- 不承诺切换历史后零重算；单活动缓存遇到不兼容历史时重算是预期行为。
- 首版不提供编辑、分支和重新生成入口，避免扩大缓存行为覆盖范围。停止后部分响应标记为“已中止”，下次按实际保留的消息续接，由后端缓存逻辑保证一致性。
- UI 每个页面一次只发一个生成请求；其他客户端继续遵循现有互斥逻辑。服务器等待锁后应检查请求是否已断开，避免已取消的排队请求之后仍开始推理。

不增加会话 ID 到 KV 的映射，不持久化 KV，不增加后台自动总结或标题生成请求；否则浏览器打开/切换页面也可能挤掉当前活动缓存。

## 6. 流式输出、停止与错误

继续使用标准 SSE，处理 `delta.content`、`delta.reasoning_content`、finish_reason、usage 和 `[DONE]`。流意外断开时标记中断，不自动重发生成请求。

停止按钮只调用 `AbortController.abort()` 关闭当前请求，不调用上游 `DELETE /v1/stream` 或 control API。后端复用 `stream_peer_gone` / `stream_heartbeat`，在现有 prefill 批次、普通 decode 或 MTP 校验边界停止并完成回滚/清理。不能承诺中断正在执行的 GPU kernel。

需验证：等待锁时取消、prefill 时取消、普通/MTP decode 时取消，以及取消后的下一轮请求。页面进入“正在停止”状态时禁用重复发送；本地 fetch 清理完成后允许下一条，但不把本地取消误当成 GPU 已停止的确认。下一条可能短暂等待服务端清理并释放现有互斥锁，不靠新增 `/slots` 轮询判断完成。

前端错误适配同时识别现有 `{"error":"..."}` 和上游 `{"error":{"message":"..."}}`；HTTP 错误和 SSE 内错误均展示可读信息。接到错误后即使还有 `[DONE]`，也不能标记成成功。

工具调用 API 保持不变，但这个测试页不发起工具执行，也不主动发送 tools。意外收到 tool_calls 时可显示只读内容/说明，不进入自动工具循环。

## 7. 计时：首版只显示 decode 速度

复用现有生成计时点，在最终 SSE usage 块和非流式 JSON 中增加可选 `timings`：

```json
{
  "timings": {
    "predicted_n": 256,
    "predicted_ms": 8000.0
  }
}
```

上例只是协议示意。页面显示 `predicted_n / (predicted_ms / 1000)`，即 32 tok/s。计数包含实际生成的思考、正文等 token；不把拒绝的草稿 token 算作输出。

- 计时在 prefill 完成后开始，在现有生成结束计时点冻结；不包含 prefill、生成后的缓存提交和 HTTP 收尾。
- 沿用现有计时点，不增加逐 token 的计时消息、GPU synchronize 或全局性能锁。
- UI 显示“生成速度”，不拿浏览器请求总时长反推 decode。中断请求未收到最终计时则显示无数据。
- usage 和现有客户端行为保持兼容，timings 为追加字段。
- 首版隐藏上游 prompt speed 项：直接用完整历史 token 数除以本轮 prefill 时间会误导。有效/首次计算 prefill 速度之后如需展示，应复用 canary 的统计定义，作为独立小改动。

## 8. 构建与维护

- UI 基础版本与 llama.cpp pin 一致；入口覆盖文件独立维护，构建产物记录基线和文件 hash，保留上游许可。
- 在隔离的前端源码副本上应用轻量补丁并构建，避免污染开发者的 llama.cpp 工作树或夹带其中的实验修改。
- 页面组件、图片处理、Markdown 和本地数据库直接复用上游；不另写聊天框。
- 静态产物用独立 UI 版本/源码 hash 标识，加入发布包 manifest 和 SHA-256；API-only 构建不用安装 npm 依赖，也不用构建 UI。
- 挂载页面只读取静态资源，不触发模型/GPU工作；预计主要新增浏览器内存和磁盘产物，不增加模型/KV 显存分配。前端体积须实际构建后记录，不预估固定数值。

## 9. 分步落地和验收

1. **后端信息与托管**：静态目录、`/props`、模型状态附加字段。验证无 UI 时 API 正常、页面能初始化、生成中读取 props 不阻塞、视觉能力与实际加载一致。
2. **轻量前端配置**：关闭不支持的初始化与功能入口，接参数白名单和普通 SSE。验证页面加载与新会话不会发隐藏推理请求，网络面板无 `/slots`、续流、tools 等请求。
3. **请求兼容与生成计时**：补输出上限校验和 timings。验证默认继承、思考强度/预算分离、流式/非流式与日志计数和耗时一致。
4. **真实会话验证**：文本多轮、图片后追问、切换历史、停止后继续；覆盖 MTP3 和关闭 MTP。确认普通续接不触发无条件 reset，缓存未命中能正常重建。
5. **打包**：随包提供静态目录，在没有 Node.js、没有开发路径的解压目录启动；IQ3/IQ4 启动脚本保持原有体验。按现行规则，出现大量 swap 即停测并记录。

服务端不需要等完整 UI 功能补齐才能交付。首版完成的边界就是：打开页面，能调受支持参数、发文字/图片、看思考/回答、停止生成并查看可靠的生成速度。

## 源码依据

- `tools/llama-kvmem-server.cpp`：当前路由、请求解析、互斥锁、SSE、断线检查与生成计时。
- `tools/llama-kvmem-driver.cpp`：前缀/检查点复用、缓存未命中重建与回滚。
- `backends/llamacpp/tools/ui/src/lib/services/props.service.ts`、`models.service.ts`：页面初始化信息。
- `backends/llamacpp/tools/ui/src/lib/services/chat.service.ts`：参数映射、SSE、控制/续流等上游依赖。
- `backends/llamacpp/tools/ui/src/lib/constants/reasoning-effort.constants.ts`：上游 effort 到 token 预算的映射。
- `backends/llamacpp/tools/ui/svelte.config.js`、`tools/server/server-http.cpp`：静态产物/hash 路由及 cpp-httplib 托管。

## 首版验证记录

- Svelte 类型检查：0 errors / 0 warnings；16 项启动器测试通过。
- 小模型 MTP 与普通 decode：props、输出上限验证、SSE/JSON、usage/timings 一致性通过。
- Chromium：文本多轮、历史刷新恢复、HTML 代码渲染、取消和后续请求通过，没有 tools/slots/续流请求。
- IQ3 27B（5060 Ti、block 32、medium、MTP3/ReplaySSM、GPU Q8 视觉头）：真实浏览器图片上传及后续追问均正确；图片历史保留，无强制 reset。
- 本地服务已用隔离 UI 构建重启；没有修改已准备好的 rc1 发布附件。
