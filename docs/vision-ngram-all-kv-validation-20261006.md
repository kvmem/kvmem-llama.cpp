# 全部KV的图片＋ngram＋并发（2026-10-06）

本报告覆盖当时的五种 KV。其后 [新增 KV 准入](other-kv-admission-validation-20261006.md) 加入 fp8、rk4v4、rk4v4-e8、rk2v4-e8。当前边界见 [多后端运行](multi-backend.md)。

按用户要求，Engine、服务端参数检查及统一 Windows 启动器在本轮允许 BF16、INT8、NVFP4、K8V4、RK8V4 的文本和 resident/CPU 图片＋MTP＋ngram，活动请求上限1–8。固定及自适应MTP1–15均可用。启动并发C2–8、请求ngram16–63时仍自动降至15，并记录原值与实际值；C1保留宽度，0保持关闭。

此次移除了图片ngram的额外KV白名单，未修改复制接受、量化数学、视觉编码或缓存恢复。页预算、全局Host预算、非法值检查，以及冷磁盘限单文本、视频/overlay/快速图片prefill的既有边界保持不变。CPU视觉编码仍串行。

## 参数与构建

- 使用既有 VS2022／MSVC14.44／CUDA13.2／SM120a／Release 构建树，8个并行编译任务，编译服务端、Engine与服务端参数测试、图片及ngram并发验证程序。
- Engine、服务端参数及Windows启动器测试全部通过。覆盖五种KV、resident/CPU、固定/自适应MTP、C1–8、关闭与合法ngram宽度；验证C2–8的16/31/63自动降至15，以及非法宽度、超过八路、没有MTP等组合继续拒绝。
- 集成补丁及版本元数据已同步，`prepare-backends.py --backend ninfer --check`通过。

## 实图验证范围

使用 RTX5060Ti（UUID `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`），既有 `Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`。图片为512×512纯红/纯蓝PNG，每图实际256 merged tokens，上限1024；逻辑context8192、R128、prefill128、device profile off。图文复制用例每路B1536，全局H为512MiB×并发数。

每组完整测试含冷启动、恢复、取消一路、恢复同伴四阶段；包含独立Python源文件复制和普通神经草稿请求。检查复制严格匹配本路源文件前缀、正向ngram接受、复制冷/恢复/恢复同伴逐token一致、配置的并发重叠、取消隔离、Host上限及Main/MTP活动页归零，最后逐路检查红/蓝颜色。

三种新增格式的完整测试均通过，未放宽复制、逐token恢复或颜色断言。

| KV／视觉／MTP／Graph | 并发 | 请求→实际ngram | 恢复／取消／恢复同伴峰值 | 恢复轮数／行数 | 恢复复制接受tokens | 颜色 |
| --- | ---: | --- | --- | --- | ---: | --- |
| K8V4／CPU／自适应3／Graph | 8 | 31→15 | 8／8／8 | 104／209 | 1673 | 8条均正确 |
| BF16／resident／固定3／Graph | 2 | 31→15 | 2／2／2 | 99／114 | 239 | 2条均正确 |
| RK8V4／CPU／自适应3／Graph | 8 | 63→15 | 8／8／8 | 106／211 | 1673 | 8条均正确 |

CPU视觉冷轮不强求八路同时解码，缓存恢复、取消与恢复同伴轮均实际达到配置并发。七路复制的C8用例每路正常输出256 tokens，各接受239个ngram tokens；C2用例含一路复制和一路普通请求。复制冷/恢复/恢复同伴逐token一致，取消只影响目标一路，Main/MTP活动页及Host活动预留归零。

INT8八路CPU视觉和NVFP4两路resident视觉的先前证据见 [ngram并发补测](ngram-concurrency-validation-20261006.md)。本轮不重复全排列，没有实测其余格式/驻留/并发的全部交叉组合，不作性能或通用视觉质量结论，也不承诺30/40系硬件及36k/16k/200k默认大预算均已验收。

## 既有K8V4观察

2026-10-05 的C1／CPU／自适应MTP15／ngram63曾出现复制回答中颜色后的换行不同，颜色和复制正文相同，搬运逐字节检查通过；来源尚未确认。旧记录与原生/ngram关闭对照见 [首次图片ngram补测](vision-ngram-validation-20261005.md) 和实验根目录 `legacy-findings.md` 的L006。当前放行按用户明确要求，不将该观察表述为已修复，也不放宽完整复制用例的断言。

本轮使用新程序、相同C1／CPU／K8V4／自适应MTP15／ngram63、B768/R128/H512MiB、图片上限256重测 `copy-compare`，仍复现该观察：冷/恢复均输出91 tokens、接受63个ngram tokens；首个不同token索引为1，冷为271（两个换行）、恢复为198（一个换行），颜色和复制正文相同。Main/MTP/GDN搬运检查通过，恢复2107-token输入checkpoint，选页和自适应窗宽记录相同；严格逐token诊断失败，没有修改断言或算法。这项单路宽窗口观察与本轮三个完整通过的多路ngram15用例分别记录。

证据位于 `backend-p0p1-20261002/results/vision-ngram-all-kv-20261006/`。新服务端及 `ninfer-serve-concurrency8.exe`已同步；测试进程退出，旧服务保持关闭。旧ZIP未更新。
