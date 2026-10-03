# RTX 5070 Ti Windows 下载与模型准备

[中文使用说明](rtx-5070ti-windows.md) · [英文使用说明](rtx-5070ti-windows.en.md)

## 1. 软件包下载

**[下载预编译引擎、管理器与 S / XXS 模型（夸克网盘）](https://pan.quark.cn/s/28b896c4b0c0)**

下载内容是完整的 `ninfer-manage` 运行目录，包含引擎、NInfer Manager 1.4.1、配置，
以及已转换好的 **Swift S、Swift XXS** 两套 `.ninfer` 模型。两套模型约占 **20 多 GB 磁盘空间**：
网盘当前显示 S 约 11.6GB、XXS 约 10.1GB，整个运行目录约 23.5GB，建议预留至少 30GB 磁盘空间。
下载时保留完整目录结构，之后双击 `NInferManager.exe` 即可使用管理器，无需再转换这两套模型。

**成品适用范围：本仓库的预编译引擎与配套 `.ninfer` 模型成品仅支持 RTX 50 系列。**
实际测试设备为 RTX 5070 Ti 16GB。16GB 及以上独立显存的 RTX 30/40 系理论上也可尝试，
但本分支尚未实测；其他系列需要从源码构建对应架构的引擎，并按目标显卡重新准备模型。
可以把仓库、[构建说明](rtx-5070ti-windows.md#7-构建与适用范围)和本页的转换步骤交给 AI，
协助完成编译、GGUF 转 `.ninfer` 与配置。下面的转换示例对应这套 RTX 50 系配置。

| 项目 | 说明 |
|---|---|
| 管理器 | NInfer Manager 1.4.1，Release 发布版，Windows x64，自带 .NET 运行环境 |
| 引擎 | CUDA 13.4.2 / nvcc 13.4.92，Native SM120a，Release 发布版，关闭 D3D12 residency |
| 已验证设备 | RTX 5070 Ti 16 GB，Windows x64，NVIDIA 驱动 617.14 |
| 默认配置 | Swift XXS：160K / chunk 1024；Swift S：128K / chunk 256；显存策略均为 `strict` |
| 模型权重 | 已包含 Swift S 与 Swift XXS 的 `.ninfer` 成品，合计约 20 多 GB 磁盘占用 |

软件运行目录包含 `NInferManager.exe`、`engine/`、`config/`、`wwwroot/`、`docs/`、
`LICENSE` 和 `licenses/`。`model/` 用来放转换后的模型。管理器已包含监控界面和 .NET
运行环境，日常运行不用 Python，也不用 CMD/PS1 启动脚本。

用户配置、历史和日志默认写入 `%LOCALAPPDATA%\NInferManager\<安装位置ID>\`。
程序目录里的 `config/` 用于首次初始化，不同安装位置分别保存参数。
“随 Windows 登录启动”默认关闭，需要在偏好设置或托盘中开启；最后注册的那份程序生效。
详细目录结构和自启动规则见上方使用说明。

## 2. 模型下载地址

上述成品已经包含 S / XXS 模型。需要自己转换、换用其他量化规格或准备其他显卡的模型时，
再从以下来源下载 GGUF，并参考后续转换步骤。

| 模型 | GGUF 下载页面 | 配套配置与分词器来源 |
|---|---|---|
| Swift 1.5 Qwen3.8-27B GSQ-RCO | [ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF) | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/tree/main) |
| Qwen3.8-27B GSQ-RCO | [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/main) |

Swift 是经过后训练的模型；ISTA 链接提供 Qwen3.8-27B 的 GSQ-RCO 量化。
两者可用同一条转换路线，但权重不同。当前管理器自带的 XXS/S 配置对应 **Swift**。

要使用当前配置中的内置 MTP，请下载带 **`-mtp.gguf`** 后缀的文件。
它已经包含 MTP 预测头，这条转换路线不需要再下载单独的 MTP 文件。

| 选择 | 对应文件名 |
|---|---|
| Swift XXS | `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` |
| Swift S | `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| ISTA XXS | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` |
| ISTA S | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |

XXS 文件较小，留给上下文的显存更多；S 文件较大。这里的 160K/128K 是本机 Swift 的已测
配置，不代表换成其他权重后也已经完成容量或性能验证。模型分发许可见各自模型页面。

## 3. 将 GSQ GGUF 转换为 `.ninfer`

这里通常所说的“编译模型”，实际是运行仓库自带的 **Python 转换器**。
`qwen3_8_27b_gguf` 配方把原 GGUF 量化块原样导入 `.ninfer`，保留 GSQ/RCO 的逐张量量化
分配，不需要重新量化、校准，也不用重新编译 CUDA 引擎。
封装时会整理张量排列、转换少量归一化参数，并补充引擎需要的辅助信息，因此输出文件的
整体布局和大小会改变；量化权重不会重新量化成另一种精度。

转换器在源码仓库的 `tools/convert` 中，运行软件包里不附带它。
以下命令在 **Windows PowerShell** 中执行。转换使用 CPU，不占用推理显存。
Python 只在转换时需要；生成 `.ninfer` 后，日常运行仍由管理器和引擎完成。

### 3.1 准备源码和 Python

先按[主指南第 7 节](rtx-5070ti-windows.md#7-构建与适用范围)准备本版本的完整源码，再安装
**64 位 Python 3.11（含 `py` 启动器）**，打开 PowerShell。下面复用已有源码，假定它位于
当前用户的 `Documents\ninfer-all`；请将 `$sourceRoot` 改成你实际准备好的源码目录。
配套源码位于 [Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco 的 `main` 分支](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco/tree/main)，
包含 1.4.1 管理器与新显存策略。上游即使已有 GSQ 转换器，也不代表已经包含本版管理器和显存策略；
首次下载源码的命令见主指南第 7.1 节。

磁盘需要同时放下下载的 GGUF、转换出的 `.ninfer` 和 Python 环境；
最后复制进软件目录时，还会占用一份模型文件的空间。

```powershell
$ErrorActionPreference = 'Stop'
$sourceRoot = Join-Path $env:USERPROFILE 'Documents\ninfer-all'
if (!(Test-Path -LiteralPath $sourceRoot -PathType Container)) {
  throw '请先按主指南第 7 节准备本版本源码，并将 $sourceRoot 设为该目录。'
}
Set-Location -LiteralPath $sourceRoot
if (!(Test-Path -LiteralPath 'tools\convert\gguf_blocks.py')) {
  throw '此目录不是包含 GSQ 转换器的源码仓库。'
}

py -3.11 -m venv .venv-convert
if ($LASTEXITCODE -ne 0) { throw '无法创建 Python 3.11 环境。' }
$python = Join-Path $PWD '.venv-convert\Scripts\python.exe'
$hf = Join-Path $PWD '.venv-convert\Scripts\hf.exe'

& $python -m pip install numpy huggingface_hub
if ($LASTEXITCODE -ne 0) { throw '转换与下载依赖安装失败。' }
& $python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU 版 PyTorch 安装失败。' }
```

后续直接使用 `$python` 和 `$hf`，不需要激活 `.ps1` 环境脚本。
`huggingface_hub` 用于下载；转换所需的外部计算依赖是 PyTorch 和 NumPy。
这条文本 + MTP 原样导入路线不需要安装额外的 `gguf` Python 包。
各段命令按顺序在同一个 PowerShell 窗口执行，前一段有报错时先处理报错。
如果还要从源码构建引擎或托盘程序，见[使用说明中的构建步骤](rtx-5070ti-windows.md#7-构建与适用范围)。

### 3.2 选择并下载输入

下面默认选择 **Swift XXS**。需要 Swift S 时，只改 `$tier` 为 `IQ3_S`。

```powershell
$tier = 'IQ3_XXS'
$ggufRepo = 'ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF'
$metadataRepo = 'ukisai/Swift-1.5-Qwen3.8-27b'
$modelName = "Swift-1.5-Qwen3.8-27B-GSQ-RCO-$tier-mtp"
```

如果要用 **ISTA** 版本，用下面这组变量替换上面四行，然后执行共同的下载与转换步骤。

```powershell
$tier = 'IQ3_S'
$ggufRepo = 'ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF'
$metadataRepo = 'Qwen/Qwen3.8-27B'
$modelName = "Qwen3.8-27B-GSQ-RCO-$tier-mtp"
```

下载所选 GGUF，以及对应模型的五个配置与分词器文件。
**不需要下载 BF16 `.safetensors` 权重。**

| 文件 | 用途 |
|---|---|
| `config.json` | 模型结构和上下文等元数据 |
| `tokenizer.json` | 分词器词表与规则 |
| `tokenizer_config.json` | 分词器设置与特殊 token |
| `chat_template.jinja` | 模型内置的对话模板 |
| `generation_config.json` | 停止 token 等生成设置 |

```powershell
$inputDir = Join-Path $PWD "conversion-input\$modelName"
$metadataDir = Join-Path $inputDir 'metadata'

& $hf download $ggufRepo "$modelName.gguf" --local-dir $inputDir
if ($LASTEXITCODE -ne 0) { throw 'GGUF 下载失败，请先处理下载报错。' }
& $hf download $metadataRepo `
  config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json `
  --local-dir $metadataDir
if ($LASTEXITCODE -ne 0) { throw '模型配置或分词器下载失败。' }
```

这里的输入目录用于转换，和管理器的运行配置目录是两回事。
Swift 的 GGUF 配 Swift 的配置与分词器；ISTA 的 GGUF 配 Qwen 的配置与分词器，
按上面的变量组合使用即可。

### 3.3 执行转换

在仓库根目录、同一个 PowerShell 窗口继续执行：

```powershell
$outputDir = Join-Path $PWD 'converted-models'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$ggufPath = Join-Path $inputDir "$modelName.gguf"
$outputPath = Join-Path $outputDir "$modelName.ninfer"
foreach ($file in @('config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'chat_template.jinja', 'generation_config.json')) {
  if (!(Test-Path -LiteralPath (Join-Path $metadataDir $file) -PathType Leaf)) {
    throw "缺少模型资源：$file"
  }
}
if (!(Test-Path -LiteralPath $ggufPath -PathType Leaf)) { throw "缺少 GGUF：$ggufPath" }

& $python -u -m tools.convert `
  --model $metadataDir `
  --recipe qwen3_8_27b_gguf `
  --source "gguf=$ggufPath" `
  --components text,mtp `
  --proposal `
  --device cpu `
  --rows-per-chunk 512 `
  --name $modelName `
  --out $outputPath
if ($LASTEXITCODE -ne 0) { throw '转换失败，请根据上方报错处理；不要继续安装不完整模型。' }
```

| 参数 | 通俗说明 |
|---|---|
| `--model` | 配置和分词器所在目录，不是 GGUF 文件 |
| `--recipe qwen3_8_27b_gguf` | 使用保留原 GGUF 量化块的 Qwen3.8-27B 转换配方 |
| `--source "gguf=..."` | 指向下载的 GSQ GGUF 文件 |
| `--components text,mtp` | 导入文本权重与内置 MTP |
| `--proposal` | 与本机已有制品一致，额外保存供 `--lm-head-draft` 使用的精简词表预测头；不是 MTP 的必需条件 |
| `--device cpu` | 使用 CPU 转换，不依赖 CUDA 版 PyTorch |
| `--rows-per-chunk 512` | 转换时分批处理的行数，与推理时的 prefill chunk 无关 |
| `--out` | 新的输出文件路径 |

输出路径若已有 `.ninfer` 或对应的 `.ninfer.conversion.json`，转换器会拒绝覆盖；
重新转换时，换一个空目录或新文件名。转换还会生成 `.ninfer.conversion.json` 报告，
运行模型不需要这份报告。
使用 `--proposal` 时，转换器会读取仓库自带的
`tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` 选择预测词表，不需要另跑校准。
完成后终端会输出 `wrote ...`；本机 Swift XXS/S 已使用上述 CPU、512 行、`text,mtp`、
`--proposal` 路线生成并运行成功。ISTA 配方与输入格式也已核对，但本机未重新转换和实测
这两个 ISTA 文件。

如果始终关闭 `--lm-head-draft`，可以省略 `--proposal`，这不会关闭内置 MTP。
如果下载的是不带 MTP 的普通 GGUF，应改为 `--components text`，并另存不使用推测解码的
启动配置：移除 `--spec`、`--draft-tokens` 和 `--ngram-draft-tokens`，手动设置过的
`--lm-head-draft`、`--adaptive-mtp`、`--mtp-attention-window` 也要移除。
只清空 `--spec`，却保留正数的 MTP／复制草稿或其他依赖它的参数，会导致检查报错；
`--lookup-ngram` 可保留在配置里，但当前执行路径仅在启用 MTP 时生效。

本例没有导入 vision 组件，生成的模型不包含图像编码能力。

### 3.4 放入管理器

先将软件下载到当前用户的 `Downloads\qwen27b`。如果解压在其他位置，只改下面的
`$packageRoot`。继续在原 PowerShell 窗口执行，按转换报告复制主文件及全部分卷，保留文件名。

```powershell
$packageRoot = Join-Path $env:USERPROFILE 'Downloads\qwen27b'
if (!(Test-Path -LiteralPath (Join-Path $packageRoot 'NInferManager.exe') -PathType Leaf)) {
  throw '此目录没有 NInferManager.exe，请先解压软件或修改 $packageRoot。'
}
$modelDir = Join-Path $packageRoot 'model'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
$report = Get-Content -LiteralPath ($outputPath + '.conversion.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($file in $report.files) {
  $target = Join-Path $modelDir ([IO.Path]::GetFileName($file.path))
  if (Test-Path -LiteralPath $target) { throw "模型文件已存在，已停止复制：$target" }
}
foreach ($file in $report.files) {
  $target = Join-Path $modelDir ([IO.Path]::GetFileName($file.path))
  [IO.File]::Copy($file.path, $target, $false)
}
```

然后在管理器中完成选择：

1. 双击 `NInferManager.exe`，从托盘打开“管理模型”。管理器会自动扫描，也可点“重新扫描”。
2. Swift XXS/S 的默认文件名与上面示例一致，分别选用对应的 XXS 或 S 启动配置。
3. 使用其他名称或 ISTA 权重时，点“新建配置”，填写显示名称，选择实际模型文件并保存。
   新建配置会带入现有参数，应逐项确认：XXS/S 的上下文和 chunk 是否合适、模型 ID 是否是
   客户端要使用的名称、文件是否确实带 MTP。需要便于区分 ISTA 时，可使用
   `qwen3.8-27b-gsq-rco` 作为模型 ID；首次加载应重新确认可用显存。
4. 从托盘的“启动模型”二级菜单或网页启动。需要登录后加载这个配置时，在“偏好设置”中选它
   作为默认启动配置，并按需要开启“随 Windows 登录启动”和“自动加载默认模型”。

要再转换另一种模型或量化档位，从 **3.2** 重新设置变量并依次执行 **3.3、3.4** 即可。
输入 GGUF 和 `.ninfer.conversion.json` 留在转换目录保存即可，运行目录只需要 `.ninfer`
及它的分卷。

运行时使用的数据目录 `config/chat_template.jinja` 与转换时封装的原始模板是两个独立
资源；默认管理器参数会选择前者。软件使用方法、显存策略、启动参数和性能实测见顶部指南。
