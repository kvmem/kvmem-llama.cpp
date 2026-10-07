#Requires -Version 5.1
# Put a model with text, Vision and MTP in the package root as model.ninfer,
# or edit the model path below. Use English paths and filenames.
# Select GPUs by UUID: -Gpu 'GPU-uuid-a,GPU-uuid-b'. The first GPU holds Vision/MTP.
# Optional -StageLayers 30,34 sets the layer split; otherwise the engine plans it.
# With -Concurrency 2..8, ngram widths above 15 automatically reduce to 15; raise -RetainedSessions as needed.
# -SinkTokens 0 keeps one 64-token page. -RecentTokens must be >=64; default 64 keeps the newest page.
param(
    [string]$Gpu = $(if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { '0' }),
    [uint32[]]$StageLayers = @(),
    [ValidateRange(0, 2147483647)][int]$SinkTokens = 0,
    [ValidateRange(64, 2147483647)][int]$RecentTokens = 64
)
& "$PSScriptRoot\start-backend.ps1" -Backend ninfer `
    -Model "$PSScriptRoot\..\..\model.ninfer" -Gpu $Gpu -StageLayers $StageLayers `
    -Port 18200 -Context 204800 -Concurrency 1 -Prefill 256 -MaxTokens 16384 `
    -Budget 36864 -Reserve 16384 -HostMiB 12288 -RetainedSessions 1 -KvType int8 `
    -MtpDrafts 4 -AdaptiveMtp -NgramDrafts 31 -Vision -VisionResidency cpu -VisionTokens 1024 `
    -ThinkingBudget 8192 -SinkTokens $SinkTokens -RecentTokens $RecentTokens
