#Requires -Version 5.1
# Put a model with text, Vision and MTP in the package root as model.ninfer,
# or edit the model path below. Use English paths and filenames.
# Edit --device to select a GPU.

& "$PSScriptRoot\..\..\workers\ninfer\ninfer-kvmem-server.exe" `
    "$PSScriptRoot\..\..\model.ninfer" `
    --host 127.0.0.1 `
    --port 18200 `
    --max-context 204800 `
    --max-concurrency 1 `
    --prefill-chunk 256 `
    --default-max-tokens 16384 `
    --kvmem-budget 36864 `
    --kvmem-gen-reserve 16384 `
    --kvmem-host-mib 12288 `
    --kvmem-sessions 1 `
    --kv-dtype int8 `
    --device 0 `
    --device-profile off `
    --spec mtp `
    --draft-tokens 4 `
    --adaptive-mtp `
    --ngram-draft-tokens 31 `
    --vision `
    --vision-residency cpu `
    --vision-max-merged 256
