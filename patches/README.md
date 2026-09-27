# llama.cpp patch replay

`llama-kvmem-current.patch` is the cumulative diff against pinned `b81c99b`.
It includes the existing KVMem hooks, multimodal batch, MTP, media
parser and mtmd helper extensions, plus FP32 GDN Record/Fold for ReplaySSM.
It also fixes reasoning-budget initialization from a template's generation prefix.
`scripts/apply-patches.sh` applies it
without creating commits and checks for an already applied tree.

`0005-hip-rdna2-quantized-kv-fa-vec.patch` is @zintown's PR #58 RDNA2
quantized-KV Flash Attention dispatch fix. It selects the existing VEC kernel
for supported dimensions, avoiding the larger tile kernel's zero-occupancy
assertion. It is applied after the cumulative patch by both Linux and Windows
build entry points. F16 draft KV is not covered by this fix.

`reasoning-budget-upgrade.patch` upgrades the v0.15.0 ReplaySSM tree.
`replayssm-upgrade.patch` upgrades the preceding multimodal/query-replay tree.
`multimodal-upgrade.patch` upgrades the KVMem working tree recorded before
the 2026-09-14 implementation to the same current code. The script checks applicability before
changing files. Unrelated local changes are preserved; conflicting changes
require review.

The numbered `0001` through `0004` files are historical patches, retained for
reference. They are superseded by the cumulative diff: the old series did
not cleanly replay on the current pin and must not be applied together with it.

To check a clean extraction without changing the active submodule:

```bash
mkdir -p /tmp/kvmem-llama-patch-check
git -C llama.cpp archive b81c99b | tar -x -C /tmp/kvmem-llama-patch-check
KVMEM_LLAMA_DIR=/tmp/kvmem-llama-patch-check scripts/apply-patches.sh
KVMEM_LLAMA_DIR=/tmp/kvmem-llama-patch-check scripts/apply-patches.sh
```
