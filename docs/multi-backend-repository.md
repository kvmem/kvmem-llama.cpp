# Multi-backend repository

The `feat/multi-backend-framework` branch is the development line for replacing the
llama.cpp-specific product with a multi-backend KVMem framework. Both engines are
Git submodules referencing existing upstream projects. The parent commit pins the
core, adapters, engine revisions and integration patches together.

| Path | Ownership |
|---|---|
| `kvmem/` | Portable memory contracts, native KV payloads, retrieval and Host storage |
| `src/adapter/`, `tools/` | llama.cpp product adapter and serving |
| `backends/llamacpp/` | Official ggml-org/llama.cpp submodule at a fixed commit |
| `backends/ninfer/` | Existing Ryan ninfer fork submodule at a fixed commit |
| `backends/patches/` | KVMem changes relative to each fixed engine revision |
| `scripts/windows/` | Build, launcher and packaging entry points |
| `backends/versions.json` | Baseline revisions, licenses, patch hashes and prepared tree hashes |

Each backend keeps its existing license and third-party notices. Source-only
upstream vocabulary and test fixtures are included; inference model weights,
builds, logs, session snapshots and personal DSH configuration are not.

## Checkout and preparation

```sh
git clone --recurse-submodules --branch feat/multi-backend-framework \
  https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
python scripts/prepare-backends.py
python scripts/prepare-backends.py --check
```

Python 3.9+ and Git are required for source preparation. An existing checkout can
run the same script: it initializes missing submodules, verifies the pins and
applies each integration patch once. `--backend llamacpp` or `--backend ninfer`
prepares only the selected engine. `--check` never fetches or changes sources; it
checks the full prepared tree, including patch-added files. The patches reconstruct
the previously tested source trees exactly, including the ninfer generated-endpoint
cache fix. They are not applied by CMake itself.

Prepared submodules normally appear modified in `git status` because the integration
is a local patch over the upstream pin. Preparation preserves developer changes
and fails on conflicts; it does not reset a backend or switch to an upstream branch.
Strict `--check` rejects additional source edits until the patch and manifest are
updated. Core-only builds need neither engine.

GitHub's automatic source ZIP omits submodule contents; use Git for development.
`scripts/rocm/export-source.py` can export a committed, fully prepared source bundle
with both engines and a per-file manifest. Such bundles can be built without Git
metadata; preparation verifies their backend files against that manifest.

## Builds

The engines run as separate workers, so they use separate CMake build directories.
This avoids sharing compiler flags, target names or device allocations between
backends. It does not require separate Git branches.

On Windows, use VS2022 C++ Build Tools, CMake/Ninja and the CUDA toolkit appropriate
for the selected engine. The locally qualified ninfer route uses CUDA 13.2 and
`120a` with native SM120 enabled for the RTX 5060 Ti. Other GPU architectures need
their own qualification. The wrapper accepts `core`, `llamacpp`, `ninfer` or `all`:

```powershell
# Run these commands in a VS2022 x64 developer PowerShell.
# Model-free portable core and host tests.
./scripts/windows/build-backends.ps1 -Backend core
ctest --test-dir build-backends/core --output-on-failure

# llama.cpp worker. Existing scripts/windows/build.ps1 remains available.
./scripts/windows/build-backends.ps1 -Backend llamacpp -CudaArchitectures 120a

# ninfer worker. Set VCPKG_ROOT to your own vcpkg checkout first.
./scripts/windows/build-backends.ps1 -Backend ninfer -CudaArchitectures 120a `
  -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' `
  -CMakeArgs @("-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
```

The build wrapper prepares its selected backend automatically. `-Python` selects
the Python executable. `-DryRun` prints the resolved argument arrays without fetching
or applying patches; `-ConfigureOnly` and `-BuildOnly`
separate configure and compilation. Backend-specific CMake arguments should be
passed with the matching `-Backend`, not indiscriminately to `all`.

The wrapper configures dependencies from the checked-in ninfer vcpkg manifest.
An existing compatible dependency installation can be passed with
`-DVCPKG_INSTALLED_DIR=... -DVCPKG_MANIFEST_INSTALL=OFF`; no local prebuilt ninfer
operator libraries are needed. The backend CMake default resolves the portable
core at `../../kvmem` relative to `backends/ninfer`.

The equivalent portable-core commands on Linux are:

```sh
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF
cmake --build build-core --parallel 2
ctest --test-dir build-core --output-on-failure
```

For direct CMake use, prepare the selected backend first. For ninfer, configure
`-S backends/ninfer` in a separate directory. Its own
`docs/` and `vcpkg.json` describe the native build dependencies. Existing
llama.cpp CUDA/HIP scripts and the UI builder prepare the referenced sources before
building. They do not change the pinned upstream revisions.

The resulting workers are `build-backends/llamacpp/bin/llama-kvmem-server.exe` and
`build-backends/ninfer/apps/ninfer-serve.exe`. Pass their paths to
`scripts/windows/start-backend.ps1` or `package-backends.ps1`; the latter's
`-NinferSource` is `backends/ninfer`. No model is included in the package.

## Development and release

Backend changes are recorded in `backends/patches/<backend>-kvmem.patch` in the same
commit as matching core/adapter changes. Export a binary/full-index Git diff against
the backend's `HEAD`, including any newly added files, then update `patch_sha256`
and `patched_tree` in `backends/versions.json`. Keep patch output byte-for-byte
(avoid PowerShell text redirection changing its encoding). The tree hash is the
result of staging those backend source changes and running `git write-tree`; no
backend commit is needed. Validate with `prepare-backends.py --check` and a fresh
checkout before publishing.

For upstream updates, deliberately update the submodule gitlink and `base_commit`,
rebase the integration patch onto that revision, update hashes and run affected
checks. Do not follow an unpinned branch. Root `patches/` remains historical material;
the active patches live under `backends/patches/`.

The current branch is a development candidate. P6-P10 implementation and selected
tests are complete, including the short-input history-reuse fix, but the complete
release matrix is not finished. See [validation status](multi-backend-validation.md)
and [supported backend combinations](multi-backend.md). Once accepted, merge into
`master` through normal Git history; do not force-replace the main branch.
