# Multi-backend repository

The `feat/multi-backend-framework` branch is the development line for replacing the
llama.cpp-specific product with a multi-backend KVMem framework. Both engine source
trees are ordinary directories in this repository. One commit pins the core,
adapters, backends, tests and launchers together. No nested Git checkout or patch
replay is needed after cloning.

| Path | Ownership |
|---|---|
| `kvmem/` | Portable memory contracts, native KV payloads, retrieval and Host storage |
| `src/adapter/`, `tools/` | llama.cpp product adapter and serving |
| `llama.cpp/` | Vendored llama.cpp including the required integration changes |
| `backends/ninfer/` | Vendored ninfer including its KVMem integration |
| `scripts/windows/` | Build, launcher and packaging entry points |
| `backends/versions.json` | Upstream baseline revisions and license locations |

Each backend keeps its existing license and third-party notices. Source-only
upstream vocabulary and test fixtures are included; inference model weights,
builds, logs, session snapshots and personal DSH configuration are not.

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

`-DryRun` prints the resolved argument arrays; `-ConfigureOnly` and `-BuildOnly`
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

For ninfer, configure `-S backends/ninfer` in a separate directory. Its own
`docs/` and `vcpkg.json` describe the native build dependencies. Existing CUDA/HIP
llama.cpp build scripts operate on the vendored source without applying patches.

The resulting workers are `build-backends/llamacpp/bin/llama-kvmem-server.exe` and
`build-backends/ninfer/apps/ninfer-serve.exe`. Pass their paths to
`scripts/windows/start-backend.ps1` or `package-backends.ps1`; the latter's
`-NinferSource` is `backends/ninfer`. No model is included in the package.

## Development and release

Backend edits belong in the same commit as the matching core/adapter changes.
For upstream updates, compare against the revision in `backends/versions.json`,
import reviewed source changes, update that record and run the affected checks.
The `patches/` directory is historical material, not a second source of truth.

The current branch is a development candidate. P6-P10 implementation and selected
tests are complete, including the short-input history-reuse fix, but the complete
release matrix is not finished. See [validation status](multi-backend-validation.md)
and [supported backend combinations](multi-backend.md). Once accepted, merge into
`master` through normal Git history; do not force-replace the main branch.
