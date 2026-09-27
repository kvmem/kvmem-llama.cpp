@echo off
REM ============================================================================
REM  KVMem + llama.cpp  --  ROCm / HIP build  (Windows)
REM ============================================================================
REM  Mirrors scripts/build-cuda.sh, but drives ggml's HIP backend instead of the
REM  CUDA backend. Everything lands in <repo>\build-hip-win -- this script never
REM  copies into another llama.cpp deployment, so any llama-server already on
REM  PATH is left untouched.
REM
REM  Usage (from cmd.exe or an x64 Native Tools Command Prompt for VS 2022):
REM      scripts\build-hip.bat
REM
REM  Optional environment overrides:
REM      GPU_TARGETS   optional override; otherwise detected by amdgpu-arch
REM      JOBS          default 12  (ninja parallelism)
REM      ROCM          default %ROCM_PATH% or %HIP_PATH%
REM      BUILD_DIR     default <repo>\build-hip-win
REM      FRESH         set to 1 to reset the CMake cache (CMake >= 3.24)
REM
REM  NOTE: MSVC 14.44 (VS 2022) is required. 14.51 (VS 2026) turns the two-arg
REM  <cmath> builtins (isgreater, isless, ...) into constexpr, which AMD clang
REM  then rejects in HIP device code. See hip-build-quickref.txt.
REM ============================================================================

setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

REM --- apply the maintained llama.cpp integration patch ----------------------
REM The submodule stores the clean upstream pin; KVMem's llama.cpp hooks live in
REM patches\llama-kvmem-current.patch. Keep this idempotent so a fresh clone and
REM an already-patched developer tree follow the same build path.
set "LLAMA_DIR=%ROOT%\llama.cpp"
set "LLAMA_PATCH=%ROOT%\patches\llama-kvmem-current.patch"
if not exist "%LLAMA_DIR%\CMakeLists.txt" (
    echo [error] llama.cpp submodule is missing. Run git submodule update --init.
    exit /b 1
)
git -C "%LLAMA_DIR%" apply --ignore-space-change --reverse --check "%LLAMA_PATCH%" >nul 2>&1
if errorlevel 1 (
    git -C "%LLAMA_DIR%" apply --ignore-space-change --check "%LLAMA_PATCH%"
    if errorlevel 1 (
        echo [error] The maintained llama.cpp patch does not apply cleanly.
        exit /b 1
    )
    git -C "%LLAMA_DIR%" apply --ignore-space-change "%LLAMA_PATCH%"
    if errorlevel 1 exit /b 1
)

set "RDNA2_PATCH=%ROOT%\patches\0005-hip-rdna2-quantized-kv-fa-vec.patch"
git -C "%LLAMA_DIR%" apply --ignore-space-change --reverse --check "%RDNA2_PATCH%" >nul 2>&1
if errorlevel 1 (
    git -C "%LLAMA_DIR%" apply --ignore-space-change --check "%RDNA2_PATCH%"
    if errorlevel 1 exit /b 1
    git -C "%LLAMA_DIR%" apply --ignore-space-change "%RDNA2_PATCH%"
    if errorlevel 1 exit /b 1
)

if "%JOBS%"=="" set "JOBS=12"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%ROOT%\build-hip-win"

if "%ROCM%"=="" set "ROCM=%ROCM_PATH%"
if "%ROCM%"=="" set "ROCM=%HIP_PATH%"
if "%ROCM%"=="" (
    echo [error] ROCm root not set. Set ROCM, ROCM_PATH or HIP_PATH first.
    exit /b 1
)

set "ROCM_BIN=%ROCM%\bin"
if not exist "%ROCM_BIN%\clang++.exe" set "ROCM_BIN=%ROCM%\llvm\bin"
if not exist "%ROCM_BIN%\clang++.exe" set "ROCM_BIN=%ROCM%\lib\llvm\bin"
if not exist "%ROCM_BIN%\clang++.exe" (
    echo [error] AMD clang not found under "%ROCM%".
    echo         Set ROCM to your ROCm root before running this script.
    exit /b 1
)

REM --- MSVC environment -------------------------------------------------------
if not "%VSCMD_ARG_TGT_ARCH%"=="x64" (
    if "%VCVARS%"=="" (
        set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
        if exist "!VSWHERE!" for /f "usebackq tokens=*" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
        if defined VSROOT set "VCVARS=!VSROOT!\VC\Auxiliary\Build\vcvars64.bat"
    )
    if not exist "!VCVARS!" (
        echo [error] Could not locate vcvars64.bat through vswhere.
        echo         Install the Visual Studio 2022 C++ Build Tools, set VCVARS,
        echo         or run from an x64 Native Tools Command Prompt.
        exit /b 1
    )
    echo [1/3] MSVC environment: "!VCVARS!"
    call "!VCVARS!" >nul
    if errorlevel 1 exit /b 1
 ) else (
    echo [1/3] Reusing the current x64 MSVC environment.
)

REM --- ROCm clang on PATH (cmake resolves it at configure time only) ----------
set "PATH=%ROCM_BIN%;%ROCM%\bin;%PATH%"


where cmake >nul 2>&1
if errorlevel 1 (
    echo [error] cmake not found on PATH.
    exit /b 1
)
where ninja >nul 2>&1
if errorlevel 1 (
    echo [error] ninja not found on PATH.
    exit /b 1
)

set "FRESH_ARG="
if "%FRESH%"=="1" set "FRESH_ARG=--fresh"
set "TARGET_ARGS="
if not defined GPU_TARGETS if defined AMDGPU_TARGETS set "GPU_TARGETS=%AMDGPU_TARGETS%"
if defined GPU_TARGETS set TARGET_ARGS=-DGPU_TARGETS="%GPU_TARGETS%" -DCMAKE_HIP_ARCHITECTURES="%GPU_TARGETS%"

REM AMD's portable Windows HIP packages can omit llvm-ranlib.exe. LLVM tools
REM select their mode from argv[0], so a build-local copy of llvm-ar.exe named
REM llvm-ranlib.exe supplies the standard ranlib entry point without modifying
REM the ROCm installation or relying on a shell wrapper that Ninja cannot spawn.
set "KVMEM_TOOL_DIR=%BUILD_DIR%\kvmem-tools"
if not exist "%KVMEM_TOOL_DIR%" mkdir "%KVMEM_TOOL_DIR%"
if not exist "%KVMEM_TOOL_DIR%\llvm-ranlib.exe" copy /y "%ROCM_BIN%\llvm-ar.exe" "%KVMEM_TOOL_DIR%\llvm-ranlib.exe" >nul
if not exist "%KVMEM_TOOL_DIR%\llvm-ranlib.exe" (
    echo [error] Could not prepare llvm-ranlib.exe in "%KVMEM_TOOL_DIR%".
    exit /b 1
)

echo.
echo   repo        : %ROOT%
echo   build dir   : %BUILD_DIR%
echo   ROCm        : %ROCM%
if "%GPU_TARGETS%"=="" (echo   GPU_TARGETS : auto-detect) else (echo   GPU_TARGETS : %GPU_TARGETS%)
echo   jobs        : %JOBS%
echo.
echo [2/3] Configuring (HIP backend, CUDA backend forced off)...
cmake %FRESH_ARG% %TARGET_ARGS% -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++ ^
    -DCMAKE_AR="%ROCM_BIN%\llvm-ar.exe" ^
    -DCMAKE_RANLIB="%KVMEM_TOOL_DIR%\llvm-ranlib.exe" ^
    -DGGML_HIP=ON ^
    -DROCM_PATH="%ROCM%" ^
    -DGGML_HIP_UMA=OFF ^
    -DGGML_NATIVE=OFF ^
    -DGGML_OPENMP=OFF ^
    -DKVMEM_ENABLE_NVME=OFF ^
    -DGGML_VULKAN=OFF ^
    -DKVMEM_BUILD_LLAMA=ON ^
    -DLLAMA_KVMEM=ON ^
    -DLLAMA_KVMEM_ROOT="%ROOT%"
if errorlevel 1 (
    echo [error] configure failed.
    exit /b 1
)

REM --- verify the build will actually be optimised -------------------------
REM A build directory that was first configured before the compiler could be
REM identified caches an EMPTY CMAKE_CXX_FLAGS_RELEASE. The build then succeeds
REM but every translation unit, HIP device code included, is compiled without
REM -O3 and inference runs roughly 400x slower. CMakeLists.txt fails the
REM configure for this too; the check here just gives a clearer message.
findstr /C:"CMAKE_CXX_FLAGS_RELEASE:STRING=-O" "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [error] CMAKE_CXX_FLAGS_RELEASE has no -O flag in:
    echo             "%BUILD_DIR%\CMakeCache.txt"
    echo         This build would run roughly 400x slower than it should.
    echo         The build directory is contaminated. Delete it and rerun:
    echo             set FRESH=1
    echo             scripts\build-hip.bat
    exit /b 1
)
echo   optimisation: -O flag present in CMakeCache.txt

if "%CONFIGURE_ONLY%"=="1" (
    echo Configure-only run finished.
    exit /b 0
)

echo.
echo [3/3] Building. GGML_CUDA_FA_ALL_QUANTS is forced ON by CMakeLists.txt
echo       ^(required for --kv-dtype q5_0^), so all 49 fattn-vec instances
echo       compile -- expect a long first build.
cmake --build "%BUILD_DIR%" --config Release -j%JOBS%
if errorlevel 1 (
    echo [error] build failed.
    exit /b 1
)

echo.
echo Done. Binaries in "%BUILD_DIR%\bin"
dir /b "%BUILD_DIR%\bin\*.exe" 2>nul

endlocal
