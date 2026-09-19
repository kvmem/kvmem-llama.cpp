@echo off
REM ============================================================================
REM  KVMem + llama.cpp  --  ROCm / HIP build  (Windows, AMD RDNA2 e.g. gfx1030)
REM ============================================================================
REM  Mirrors scripts/build-cuda.sh, but drives ggml's HIP backend instead of the
REM  CUDA backend. Everything lands in  <repo>\build-hip  -- this script never
REM  copies into another llama.cpp deployment, so any llama-server already on
REM  PATH is left untouched.
REM
REM  Usage (from an "x64 Native Tools Command Prompt for VS 2022"):
REM      scripts\build-hip.bat
REM
REM  Optional environment overrides:
REM      GPU_TARGETS   default gfx1030
REM      JOBS          default 12  (ninja parallelism)
REM      ROCM          default %ROCM_PATH% or D:\ROCm\10.0.0
REM      BUILD_DIR     default <repo>\build-hip
REM      FRESH         set to 1 to delete the build directory first
REM
REM  NOTE: MSVC 14.44 (VS 2022) is required. 14.51 (VS 2026) turns the two-arg
REM  <cmath> builtins (isgreater, isless, ...) into constexpr, which AMD clang
REM  then rejects in HIP device code. See hip-build-quickref.txt.
REM ============================================================================

setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

if "%GPU_TARGETS%"=="" set "GPU_TARGETS=gfx1030"
if "%JOBS%"=="" set "JOBS=12"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%ROOT%\build-hip"

if "%ROCM%"=="" set "ROCM=%ROCM_PATH%"
if "%ROCM%"=="" set "ROCM=%HIP_PATH%"
if "%ROCM%"=="" (
    echo [error] ROCm root not set. Set ROCM, ROCM_PATH or HIP_PATH first.
    exit /b 1
)

if not exist "%ROCM%\lib\llvm\bin\clang++.exe" (
    echo [error] AMD clang not found under "%ROCM%".
    echo         Set ROCM to your ROCm root before running this script.
    exit /b 1
)

REM --- MSVC environment -------------------------------------------------------
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [error] Could not locate vcvars64.bat for Visual Studio 2022.
    echo         Run this script from an "x64 Native Tools Command Prompt for VS 2022".
    exit /b 1
)
echo [1/3] MSVC environment: "%VCVARS%"
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

REM --- ROCm clang on PATH (cmake resolves it at configure time only) ----------
set "PATH=%ROCM%\lib\llvm\bin;%PATH%"


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

if "%FRESH%"=="1" (
    echo [2/3] Removing existing build directory "%BUILD_DIR%"
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
)

echo.
echo   repo        : %ROOT%
echo   build dir   : %BUILD_DIR%
echo   ROCm        : %ROCM%
echo   GPU_TARGETS : %GPU_TARGETS%
echo   jobs        : %JOBS%
echo.
echo [2/3] Configuring (HIP backend, CUDA backend forced off)...
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++ ^
    -DGGML_HIP=ON ^
    -DGPU_TARGETS=%GPU_TARGETS% ^
    -DGGML_HIP_UMA=OFF ^
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
