@echo off
setlocal
REM Control-arm build driver for the serve A/B rig on Windows. Mirrors build-native.bat
REM (vcvars64 + vcpkg FFmpeg env) but targets the control worktree/build and builds only the
REM ninfer-serve app (tests/benchmarks off).
REM control\src must be a checkout/worktree of the control branch; control\build is
REM created by configure. Paths are relative to this script so the rig stays
REM self-contained in bench/ab. Adjust the VS/vcpkg/CUDA paths for your machine.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "SRC=%~dp0control\src"
set "BLD=%~dp0control\build"
if "%~1"=="configure" (
  cmake -S "%SRC%" -B "%BLD%" -G "Visual Studio 18 2026" -A x64 -T "cuda=%CUDA_PATH%" -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_PREFIX_PATH="C:\vcpkg\installed\x64-windows" -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
) else (
  cmake --build "%BLD%" --config Release --target ninfer-serve --parallel
)
exit /b %errorlevel%
