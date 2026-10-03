@echo off
setlocal
REM Control-arm build driver for the agentic A/B suite on Windows.
REM
REM The control checkout is the build this line is compared with, for example the previous
REM release or upstream at the commit this line has merged. Example:
REM   git worktree add -b ab/control <dir>\src <commit>
REM
REM Usage: build_control.bat configure|build
REM   AB_CONTROL_SRC    control source checkout   (default: .\control\src)
REM   AB_CONTROL_BUILD  control build directory   (default: .\control\build)
REM Builds only ninfer-serve (apps on, tests/benchmarks off). Adjust the VS, vcpkg and
REM CUDA paths below for your machine.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if "%AB_CONTROL_SRC%"=="" set "AB_CONTROL_SRC=%~dp0control\src"
if "%AB_CONTROL_BUILD%"=="" set "AB_CONTROL_BUILD=%~dp0control\build"
if "%~1"=="configure" (
  cmake -S "%AB_CONTROL_SRC%" -B "%AB_CONTROL_BUILD%" -G "Visual Studio 18 2026" -A x64 -T "cuda=%CUDA_PATH%" -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_PREFIX_PATH="C:\vcpkg\installed\x64-windows" -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
) else if "%~1"=="build" (
  cmake --build "%AB_CONTROL_BUILD%" --config Release --target ninfer-serve --parallel
) else (
  echo usage: build_control.bat configure^|build
  exit /b 2
)
exit /b %errorlevel%
