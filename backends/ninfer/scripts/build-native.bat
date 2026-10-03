@echo off
setlocal
REM Native MSVC + CUDA build against a prebuilt vcpkg triplet tree, without the vcpkg toolchain
REM file: FFmpeg and libcurl are found through CMAKE_PREFIX_PATH and cmake\FindFFMPEG.cmake, which
REM reads VCPKG_ROOT and VCPKG_TARGET_TRIPLET. Every path below can be overridden from the
REM environment.
REM
REM   scripts\build-native.bat configure
REM   scripts\build-native.bat build
REM   scripts\build-native.bat target ninfer-serve [more targets]
if not defined NINFER_VCVARS set "NINFER_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined NINFER_GENERATOR set "NINFER_GENERATOR=Visual Studio 18 2026"
if not defined NINFER_CUDA_PATH set "NINFER_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if not defined NINFER_VCPKG_ROOT set "NINFER_VCPKG_ROOT=C:\vcpkg"
if not defined NINFER_VCPKG_TRIPLET set "NINFER_VCPKG_TRIPLET=x64-windows"
if not defined NINFER_CUDA_ARCH set "NINFER_CUDA_ARCH=120a"
if not defined NINFER_BUILD_DIR set "NINFER_BUILD_DIR=build-windows"

call "%NINFER_VCVARS%" >nul
if errorlevel 1 exit /b 1
REM vcvars64 exports its own VCPKG_ROOT (the tree bundled with Visual Studio); re-assert ours.
set "VCPKG_ROOT=%NINFER_VCPKG_ROOT%"
set "VCPKG_TARGET_TRIPLET=%NINFER_VCPKG_TRIPLET%"
set "CUDA_PATH=%NINFER_CUDA_PATH%"
cd /d "%~dp0.."

if "%1"=="configure" (
  cmake -S . -B "%NINFER_BUILD_DIR%" -G "%NINFER_GENERATOR%" -A x64 ^
    -T "cuda=%CUDA_PATH%" ^
    -DVCPKG_TARGET_TRIPLET=%VCPKG_TARGET_TRIPLET% ^
    -DCMAKE_PREFIX_PATH="%VCPKG_ROOT%\installed\%VCPKG_TARGET_TRIPLET%" ^
    -DCMAKE_CUDA_ARCHITECTURES=%NINFER_CUDA_ARCH% ^
    -DBUILD_TESTING=ON
) else if "%1"=="build" (
  cmake --build "%NINFER_BUILD_DIR%" --config Release --parallel
) else if "%1"=="target" (
  if "%2"=="" (
    echo usage: build-native.bat target ^<target^> [^<target^> ...]
    exit /b 2
  )
  for /f "tokens=1,*" %%a in ("%*") do cmake --build "%NINFER_BUILD_DIR%" --config Release --parallel --target %%b
) else (
  echo usage: build-native.bat configure^|build^|target ^<target^>...
  exit /b 2
)
exit /b %errorlevel%
