@echo off
REM Toolchain pinning this host needs (see docs/rtx-3090-windows.md):
REM  - MSVC 14.44 from VS 2022 BuildTools. VS 2026's 14.50 is _MSC_VER 1950 and CUDA 12.8's
REM    host_config.h accepts 1910-1949 only, so vcvars64.bat must win the PATH race.
REM  - CUDA 12.8 forced via CUDACXX; 12.4 is also installed and would be found first.
REM  - -Xcompiler=/Zc:__cplusplus, without which MSVC reports __cplusplus == 199711L under
REM    /std:c++20 and hides CUDA's constexpr dim3 constructors.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/Zc:__cplusplus" > configure.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
REM -k 0: keep going after a failure so one pass reports every broken target instead of stopping
REM at the first. Ninja otherwise exits after the first error, which costs a full cycle per fix.
cmake --build build-ninja --parallel -- -k 0 > build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
