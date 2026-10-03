@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe
cmake --build build-ninja > build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL% >> build.log
