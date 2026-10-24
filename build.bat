@echo off
rem Build moss_tts.exe with MSVC.
rem
rem Mirrors claunch's build.bat: locate the Visual Studio dev environment, then
rem compile ONE translation unit into a self-contained console exe.
rem
rem Dependency notes:
rem   /MT            -> the C runtime is linked statically, so the exe imports no
rem                     vcruntime140.dll / ucrtbase.dll.  The only DLL that has
rem                     to ship next to it is onnxruntime.dll.
rem   no import lib   -> onnxruntime is resolved at runtime with LoadLibraryA +
rem                     GetProcAddress("OrtGetApiBase"), so onnxruntime.lib is
rem                     NOT required (the release only ships the DLL anyway).
rem   /utf-8          -> the source has Chinese comments; without this MSVC reads
rem                     it as GBK and mangles the string literals.
rem (Keep comments ASCII: cmd.exe line parsing chokes on non-ASCII bytes in .bat)

setlocal
set "VSD=D:/Program Files\Common7\Tools\vsdevcmd.bat"
if not exist "%VSD%" set "VSD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\vsdevcmd.bat"
rem last resort: whatever vswhere reports as the newest toolset install (CI
rem images ship a different edition than the hardcoded paths above)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSD%" if exist "%VSWHERE%" for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSD=%%i\Common7\Tools\vsdevcmd.bat"
if not exist "%VSD%" (
    echo build.bat: vsdevcmd.bat not found; edit VSD in this script.
    exit /b 1
)
call "%VSD%" -arch=x64 -no_logo
if errorlevel 1 exit /b 1

rem /O2 /GL + /LTCG: whole-program optimisation; the tokenizer's hash lookups and
rem the BPE merge loop are the only real CPU cost on our side, everything else is
rem inside onnxruntime.
cl /nologo /std:c11 /W3 /O2 /GL /MT /utf-8 /D_CRT_SECURE_NO_WARNINGS ^
    moss_tts.c ^
    /link /LTCG /OPT:REF /OPT:ICF /OUT:moss_tts.exe ^
    kernel32.lib
if errorlevel 1 exit /b 1
del /q moss_tts.obj 2>nul

echo.
echo built: moss_tts.exe   (needs onnxruntime.dll + models\ next to it)
echo run  : moss_tts.exe "你好，世界" --out hello.wav
