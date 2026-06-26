@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
call C:\Users\ppala\shakermaker_venv\Scripts\activate.bat
mpiexec -n 10 python %1
pause

# run_shakermaker.bat main_shakermaker.py