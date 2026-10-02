@echo off
chcp 65001 >nul
cls

:MENU
echo.
echo  +==========================================================+
echo  ^|             ShakerMaker - Windows Setup Menu            ^|
echo  +==========================================================+
echo  ^|                                                          ^|
echo  ^|   [1]  Step 1 - Install Prerequisites                   ^|
echo  ^|         (Python, Git, VS2022, Intel oneAPI, venv, deps) ^|
echo  ^|                                                          ^|
echo  ^|   [2]  Step 2 - Create Junction                         ^|
echo  ^|         (link your repo to a space-free build path)     ^|
echo  ^|                                                          ^|
echo  ^|   [3]  Step 3 - Build and Compile                       ^|
echo  ^|         (compile ShakerMaker + smoke test)              ^|
echo  ^|                                                          ^|
echo  ^|   [4]  Run All Steps in Order (1 then 2 then 3)         ^|
echo  ^|                                                          ^|
echo  ^|   [Q]  Quit                                             ^|
echo  ^|                                                          ^|
echo  +==========================================================+
echo.

set /p CHOICE="  Enter your choice: "

if /i "%CHOICE%"=="1" goto STEP1
if /i "%CHOICE%"=="2" goto STEP2
if /i "%CHOICE%"=="3" goto STEP3
if /i "%CHOICE%"=="4" goto RUNALL
if /i "%CHOICE%"=="Q" goto QUIT
if /i "%CHOICE%"=="q" goto QUIT

echo  Invalid choice. Please enter 1, 2, 3, 4 or Q.
goto MENU

:STEP1
echo.
echo  Launching Step 1 - Prerequisites Setup...
echo.
PowerShell -ExecutionPolicy Bypass -File "%~dp001_shakermaker_setup.ps1"
goto MENU

:STEP2
echo.
echo  Launching Step 2 - Junction Setup...
echo.
PowerShell -ExecutionPolicy Bypass -File "%~dp002_shakermaker_junction.ps1"
goto MENU

:STEP3
echo.
echo  Launching Step 3 - Build and Compile...
echo.
PowerShell -ExecutionPolicy Bypass -File "%~dp003_shakermaker_build.ps1"
goto MENU

:RUNALL
echo.
echo  Running all steps in order...
echo.
echo  --- Step 1 ---
PowerShell -ExecutionPolicy Bypass -File "%~dp001_shakermaker_setup.ps1" -NonInteractive
if errorlevel 1 ( echo  [!!] Step 1 failed. Stopping. & pause & goto MENU )
echo  --- Step 2 ---
PowerShell -ExecutionPolicy Bypass -File "%~dp002_shakermaker_junction.ps1" -NonInteractive
if errorlevel 1 ( echo  [!!] Step 2 failed. Stopping. & pause & goto MENU )
echo  --- Step 3 ---
PowerShell -ExecutionPolicy Bypass -File "%~dp003_shakermaker_build.ps1" -NonInteractive
if errorlevel 1 ( echo  [!!] Step 3 failed. Stopping. & pause & goto MENU )
echo.
echo  All steps complete.
pause
goto MENU

:QUIT
echo.
echo  Goodbye Ladruno!
echo.
exit /b 0
