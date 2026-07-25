@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ============================================================
REM  TDSE one-click runner (launcher.py: merge -> run -> plot)
REM ============================================================

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "LAUNCHER=C:/fortran/三维TDSE/v5.1-release/tools/launcher.py"

echo ============================================================
echo   TDSE Project Runner
echo   Project : %PROJECT_DIR%
echo ============================================================
echo.

python "%LAUNCHER%" run "%PROJECT_DIR%"
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Run failed with exit code %ERRORLEVEL%
)
pause
