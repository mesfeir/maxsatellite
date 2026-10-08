@echo off
REM MaxSatellite API launcher - starts the local HTTP API and keeps the log
REM window open so you can see every request an agent makes.
cd /d "%~dp0"
set PORT=%1
if "%PORT%"=="" set PORT=8791
if exist "target\release\maxsatellite.exe" (
    echo MaxSatellite API on http://127.0.0.1:%PORT%
    echo Close this window to stop it.
    echo.
    "target\release\maxsatellite.exe" serve --port %PORT%
) else if exist "target\debug\maxsatellite.exe" (
    echo MaxSatellite API on http://127.0.0.1:%PORT%  (debug build)
    echo Close this window to stop it.
    echo.
    "target\debug\maxsatellite.exe" serve --port %PORT%
) else (
    echo No build found. Run:  cargo build --release
    pause
)
