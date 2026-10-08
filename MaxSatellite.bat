@echo off
REM MaxSatellite launcher - starts the GUI with no console window.
cd /d "%~dp0"
if exist "target\release\maxsatellite.exe" (
    start "" "target\release\maxsatellite.exe" gui
) else if exist "target\debug\maxsatellite.exe" (
    start "" "target\debug\maxsatellite.exe" gui
) else (
    echo No build found. Run:  cargo build --release
    pause
)
