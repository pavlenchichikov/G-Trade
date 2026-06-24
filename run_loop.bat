@echo off
REM Daily self-maintaining loop. Register with Windows Task Scheduler (run once):
REM   schtasks /Create /TN "G-Trade Loop" /TR "\"%CD%\run_loop.bat\"" /SC DAILY /ST 23:30
REM Deploy this ONLY after the baseline training has finished.
cd /d "%~dp0"
python loop_cycle.py >> loop.log 2>&1
