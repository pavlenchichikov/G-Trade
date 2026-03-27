@echo off
title G-TRADE: GPU ACCELERATED
color 0A
set PYTHONIOENCODING=utf-8
cls
cd /d "%~dp0"

REM Auto-detect Python: conda env > system python
set PY_PATH=python
where conda >nul 2>nul
if %errorlevel% equ 0 (
    call conda activate gtrade_gpu 2>nul
)
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ or run setup_gpu.bat
    pause
    exit
)

:menu
cls
echo =======================================================
echo    G-TRADE V50: GPU ACCELERATED (RTX 2050)
echo =======================================================
echo.
echo  CORE                     ANALYTICS
echo    [1] Full Cycle           [N] News Analyzer
echo    [2] Dashboard            [D] News Digest
echo    [3] Predict (Radar)      [R] Regime Detector
echo                             [C] Correlation Alert
echo                             [WL] Watchlist
echo                             [T] Optuna Tune
echo  DATA / TRAINING           [P] Paper Trading
echo    [4] Data Update
echo    [5] Train Models       REPORTS
echo    [6] Backtest             [M] Model Health
echo                             [E] Export Signals CSV
echo  WHAT-IF SIMULATOR          [L] Signal Log
echo    [W1] Top-5  90d equal    [H] HTML Report
echo    [W2] Top-10 90d equal    [Q] Equity Curve
echo    [W3] Top-5 180d equal  OTHER
echo    [W4] Top-5  90d Kelly    [B] DB Backup
echo    [W5] Custom assets       [I] Install/Repair
echo  SERVICES
echo    [7] Telegram Bot  [8] Scheduler  [9] DB Check  [F] DB Fix  [G] GUI  [0] EXIT
echo.
echo =======================================================
set /p choice="Select: "

if "%choice%"=="1" goto full_run
if "%choice%"=="2" goto dashboard
if "%choice%"=="3" goto predict
if "%choice%"=="4" goto data_only
if "%choice%"=="5" goto train_only
if "%choice%"=="6" goto backtest
if "%choice%"=="7" goto telegram_bot
if "%choice%"=="8" goto scheduler
if "%choice%"=="9" goto db_check
if /i "%choice%"=="N" goto news
if /i "%choice%"=="D" goto digest
if /i "%choice%"=="R" goto regime
if /i "%choice%"=="C" goto corr
if /i "%choice%"=="WL" goto watchlist
if /i "%choice%"=="W1" goto whatif_top5
if /i "%choice%"=="W2" goto whatif_top10
if /i "%choice%"=="W3" goto whatif_180
if /i "%choice%"=="W4" goto whatif_kelly
if /i "%choice%"=="W5" goto whatif_custom
if /i "%choice%"=="P" goto paper
if /i "%choice%"=="M" goto model_health
if /i "%choice%"=="E" goto export
if /i "%choice%"=="L" goto signal_log
if /i "%choice%"=="H" goto report
if /i "%choice%"=="F" goto db_fix
if /i "%choice%"=="B" goto backup
if /i "%choice%"=="I" goto install_fix
if /i "%choice%"=="G" goto gui
if /i "%choice%"=="Q" goto equity
if /i "%choice%"=="T" goto optuna
if "%choice%"=="0" exit
goto menu

:full_run
cls
python data_engine.py
python train_hybrid.py
python -m streamlit run app.py
pause
goto menu

:dashboard
cls
python -m streamlit run app.py
pause
goto menu

:predict
cls
python predict.py
pause
goto menu

:data_only
cls
python data_engine.py
pause
goto menu

:train_only
cls
python train_hybrid.py
pause
goto menu

:backtest
cls
python backtest.py
pause
goto menu

:telegram_bot
cls
echo [INFO] Starting bot... (Do not close this window!)
python alert_bot.py
echo.
echo [WARNING] Bot stopped. Check above for errors.
pause
goto menu

:scheduler
cls
python scheduler.py
pause
goto menu

:db_check
cls
python db_check.py
pause
goto menu

:news
cls
python news_analyzer.py
pause
goto menu

:digest
cls
python news_analyzer.py --digest
pause
goto menu

:regime
cls
python regime_detector.py
pause
goto menu

:corr
cls
python correlation_alert.py
pause
goto menu

:watchlist
cls
python watchlist.py
pause
goto menu

:paper
cls
python paper_trading.py
pause
goto menu

:model_health
cls
python model_health.py
pause
goto menu

:export
cls
python export_signals.py
pause
goto menu

:signal_log
cls
python signal_log.py
pause
goto menu

:report
cls
python performance_report.py
pause
goto menu

:equity
cls
python equity_curve.py
pause
goto menu

:db_fix
cls
python db_check.py --fix
pause
goto menu

:backup
cls
python db_backup.py
pause
goto menu

:install_fix
cls
python -m pip install apimoex requests yfinance pandas "numpy<2" plotly streamlit sqlalchemy catboost scikit-learn pyTelegramBotAPI pysocks python-dotenv tabulate tqdm optuna --no-cache-dir
pause
goto menu

:optuna
cls
python optuna_tune.py
pause
goto menu

:gui
cls
start "" python launcher.py
goto menu

:whatif_top5
cls
echo [What-If] Top-5 активов, 90 дней, равное распределение...
python whatif_simulator.py --top 5 --days 90 --strategy equal
pause
goto menu

:whatif_top10
cls
echo [What-If] Top-10 активов, 90 дней, равное распределение...
python whatif_simulator.py --top 10 --days 90 --strategy equal
pause
goto menu

:whatif_180
cls
echo [What-If] Top-5 активов, 180 дней, равное распределение...
python whatif_simulator.py --top 5 --days 180 --strategy equal
pause
goto menu

:whatif_kelly
cls
echo [What-If] Top-5 активов, 90 дней, Kelly-аллокация...
python whatif_simulator.py --top 5 --days 90 --strategy kelly
pause
goto menu

:whatif_custom
cls
set /p WI_ASSETS="Активы через пробел (BTC ETH NVDA ...): "
set /p WI_DAYS="Количество дней (Enter = 90): "
if "%WI_DAYS%"=="" set WI_DAYS=90
set /p WI_CAP="Капитал USD (Enter = 10000): "
if "%WI_CAP%"=="" set WI_CAP=10000
echo.
echo [What-If] Активы: %WI_ASSETS% | Дней: %WI_DAYS% | Капитал: $%WI_CAP%
python whatif_simulator.py %WI_ASSETS% --days %WI_DAYS% --capital %WI_CAP%
pause
goto menu
