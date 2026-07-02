@echo off
REM ===========================================================================
REM  AUTO-RESEARCH AGENT launcher (interactive menu).
REM  Answer the prompts (Enter = default) and the agent starts. It NEVER
REM  touches production: variants train into isolated temp dirs and nothing is
REM  auto-adopted - the agent only flags winners for a human. See README.
REM
REM  Cross-run memory: _ar_tried.json (never re-tests a candidate),
REM  _ar_eval_cache.json (base runs reused while the data is unchanged),
REM  _ar_findings.json (cumulative findings journal). Budget = NEW iterations
REM  per run, so periodic launches keep exploring fresh candidates.
REM
REM  Advanced knobs (screen, prune floor, QD sizes, seed, LLM model/URL) live
REM  below as set lines; the menu only asks the everyday questions.
REM ===========================================================================

cd /d "%~dp0"

REM == Advanced knobs (edit here; the menu does not ask about these) ===========
set "GTRADE_AR_SCREEN=1"
set "GTRADE_AR_SCREEN_MIN=0.0"
set "GTRADE_AR_PRUNE_MIN=8"
set "GTRADE_AR_QD_INIT=8"
set "GTRADE_AR_QD_FINAL=3"
set "GTRADE_AR_SEED=42"
set "AR_PRESCREEN_MIN=0.02"
set "GTRADE_AR_QD_LLM_P=0.3"
REM    Model / base URL overrides (blank = provider default / auto-detect):
set "GTRADE_AR_LLM_MODEL="
set "GTRADE_AR_LLM_BASE_URL="

echo ============================================================
echo   AUTO-RESEARCH  (Enter = default)
echo ============================================================
echo.
echo [1] Mode:
echo     1 = qd (MAP-Elites quality-diversity, the flagship)
echo     2 = features (DSL forward-selection)
echo     3 = labeling,pruning
echo     4 = custom (type your own axes list)
set "MODE=1"
set /p "MODE=    choice [1]: "
if "%MODE%"=="1" set "GTRADE_AR_AXES=qd"
if "%MODE%"=="2" set "GTRADE_AR_AXES=features"
if "%MODE%"=="3" set "GTRADE_AR_AXES=labeling,pruning"
if "%MODE%"=="4" set /p "GTRADE_AR_AXES=    axes (comma-separated): "

echo.
echo [2] Proposer:
echo     1 = evolutionary (no LLM, fully autonomous)
echo     2 = local LLM via Ollama (gemma auto-detected; Ollama must be running)
echo     3 = Anthropic API (needs ANTHROPIC_API_KEY)
set "PROP=1"
set /p "PROP=    choice [1]: "
set "GTRADE_AR_PROPOSER=evolutionary"
set "GTRADE_AR_LLM="
if "%PROP%"=="2" (set "GTRADE_AR_PROPOSER=llm" & set "GTRADE_AR_LLM=ollama")
if "%PROP%"=="3" (set "GTRADE_AR_PROPOSER=llm" & set "GTRADE_AR_LLM=anthropic")

echo.
set "AR_BUDGET=15"
set /p "AR_BUDGET=[3] Budget (NEW search iterations this run) [15]: "

echo.
echo [4] Objective:  1 = mean (average lift)   2 = min (lift the floor)
set "OBJ=1"
set /p "OBJ=    choice [1]: "
set "GTRADE_AR_OBJECTIVE=mean"
if "%OBJ%"=="2" set "GTRADE_AR_OBJECTIVE=min"

echo.
echo ------------------------------------------------------------
echo   axes=%GTRADE_AR_AXES%  proposer=%GTRADE_AR_PROPOSER%  llm=%GTRADE_AR_LLM%
echo   budget=%AR_BUDGET%  objective=%GTRADE_AR_OBJECTIVE%
echo ------------------------------------------------------------
set "GO=Y"
set /p "GO=Start? [Y/n]: "
if /i "%GO%"=="n" exit /b 0

python auto_research.py

echo.
echo Done. Review _auto_research_log.json / _qd_archive.json / _ar_findings.json.
pause
