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
REM  Chronos forecast features (menu item 5): zero-shot price-forecast columns
REM  (chronos_dir / chronos_ret / chronos_spread) from Amazon's Chronos model,
REM  added to every asset as extra inputs. They are OFF by default and require
REM  a one-time setup:
REM    1) pip install -r requirements-chronos.txt   (torch + chronos-forecasting)
REM    2) python precompute_chronos.py              (fills the forecast cache)
REM  Without that cache the columns are empty and the toggle is a no-op.
REM
REM  Research wiki (menu item 6): a compounding, self-maintained knowledge base
REM  (GTRADE_AR_WIKI). After each run an LLM distills the findings journal into
REM  _ar_wiki/*.md topic pages the proposer then reads, so learning accumulates
REM  across runs instead of a sliding window. It uses the LLM backend, so pick an
REM  LLM proposer (or it defaults to Anthropic and needs ANTHROPIC_API_KEY). Off
REM  by default, so it is byte-identical. When on, this script also offers a wiki lint
REM  (reconcile contradictions + prune stale claims) after the run.
REM
REM  RL scheduler (menu item 7): a learned budget allocator over the QD child
REM  sources (GTRADE_AR_RL). A discounted Thompson bandit learns which of the
REM  nine emitters (feature/hyper/nets/tuning mutations, crossover, LLM,
REM  surrogate, CMA over the numeric genes, novelty targeting of empty niches)
REM  actually produce archive improvements, plus curiosity-based parent
REM  selection. The statistical adoption gate is untouched - the scheduler only
REM  decides what to TRY. Off by default (byte-identical uniform search); it
REM  self-disables within a run if it underperforms the uniform baseline.
REM  State: rl_scheduler_v1.json (posteriors are printed at run start/end).
REM
REM  Advanced knobs (screen, prune floor, QD sizes, seed, base URL, exhaustion
REM  cutoff) live below as set lines; the menu only asks the everyday questions.
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
set "GTRADE_AR_QD_MAX_MISSES=5"
REM    Base URL override (blank = provider default / Ollama localhost):
set "GTRADE_AR_LLM_BASE_URL="
REM    Model override (blank = auto-detect for Ollama / provider default);
REM    the menu sets this for you when you pick a local or OpenAI model.
set "GTRADE_AR_LLM_MODEL="

echo ============================================================
echo   AUTO-RESEARCH  (Enter = default)
echo ============================================================
echo.
echo [0] Action:  1 = search for new candidates (default)
echo              2 = re-gate stored candidates under the current gate (reuse past runs)
set "ACT=1"
set /p "ACT=    choice [1]: "
if "%ACT%"=="2" goto :regate

echo.
echo [1] Mode (type the number, or an axes name/list directly):
echo     1 = qd (MAP-Elites quality-diversity, the flagship; genome now also
echo         carries hyperparameter, net-hygiene and triple-barrier genes)
echo     2 = features (DSL forward-selection)
echo     3 = labeling,pruning (rel_median windows + triple_barrier horizons; drops)
echo     4 = hyper,nets,thresholds,regime (model + tuning levers)
echo     5 = custom (type your own axes list)
set "MODE=1"
set /p "MODE=    choice [1]: "
set "GTRADE_AR_AXES="
if "%MODE%"=="1" set "GTRADE_AR_AXES=qd"
if "%MODE%"=="2" set "GTRADE_AR_AXES=features"
if "%MODE%"=="3" set "GTRADE_AR_AXES=labeling,pruning"
if "%MODE%"=="4" set "GTRADE_AR_AXES=hyper,nets,thresholds,regime"
if "%MODE%"=="5" set /p "GTRADE_AR_AXES=    axes (comma-separated): "
REM  Not one of 1-5: use whatever was typed verbatim as the axes (e.g. "qd" or "qd,features").
if not defined GTRADE_AR_AXES set "GTRADE_AR_AXES=%MODE%"

echo.
echo [2] Proposer:
echo     1 = evolutionary (no LLM, fully autonomous)
echo     2 = local LLM (Ollama; any installed model - you pick below)
echo     3 = Anthropic API (needs ANTHROPIC_API_KEY)
echo     4 = OpenAI API (needs OPENAI_API_KEY)
set "PROP=1"
set /p "PROP=    choice [1]: "
set "GTRADE_AR_PROPOSER=evolutionary"
set "GTRADE_AR_LLM="

if "%PROP%"=="2" (
  set "GTRADE_AR_PROPOSER=llm"
  set "GTRADE_AR_LLM=ollama"
  echo.
  echo     Installed local models:
  python -m core.llm_proposer --list-ollama
  echo     Enter = auto-detect ^(first gemma, else first installed^).
  set /p "GTRADE_AR_LLM_MODEL=    model name [auto]: "
)

if "%PROP%"=="3" (
  set "GTRADE_AR_PROPOSER=llm"
  set "GTRADE_AR_LLM=anthropic"
  set /p "GTRADE_AR_LLM_MODEL=    Anthropic model [claude-opus-4-8]: "
)

if "%PROP%"=="4" (
  set "GTRADE_AR_PROPOSER=llm"
  set "GTRADE_AR_LLM=openai"
  set /p "GTRADE_AR_LLM_MODEL=    OpenAI model [gpt-4o]: "
)

REM  Token budget for the LLM (proposer + wiki). Reasoning models like gemma spend
REM  tokens on an internal trace before the answer; too small a cap returns EMPTY
REM  content. 0 = no cap (the local model is free; the only cost is wall-clock time).
if not "%GTRADE_AR_PROPOSER%"=="llm" goto :skiptoks
echo.
echo     LLM max tokens  (0 = no cap; gemma reasoning needs room)
set "GTRADE_AR_LLM_MAX_TOKENS=8000"
set /p "GTRADE_AR_LLM_MAX_TOKENS=    max tokens [8000]: "
:skiptoks

echo.
set "AR_BUDGET=15"
set /p "AR_BUDGET=[3] Budget (NEW search iterations this run) [15]: "

echo.
echo [4] Objective (how per-asset held-out lifts are reduced to one number):
echo     1 = mean (average)   2 = min (lift the floor)   3 = median (robust average)
echo     4 = cvar (mean of the worst 25%%)   5 = sharpe (consistency)   6 = trimmed (no extremes)
set "OBJ=1"
set /p "OBJ=    choice [1]: "
set "GTRADE_AR_OBJECTIVE=mean"
if "%OBJ%"=="2" set "GTRADE_AR_OBJECTIVE=min"
if "%OBJ%"=="3" set "GTRADE_AR_OBJECTIVE=median"
if "%OBJ%"=="4" set "GTRADE_AR_OBJECTIVE=cvar"
if "%OBJ%"=="5" set "GTRADE_AR_OBJECTIVE=sharpe"
if "%OBJ%"=="6" set "GTRADE_AR_OBJECTIVE=trimmed_mean"

echo.
echo [5] Chronos forecast features?  (needs setup - see top of this file)
echo     1 = off (default)   2 = on
set "CHR=1"
set /p "CHR=    choice [1]: "
set "GTRADE_CHRONOS="
if "%CHR%"=="2" (
  set "GTRADE_CHRONOS=1"
  set "GTRADE_EXTRA_FEATURES=chronos_dir,chronos_ret,chronos_spread"
  echo     Chronos ON - make sure you ran: python precompute_chronos.py
)

echo.
echo [6] Research wiki?  (compounding findings; uses the LLM backend)
echo     1 = off (default)   2 = on
set "WIKI=1"
set /p "WIKI=    choice [1]: "
set "GTRADE_AR_WIKI="
if "%WIKI%"=="2" set "GTRADE_AR_WIKI=1"

echo.
echo [7] RL scheduler?  (learned budget allocation over the QD child sources;
echo     Thompson bandit + CMA/novelty emitters; the adoption gate is untouched)
echo     1 = off (default)   2 = on
set "RL=1"
set /p "RL=    choice [1]: "
set "GTRADE_AR_RL="
if "%RL%"=="2" set "GTRADE_AR_RL=1"

echo.
echo ------------------------------------------------------------
echo   axes=%GTRADE_AR_AXES%  proposer=%GTRADE_AR_PROPOSER%  llm=%GTRADE_AR_LLM%
echo   model=%GTRADE_AR_LLM_MODEL%  maxtok=%GTRADE_AR_LLM_MAX_TOKENS%  chronos=%GTRADE_CHRONOS%  wiki=%GTRADE_AR_WIKI%
echo   budget=%AR_BUDGET%  objective=%GTRADE_AR_OBJECTIVE%  rl=%GTRADE_AR_RL%
echo ------------------------------------------------------------
set "GO=Y"
set /p "GO=Start? [Y/n]: "
if /i "%GO%"=="n" exit /b 0

python auto_research.py

REM  Optional wiki lint (flat, no parenthesized block, so set /p + if see the fresh value).
if not "%GTRADE_AR_WIKI%"=="1" goto :nolint
echo.
set "LINT=n"
set /p "LINT=Lint the research wiki now (reconcile + prune)? [y/N]: "
if /i "%LINT%"=="y" python -c "from core import ar_wiki; ar_wiki.lint_wiki()"
:nolint

echo.
echo Done. Review _auto_research_log.json / _qd_archive.json / _ar_findings.json.
if "%GTRADE_AR_WIKI%"=="1" echo Research wiki: _ar_wiki\*.md  (also on the /research Web UI page).
if "%GTRADE_AR_RL%"=="1" echo RL scheduler state: rl_scheduler_v1.json  (arm posteriors are in the run log above).
pause
goto :end

REM == Re-gate: re-score the best already-found candidates under the current gate ==
:regate
echo.
echo   RE-GATE: re-scores the best already-found candidate genomes (from _qd_archive +
echo   _ar_findings) under the current stronger gate. Reuses past experiments; trains only
echo   the top-K on the held-out set. Adopts nothing - flags winners for you.
set "RGK=8"
set /p "RGK=    top-K candidates [8]: "
echo.
echo     CB-only pre-screen first (cheaper, coarser)?  1 = no (default)   2 = yes
set "RGS=1"
set /p "RGS=    choice [1]: "
set "RGSCREEN="
if "%RGS%"=="2" set "RGSCREEN=--regate-screen"
echo.
echo   Running: auto_research.py --regate --regate-k %RGK% %RGSCREEN%
python auto_research.py --regate --regate-k %RGK% %RGSCREEN%
echo.
echo Done. Review _ar_findings.json (mode=regate) for the new verdicts.
pause

:end
