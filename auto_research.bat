@echo off
REM ===========================================================================
REM  AUTO-RESEARCH AGENT launcher (central control panel).
REM  Edit the knobs below, then run this file. It NEVER touches production:
REM  variants train into isolated temp dirs and nothing is auto-adopted - the
REM  agent only flags winners for a human. See README "Auto-research agent".
REM
REM  Default is fully autonomous (no LLM, no API key). The cheap CatBoost-only
REM  screen makes most candidates fast; only survivors / finalists pay a full
REM  ensemble train, so a run is hours, not days.
REM ===========================================================================

cd /d "%~dp0"

REM == 1. MODE: which search axes to run (comma-separated) =====================
REM    qd        : MAP-Elites quality-diversity agent (the flagship - illuminates
REM                a diverse archive of feature/label experiments). Runs ALONE.
REM    features  : evolve engineered features (DSL forward-selection)
REM    labeling  : sweep the rel_median label window {20,30,60}
REM    pruning   : backward-elimination over the active features
REM    Examples: "qd"   or   "labeling,pruning"   or   "features,labeling"
set "GTRADE_AR_AXES=qd"

REM == 2. Budget + cheap screen ===============================================
REM    AR_BUDGET        : search iterations (qd: illuminate steps; axes: rounds).
REM    GTRADE_AR_SCREEN : 1 = CatBoost-only screen before the full eval (fast).
REM    GTRADE_AR_SCREEN_MIN : a candidate passes the screen above this delta.
set "AR_BUDGET=15"
set "GTRADE_AR_SCREEN=1"
set "GTRADE_AR_SCREEN_MIN=0.0"

REM == 3. Objective + pruning floor ===========================================
REM    GTRADE_AR_OBJECTIVE : mean (default) or min (lift-the-floor / weak cluster).
REM    GTRADE_AR_PRUNE_MIN : pruning never drops below this many active features.
set "GTRADE_AR_OBJECTIVE=mean"
set "GTRADE_AR_PRUNE_MIN=8"

REM == 4. QD (MAP-Elites) knobs - only used when GTRADE_AR_AXES has qd =========
REM    GTRADE_AR_QD_INIT  : random seed genomes to start the archive.
REM    GTRADE_AR_QD_FINAL : top diverse elites that get the full held-out gate.
set "GTRADE_AR_QD_INIT=8"
set "GTRADE_AR_QD_FINAL=3"

REM == 5. Proposer (for the features axis) ====================================
REM    GTRADE_AR_PROPOSER : evolutionary (no LLM) or llm.
REM    GTRADE_AR_SEED     : integer for reproducible search; clear it for random.
REM    AR_PRESCREEN_MIN   : min univariate corr to bother A/B-testing a feature.
set "GTRADE_AR_PROPOSER=evolutionary"
set "GTRADE_AR_SEED=42"
set "AR_PRESCREEN_MIN=0.02"

REM == 6. LLM settings (ONLY when GTRADE_AR_PROPOSER=llm) ======================
REM    GTRADE_AR_LLM         : anthropic (needs anthropic SDK + ANTHROPIC_API_KEY)
REM                            or openai (or any OpenAI-compatible base URL).
REM    GTRADE_AR_LLM_MODEL   : blank = provider default.
REM    GTRADE_AR_LLM_BASE_URL: blank = provider default.
set "GTRADE_AR_LLM=anthropic"
set "GTRADE_AR_LLM_MODEL="
set "GTRADE_AR_LLM_BASE_URL="

echo ============================================================
echo   AUTO-RESEARCH  axes=%GTRADE_AR_AXES%  budget=%AR_BUDGET%  objective=%GTRADE_AR_OBJECTIVE%  screen=%GTRADE_AR_SCREEN%
echo ============================================================
python auto_research.py

echo.
echo Done. Review _auto_research_log.json (axes) or _qd_archive.json (qd) for the verdict.
pause
