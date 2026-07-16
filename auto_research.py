"""Auto-research loop (local). The agent proposes feature variants
from the DSL, the harness A/B-tests them base vs variant on a selection subset,
and winners are checked on a held-out subset and flagged for a human. It never
retrains production.

Proposer is autonomous (evolutionary search, no LLM) by default. Set
GTRADE_AR_PROPOSER=llm to use the LLM proposer instead; the LLM layer lives in
core/llm_proposer.py and supports anthropic (default) and openai providers
(ollama arrives in the next task). GTRADE_AR_SEED makes the evolutionary search
reproducible.

Run:  python auto_research.py
      GTRADE_AR_PROPOSER=llm python auto_research.py
"""
import bisect
import copy
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime

from core import ar_memory
from core import ar_wiki
from core import llm_proposer
from core import qd_surrogate
from core.feature_dsl import validate_spec
from core.logger import get_logger

logger = get_logger("auto_research")

BASE = os.path.dirname(os.path.abspath(__file__))
SELECTION_ASSETS = "SP500,NVDA,BTC,ETH,EURUSD,GBPJPY,GAS,AAPL,SBER,DAX"
HELDOUT_ASSETS = "MSFT,GOLD,USDJPY,ADA,CAC40,XOM,GOOGL,SOL,SILVER,GBPUSD,NASDAQ,DXY,TNX,AIRBUS"
BUDGET = int(os.getenv("AR_BUDGET", "15"))
ADOPT_MEAN_SCORE_DELTA = 0.5

LIGHT_ENV = {
    "GTRADE_WORKERS": "4", "GTRADE_NEURAL_SLOTS": "4", "GTRADE_TF_THREADS": "2",
    "GTRADE_CB_THREADS": "1", "GTRADE_MAX_FOLDS": "5", "GTRADE_ADAPTIVE_NETS": "1",
    "GTRADE_NET_CAP": "80", "GTRADE_EPOCHS_LSTM": "90", "GTRADE_EPOCHS_TF": "60",
    "GTRADE_EPOCHS_TCN": "50", "GTRADE_FORCE_PROMOTE": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
}


def _reduce_deltas(deltas, objective):
    """Reduce per-asset Score deltas to one objective value. 'mean' (average lift) and
    'min' (lift-the-floor) are the originals; the diversifiers are 'median' (robust
    average), 'cvar' (mean of the worst quartile - a softer, less-noisy floor than min),
    'trimmed_mean' (average without the single best/worst), and 'sharpe' (mean/std =
    consistency; a DIFFERENT, dimensionless scale gated by GTRADE_AR_ADOPT_SHARPE, not
    the Score-delta floor). Unknown - mean."""
    n = len(deltas)
    if n == 0:
        return 0.0
    if objective == "min":
        return min(deltas)
    if objective == "median":
        return statistics.median(deltas)
    if objective == "cvar":
        k = max(1, math.ceil(n * 0.25))
        return sum(sorted(deltas)[:k]) / k
    if objective == "trimmed_mean":
        if n < 4:
            return sum(deltas) / n
        core = sorted(deltas)[1:-1]
        return sum(core) / len(core)
    if objective == "sharpe":
        sd = statistics.pstdev(deltas) if n > 1 else 0.0
        return statistics.mean(deltas) / (sd if sd > 1e-9 else 1e-9)
    return sum(deltas) / n


def _objective_delta(var_rows, base_score, objective="mean"):
    """Paired (variant minus base) Score deltas over shared assets, reduced by the
    objective (see _reduce_deltas). Returns (value, deltas)."""
    e = {r["Asset"]: r.get("Score", 0.0) for r in var_rows}
    common = sorted(set(e) & set(base_score))
    if not common:
        return 0.0, []
    deltas = [e[a] - base_score[a] for a in common]
    return _reduce_deltas(deltas, objective), deltas


def _mean_delta(var_rows, base_score):
    """Mean paired Score delta (backward-compatible wrapper over _objective_delta)."""
    return _objective_delta(var_rows, base_score, "mean")


def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up FDR. Returns a bool per input p-value (significant)
    in the ORIGINAL order. Empty input returns []."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    thresh_rank = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= (rank / m) * alpha:
            thresh_rank = rank
    if thresh_rank == 0:
        return [False] * m
    cutoff = pvals[order[thresh_rank - 1]]
    return [pvals[i] <= cutoff for i in range(m)]


def _sign_test_p(deltas):
    """One-sided sign-test p-value: P(X >= k) under a fair coin, k = assets improved.
    Small p means the improvement is unlikely to be chance across the held-out set."""
    n = len(deltas)
    if n == 0:
        return 1.0
    k = sum(1 for d in deltas if d > 0)
    return sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)


def _wilcoxon_p(deltas):
    """One-sided Wilcoxon signed-rank p that the per-asset deltas are > 0 - a
    magnitude-aware improvement test (a consistent small edge across most assets
    passes even if a few are slightly negative), far more powerful than the sign
    test at this n. Returns 1.0 (no evidence) on inputs scipy cannot test: fewer
    than 2 non-zero deltas, or any scipy error."""
    nz = [d for d in deltas if d != 0]
    if len(nz) < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(deltas, alternative="greater").pvalue)
    except Exception:
        return 1.0


_OBJECTIVES = ("mean", "min", "median", "cvar", "sharpe", "trimmed_mean")


def _objective():
    """The search/gate objective (see _reduce_deltas): mean (default) / min / median /
    cvar / sharpe / trimmed_mean."""
    o = (os.getenv("GTRADE_AR_OBJECTIVE") or "mean").strip().lower()
    if o not in _OBJECTIVES:
        logger.warning("unknown GTRADE_AR_OBJECTIVE %r, using mean", o)
        return "mean"
    return o


def _adopt_floor(objective="mean"):
    """The practical-effect floor the objective value must beat to adopt. Score-scale
    objectives (mean/min/median/cvar/trimmed_mean) use ADOPT_MEAN_SCORE_DELTA; the
    dimensionless 'sharpe' uses GTRADE_AR_ADOPT_SHARPE (default 0.5)."""
    if objective == "sharpe":
        try:
            return float(os.getenv("GTRADE_AR_ADOPT_SHARPE") or "0.5")
        except ValueError:
            return 0.5
    return ADOPT_MEAN_SCORE_DELTA


def holdout_stats(base_rows, ext_rows, objective="mean"):
    """Raw held-out stats for a variant: (wilcoxon p, objective value, deltas, tag).
    No adoption decision - main applies BH across the axis-winners."""
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in base_rows}
    value, deltas = _objective_delta(ext_rows, base_score, objective)
    if not deltas:
        return 1.0, 0.0, [], "no common held-out assets"
    p = _wilcoxon_p(deltas)
    up = sum(1 for d in deltas if d > 0)
    tag = "%s dScore %.2f, wilcoxon p=%.3f (%d/%d up)" % (objective, value, p, up, len(deltas))
    return p, value, deltas, tag


def is_adoptable(base_rows, ext_rows, n_experiments, budget, alpha=0.05, objective="mean"):
    """Single-test adoption (kept for ab_labeling and single-axis use): significant
    (sign-test p < alpha) AND practically meaningful (objective value over the
    threshold), within budget."""
    if n_experiments > budget:
        return False, "over iteration budget (%d > %d)" % (n_experiments, budget)
    p, value, deltas, tag = holdout_stats(base_rows, ext_rows, objective)
    if not deltas:
        return False, "no common held-out assets"
    if p < alpha and value > _adopt_floor(objective):
        return True, tag
    return False, tag + " (below bar)"


def _train(subset, env_overrides, model_dir):
    env = dict(os.environ)
    env.update(LIGHT_ENV)
    env["GTRADE_ASSETS"] = subset
    env["GTRADE_MODEL_DIR"] = model_dir
    env.update(env_overrides)
    subprocess.run([sys.executable, "train_hybrid.py"], cwd=BASE, env=env)
    path = os.path.join(model_dir, "quality_report.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def train_env(subset, env_overrides):
    """Train the subset once with the given training env overrides; returns the
    quality_report rows ([] on failure). The generic primitive every axis uses.

    The temp model dir is removed after the rows are read back, so long runs
    (e.g. a large re-gate) do not leak thousands of ar_* dirs into %TEMP%."""
    tmp = tempfile.mkdtemp(prefix="ar_")
    try:
        return _train(subset, dict(env_overrides), os.path.join(tmp, "run"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def train_base_cached(subset, env):
    """Cache-first BASE training: identical subset + env + feature space +
    data snapshot reuses the stored quality rows instead of retraining.
    Candidate runs never go through here (their envs embed temp paths)."""
    key = ar_memory.base_key(subset, env)
    rows = ar_memory.cache_get(key)
    if rows is not None:
        print("[auto-research] base cache hit: %s %s" % (subset, env or "{}"))
        return rows
    rows = train_env(subset, env)
    if rows:
        ar_memory.cache_put(key, rows)
    return rows


def neural_contribution(full_rows, cbonly_rows):
    """Per-asset neural-member contribution = full ensemble Score minus the CB-only
    Score (nets replaced by neutral 0.5 under GTRADE_SCREEN_ONLY), over the shared
    assets. Assets missing from either side are skipped."""
    cb = {r["Asset"]: r.get("Score", 0.0) for r in cbonly_rows}
    out = {}
    for r in full_rows:
        a = r["Asset"]
        if a in cb:
            out[a] = r.get("Score", 0.0) - cb[a]
    return out


def contribution_rows(subset, env, full_fn):
    """Neural-contribution rows for a config: a full train (via full_fn, e.g.
    train_base_cached for bases or train_env for candidates) minus a CB-only train
    (also via full_fn with GTRADE_SCREEN_ONLY added). Empty if either train yields no
    rows. Using full_fn for the CB train means a cached base (full_fn=train_base_cached)
    gets a cached CB train too, and an injected fake trainer intercepts the CB train in
    tests. For candidates and winners full_fn is train_env, so their CB train is
    unchanged. This is a consistency change only; production result numbers are the same."""
    full = full_fn(subset, env)
    cb = full_fn(subset, screen_env(env))
    return [{"Asset": a, "Score": c}
            for a, c in neural_contribution(full, cb).items()]


def _heldout_eval(subset, env, full_fn):
    """(full_rows, contribution_rows) for one config, sharing the single full train
    so the metric never pays for a redundant full train. The CB train uses full_fn
    (with GTRADE_SCREEN_ONLY added): a cached base gets a cached CB train, and an
    injected fake trainer intercepts the CB train in tests. For candidates and winners
    full_fn is train_env, so their CB train is unchanged."""
    full = full_fn(subset, env)
    cb = full_fn(subset, screen_env(env))
    contrib = [{"Asset": a, "Score": c}
               for a, c in neural_contribution(full, cb).items()]
    return full, contrib


def _score_basis():
    """What the search scores: 'raw' (default) or 'neural' (neural contribution)."""
    b = (os.getenv("GTRADE_AR_SCORE_BASIS") or "raw").strip().lower()
    if b not in ("raw", "neural"):
        logger.warning("unknown GTRADE_AR_SCORE_BASIS %r, using raw", b)
        return "raw"
    return b


def score_rows(subset, env, full_fn):
    """Scoring rows for a config under the active basis. raw - full_fn(subset, env)
    (current behavior, caching preserved). neural - neural-contribution rows."""
    if _score_basis() == "neural":
        return contribution_rows(subset, env, full_fn)
    return full_fn(subset, env)


def _feature_env(specs, extra_names):
    """Feature-axis env overrides: materialize DSL specs to a temp file and point
    train_hybrid at them. Empty specs means no overrides (the plain base set)."""
    if not specs:
        return {}
    tmp = tempfile.mkdtemp(prefix="ar_specs_")
    spath = os.path.join(tmp, "specs.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(specs, f)
    return {"GTRADE_DSL_SPECS": spath, "GTRADE_EXTRA_FEATURES": ",".join(extra_names)}


def train_rows(subset, specs, extra_names):
    """Back-compat wrapper: train the subset with DSL feature specs as extra features."""
    return train_env(subset, _feature_env(specs, extra_names))


# Gene palettes for the model-hyperparameter and net-hygiene groups. Values are
# RELATIVE (deltas/multipliers on each asset's tuned baseline - see train_hybrid
# cb_params_for/lookback_for), so one candidate composes with 181 baselines.
DEPTH_DELTAS = (-2, -1, 0, 1, 2)
LR_MULTS = (0.5, 0.7, 1.0, 1.5, 2.0)
ITER_MULTS = (0.7, 1.0, 1.5)
LOOKBACK_DELTAS = (-10, -5, 0, 5, 10)
NET_SEED_CHOICES = (1, 3)
LABEL_MODES = ("direction", "rel_median", "triple_barrier")
TB_HORIZONS = (5, 10, 20)  # triple_barrier reuses label_window as the horizon H

_HYPER_DEFAULTS = (0, 1.0, 1.0, 0)
_NET_DEFAULTS = (1, 0, 0)


@dataclass
class Genome:
    """A composable cross-axis experiment: feature drops + extra DSL specs + labeling
    + relative model hyperparameters + net-hygiene toggles. The objective is NOT here
    (it is the QD fitness). Empty == production default.

    label_window doubles as the horizon H when label_mode == "triple_barrier".
    net_uniqueness is a SEPARATE gene from the label on purpose: the negative
    2026-07-11 triple-barrier A/B bundled them, so the agent could never tell which
    one hurt - now it can test the label with and without uniqueness weighting."""
    drops: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    label_mode: str = "direction"
    label_window: int = 30
    cb_depth_delta: int = 0
    cb_lr_mult: float = 1.0
    cb_iter_mult: float = 1.0
    lookback_delta: int = 0
    net_seeds: int = 1
    net_uniqueness: int = 0
    net_calibrate: int = 0


def _hyper_genes(g):
    return (g.cb_depth_delta, g.cb_lr_mult, g.cb_iter_mult, g.lookback_delta)


def _net_genes(g):
    return (g.net_seeds, g.net_uniqueness, g.net_calibrate)


def genome_to_env(g):
    """Compose the genome into one training env-override dict (empty - no overrides)."""
    env = dict(_feature_env(g.extra, [s["name"] for s in g.extra]))
    if g.drops:
        env["GTRADE_DROP_FEATURES"] = ",".join(g.drops)
    if g.label_mode == "triple_barrier":
        env["GTRADE_LABEL_MODE"] = "triple_barrier"
        env["GTRADE_LABEL_HORIZON"] = str(g.label_window)
    elif g.label_mode != "direction":
        env["GTRADE_LABEL_MODE"] = g.label_mode
        env["GTRADE_LABEL_WINDOW"] = str(g.label_window)
    if g.cb_depth_delta:
        env["GTRADE_CB_DEPTH_DELTA"] = str(g.cb_depth_delta)
    if g.cb_lr_mult != 1.0:
        env["GTRADE_CB_LR_MULT"] = str(g.cb_lr_mult)
    if g.cb_iter_mult != 1.0:
        env["GTRADE_CB_ITER_MULT"] = str(g.cb_iter_mult)
    if g.lookback_delta:
        env["GTRADE_LOOKBACK_DELTA"] = str(g.lookback_delta)
    if g.net_seeds > 1:
        env["GTRADE_NET_SEEDS"] = str(g.net_seeds)
    if g.net_uniqueness:
        env["GTRADE_NET_UNIQUENESS"] = "1"
    if g.net_calibrate:
        env["GTRADE_NET_CALIBRATE"] = "1"
    return env


def _spec_signature(spec):
    """Identity of a spec ignoring its name, used to dedup against the log."""
    return (spec["op"], tuple(spec.get("inputs") or []),
            tuple(sorted((spec.get("params") or {}).items())))


def _canon_genome(g):
    """direction ignores the window; canonicalize so equivalent genomes dedup."""
    if g.label_mode == "direction":
        g.label_window = 30
    return g


def genome_sig(g):
    """Canonical cross-run identity of a genome (drop order and the arbitrary
    spec NAMES are ignored; only the spec signatures matter).

    The hyper/nets gene groups enter the signature ONLY when they differ from the
    production default, so every genome signature recorded before those genes
    existed stays byte-identical - the tried-registry and the replication gate
    keep their history."""
    d = {
        "drops": sorted(g.drops),
        "extra": sorted(json.dumps(_spec_signature(s)) for s in g.extra),
        "label": [g.label_mode, g.label_window],
    }
    if _hyper_genes(g) != _HYPER_DEFAULTS:
        d["hyper"] = list(_hyper_genes(g))
    if _net_genes(g) != _NET_DEFAULTS:
        d["nets"] = list(_net_genes(g))
    return json.dumps(d, sort_keys=True)


def valid(g, active, prune_min):
    """Well-formedness against the active feature set and the prune floor."""
    aset = set(active)
    extra_names = {s.get("name") for s in g.extra}
    if any(d not in aset for d in g.drops):
        return False
    if set(g.drops) & extra_names:
        return False
    cols = aset | extra_names
    if any(not validate_spec(s, cols) for s in g.extra):
        return False
    if g.label_mode not in LABEL_MODES:
        return False
    if g.label_window <= 0:
        return False
    if g.cb_depth_delta not in DEPTH_DELTAS:
        return False
    if g.cb_lr_mult not in LR_MULTS or g.cb_iter_mult not in ITER_MULTS:
        return False
    if g.lookback_delta not in LOOKBACK_DELTAS:
        return False
    if g.net_seeds not in NET_SEED_CHOICES:
        return False
    if g.net_uniqueness not in (0, 1) or g.net_calibrate not in (0, 1):
        return False
    if len(aset) - len(set(g.drops)) < prune_min:
        return False
    return True


def random_genome(active, base_features):
    """A random valid starting genome for QD initialization."""
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    g = Genome()
    max_drops = max(0, len(active) - prune_min)
    n_drop = random.randint(0, min(3, max_drops))
    g.drops = random.sample(list(active), n_drop) if n_drop else []
    for i in range(random.randint(0, 2)):
        spec = _random_spec(base_features, "g%d_%d" % (random.randint(0, 99999), i), None)
        if spec and spec["name"] not in g.drops:
            g.extra.append(spec)
    r = random.random()
    if r < 0.35:
        g.label_mode = "rel_median"
        g.label_window = random.choice([20, 30, 60])
    elif r < 0.5:
        g.label_mode = "triple_barrier"
        g.label_window = random.choice(TB_HORIZONS)
    if random.random() < 0.3:
        _mutate_hyper(g)
    if random.random() < 0.2:
        _mutate_nets(g)
    return g if valid(g, active, prune_min) else Genome()


def _mutate_hyper(g):
    """Set one random hyperparameter gene to a random palette value (may be the
    default - that is how a gene walks back to baseline)."""
    gene = random.choice(("depth", "lr", "iter", "lookback"))
    if gene == "depth":
        g.cb_depth_delta = random.choice(DEPTH_DELTAS)
    elif gene == "lr":
        g.cb_lr_mult = random.choice(LR_MULTS)
    elif gene == "iter":
        g.cb_iter_mult = random.choice(ITER_MULTS)
    else:
        g.lookback_delta = random.choice(LOOKBACK_DELTAS)


def _mutate_nets(g):
    """Flip one net-hygiene gene."""
    gene = random.choice(("seeds", "uniq", "calib"))
    if gene == "seeds":
        g.net_seeds = random.choice([s for s in NET_SEED_CHOICES if s != g.net_seeds])
    elif gene == "uniq":
        g.net_uniqueness = 1 - g.net_uniqueness
    else:
        g.net_calibrate = 1 - g.net_calibrate


def mutate(g, active, base_features):
    """One random gene change; always returns a valid genome (or the input if no valid
    single change exists)."""
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    ops = ["add_drop", "rm_drop", "add_extra", "rm_extra", "flip_label", "win",
           "hyper", "nets"]
    random.shuffle(ops)
    for op in ops:
        ng = copy.deepcopy(g)
        if op == "add_drop":
            taken = set(ng.drops) | {s.get("name") for s in ng.extra}
            cand = [f for f in active if f not in taken]
            if cand:
                ng.drops.append(random.choice(cand))
        elif op == "rm_drop":
            if ng.drops:
                ng.drops.pop(random.randrange(len(ng.drops)))
        elif op == "add_extra":
            spec = _random_spec(base_features, "m%d" % random.randint(0, 99999), None)
            if spec and spec["name"] not in ng.drops:
                ng.extra.append(spec)
        elif op == "rm_extra":
            if ng.extra:
                ng.extra.pop(random.randrange(len(ng.extra)))
        elif op == "flip_label":
            ng.label_mode = random.choice([m for m in LABEL_MODES if m != ng.label_mode])
            if ng.label_mode == "triple_barrier" and ng.label_window not in TB_HORIZONS:
                ng.label_window = random.choice(TB_HORIZONS)
        elif op == "win":
            pool = TB_HORIZONS if ng.label_mode == "triple_barrier" else (20, 30, 60)
            ng.label_window = random.choice(
                [w for w in pool if w != ng.label_window] or [pool[1]])
        elif op == "hyper":
            _mutate_hyper(ng)
        elif op == "nets":
            _mutate_nets(ng)
        if ng != g and valid(ng, active, prune_min):
            return ng
    return g


def crossover(g1, g2, active):
    """Uniform crossover over gene-groups; resolves drop-extra conflicts (drop loses)
    and the prune floor; always returns a valid genome."""
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    child = Genome()
    child.drops = copy.deepcopy(g1.drops if random.random() < 0.5 else g2.drops)
    child.extra = copy.deepcopy(g1.extra if random.random() < 0.5 else g2.extra)
    if random.random() < 0.5:
        child.label_mode, child.label_window = g1.label_mode, g1.label_window
    else:
        child.label_mode, child.label_window = g2.label_mode, g2.label_window
    hp = g1 if random.random() < 0.5 else g2
    child.cb_depth_delta, child.cb_lr_mult = hp.cb_depth_delta, hp.cb_lr_mult
    child.cb_iter_mult, child.lookback_delta = hp.cb_iter_mult, hp.lookback_delta
    np_ = g1 if random.random() < 0.5 else g2
    child.net_seeds, child.net_uniqueness = np_.net_seeds, np_.net_uniqueness
    child.net_calibrate = np_.net_calibrate
    aset = set(active)
    cols = aset | {s.get("name") for s in child.extra}
    child.extra = [s for s in child.extra if validate_spec(s, cols)]
    extra_names = {s.get("name") for s in child.extra}
    child.drops = [d for d in dict.fromkeys(child.drops) if d in aset and d not in extra_names]
    while len(aset) - len(set(child.drops)) < prune_min and child.drops:
        child.drops.pop()
    return child


_FLOOR_EDGES = (-1.0, -0.25, 0.25, 1.0)
_COUNT_EDGES = (12, 18, 24, 30)


def _bin(value, edges):
    """Bin index 0..len(edges) by right-insertion into the edge list."""
    return bisect.bisect_right(edges, value)


def fitness(rows, base_score):
    """Archive cell quality: the MEAN paired Score delta."""
    return _objective_delta(rows, base_score, "mean")[0]


def behavior(genome, rows, base_score, active):
    """Behavior descriptors: (worst-asset delta bin = floor lift, feature-count bin)."""
    min_delta = _objective_delta(rows, base_score, "min")[0]
    count = len(active) - len(set(genome.drops)) + len(genome.extra)
    return _bin(min_delta, _FLOOR_EDGES), _bin(count, _COUNT_EDGES)


_QD_ARCHIVE_PATH = os.path.join(BASE, "_qd_archive.json")


def archive_put(archive, genome, rows, base_score, active):
    """Place a genome in its (floor, complexity) niche if it beats the niche's mean
    fitness (or the niche is empty). Returns True when stored."""
    bd = behavior(genome, rows, base_score, active)
    key = "%d_%d" % bd
    f = fitness(rows, base_score)
    cur = archive.get(key)
    if cur is None or f > cur["fitness"]:
        archive[key] = {"genome": genome, "fitness": f, "rows": rows}
        return True
    return False


def _qd_save(archive):
    out = {k: {"genome": asdict(v["genome"]), "fitness": v["fitness"]}
           for k, v in archive.items()}
    with open(_QD_ARCHIVE_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh)


def _qd_load():
    """Reload the archive (genomes only; rows are re-derived on resume) or {}."""
    if not os.path.exists(_QD_ARCHIVE_PATH):
        return {}
    try:
        with open(_QD_ARCHIVE_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
        return {k: {"genome": Genome(**v["genome"]), "fitness": v["fitness"], "rows": []}
                for k, v in raw.items()}
    except Exception:
        return {}


def _llm_child(elites, active, base_features):
    """A genome proposed by the LLM, converted and validated; None on ANY
    problem (the QD loop then falls back to the evolutionary operators, so an
    unreachable Ollama can never kill the search)."""
    top = sorted(elites, key=lambda e: e["fitness"], reverse=True)[:5]
    parent = random.choice(top)["genome"]
    summary = [{"genome": asdict(e["genome"]), "fitness": e["fitness"]} for e in top]
    avoid = ar_memory.tried_recent("genome", 30)
    try:
        obj = llm_proposer.propose_genome(
            asdict(parent), summary, active, base_features, avoid=avoid)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        g = Genome(drops=list(obj.get("drops") or []),
                   extra=list(obj.get("extra") or []),
                   label_mode=str(obj.get("label_mode", "direction")),
                   label_window=int(obj.get("label_window", 30)))
    except (TypeError, ValueError):
        return None
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    return g if valid(g, active, prune_min) else None


def _surrogate_child(archive, active, base_features):
    """Generate up to n_candidates() unseen children via mutate/crossover (valid by
    construction, like the plain loop), score each with a surrogate fit on the archive,
    and return the highest-predicted previously-untried child. None when the surrogate
    cannot be fit or every candidate is already tried."""
    elites = list(archive.values())
    samples = [(qd_surrogate.genome_vector(e["genome"], active, base_features), e["fitness"])
               for e in elites]
    model = qd_surrogate.fit_surrogate(samples)
    if model is None:
        return None
    best, best_pred = None, None
    for _ in range(qd_surrogate.n_candidates()):
        parent = random.choice(elites)["genome"]
        if len(elites) >= 2 and random.random() < 0.5:
            child = crossover(parent, random.choice(elites)["genome"], active)
        else:
            child = mutate(parent, active, base_features)
        child = _canon_genome(child)
        if ar_memory.tried_seen("genome", genome_sig(child)):
            continue
        pred = qd_surrogate.predict(
            model, qd_surrogate.genome_vector(child, active, base_features))
        if best is None or pred > best_pred:
            best, best_pred = child, pred
    return best


def next_child(archive, active, base_features, attempts=10):
    """One unseen child genome for the QD loop: LLM-proposed (when the llm proposer is selected, with probability GTRADE_AR_QD_LLM_P) else mutate/crossover retried against the tried-registry.
    None when the archive is empty or every attempt lands on an already-tried genome."""
    elites = list(archive.values())
    if not elites:
        return None
    if llm_proposer.llm_selected() and random.random() < float(
            os.getenv("GTRADE_AR_QD_LLM_P") or "0.3"):
        child = _llm_child(elites, active, base_features)
        if child is not None:
            child = _canon_genome(child)
            if not ar_memory.tried_seen("genome", genome_sig(child)):
                return child
    if qd_surrogate.surrogate_on():
        try:
            child = _surrogate_child(archive, active, base_features)
        except Exception:
            child = None
        if child is not None:
            return child
    for _ in range(attempts):
        parent = random.choice(elites)["genome"]
        if len(elites) >= 2 and random.random() < 0.5:
            child = crossover(parent, random.choice(elites)["genome"], active)
        else:
            child = mutate(parent, active, base_features)
        child = _canon_genome(child)
        if not ar_memory.tried_seen("genome", genome_sig(child)):
            return child
    return None


def run_qd(train_fn=None):
    """MAP-Elites: illuminate an archive of diverse genomes via the cheap CB screen,
    then full-evaluate + honest-gate the top elites. Returns the archive."""
    from core.features import active_candidate_features

    base_fn = train_base_cached if train_fn is None else train_fn
    train_fn = train_fn or train_env
    base_features = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_z", "rsi",
                     "macd_hist", "bb_pos", "trend_strength", "atr"]
    active = active_candidate_features()
    init = int(os.getenv("GTRADE_AR_QD_INIT", "8"))
    n_final = int(os.getenv("GTRADE_AR_QD_FINAL", "3"))

    screen_base = base_fn(SELECTION_ASSETS, {"GTRADE_SCREEN_ONLY": "1"})
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in screen_base}

    def _screen_eval(g):
        return train_fn(SELECTION_ASSETS, screen_env(genome_to_env(g)))

    archive = _qd_load()
    if not archive:
        for _ in range(init):
            g = _canon_genome(random_genome(active, base_features))
            ar_memory.tried_add("genome", genome_sig(g))
            archive_put(archive, g, _screen_eval(g), base_score, active)
        _qd_save(archive)

    # NOTE: archive illumination always uses the cheap raw CB screen (fitness vs the
    # CB base_score); GTRADE_AR_SCORE_BASIS=neural only re-scores the FINAL elite gate
    # on neural contribution - it does NOT change which genomes become elites. Feeding
    # contribution into illumination would require a full ensemble train per step.
    max_misses = int(os.getenv("GTRADE_AR_QD_MAX_MISSES", "5"))
    misses = 0
    for _ in range(BUDGET):
        if not archive:
            break
        child = next_child(archive, active, base_features)
        if child is None:
            misses += 1
            print("[qd] dedup: no unseen child this step, skipping.")
            if misses >= max_misses:
                print("[qd] search space exhausted vs the tried-registry after %d "
                      "misses; stopping early (raise GTRADE_AR_QD_MAX_MISSES to "
                      "keep trying)." % misses)
                break
            continue
        misses = 0
        ar_memory.tried_add("genome", genome_sig(child))
        archive_put(archive, child, _screen_eval(child), base_score, active)
        _qd_save(archive)

    elites = sorted(archive.values(), key=lambda e: e["fitness"], reverse=True)[:n_final]
    if not elites:
        print("[qd] no elites in the archive.")
        finding_winners = []
    else:
        obj = _objective()
        basis = _score_basis()
        ho_base_full, ho_base_contrib = _heldout_eval(HELDOUT_ASSETS, {}, base_fn)
        base_contrib = {r["Asset"]: r["Score"] for r in ho_base_contrib}
        results = []
        for e in elites:
            g = e["genome"]
            var_full, var_contrib = _heldout_eval(
                HELDOUT_ASSETS, genome_to_env(g), train_fn)
            nl, _d = _objective_delta(var_contrib, base_contrib, "mean")
            nl = round(nl, 4) if _d else None
            if basis == "neural":
                p, value, _d, tag = holdout_stats(ho_base_contrib, var_contrib, obj)
            else:
                p, value, _d, tag = holdout_stats(ho_base_full, var_full, obj)
            results.append((g, p, value, tag, nl))
        flags = benjamini_hochberg([r[1] for r in results])
        ts_qd = datetime.utcnow().isoformat()
        finding_winners = []
        for (g, p, value, tag, nl), s in zip(results, flags):
            ok = bool(s and value > _adopt_floor(obj))
            replicated = clears = None
            if ok:
                gsig = genome_sig(g)
                replicated = ar_memory.replication_seen(gsig)
                clears = ar_memory.replication_add(gsig, ts_qd)
                if ar_wiki.wiki_on() and clears >= 2:
                    ar_wiki.note_replicated(gsig, "replicated (%d clears)" % clears)
            finding_winners.append({"axis": "qd", "genome": asdict(g), "p": p,
                                    "value": value, "tag": tag, "adoptable": ok,
                                    "neural_lift": nl, "replicated": bool(replicated),
                                    "clears": clears or 0})
            nl_str = "" if nl is None else " | neural_lift %+.2f" % nl
            print("[qd] elite drops=%s label=%s/%d extra=%d: %s | %s%s" % (
                g.drops, g.label_mode, g.label_window, len(g.extra),
                _gate_verdict(ok, bool(replicated), clears), tag, nl_str))
    ar_memory.findings_append({
        "ts": datetime.utcnow().isoformat(), "mode": "qd",
        "budget": BUDGET, "winners": finding_winners})
    if ar_wiki.wiki_on():
        ar_wiki.compile_wiki()
    mem = ar_memory.findings_summary()
    print("[auto-research] memory: %d experiments tried, %d adoptable, %d replicated so far."
          % (mem["experiments"], mem["adoptable"], mem["replicated"]))
    print("[qd] %d niches illuminated; review _qd_archive.json; nothing auto-adopted." % len(archive))
    return archive


def _regate_candidates(archive_raw, findings, k):
    """Distinct stored candidate genomes to re-gate, capped at k. Findings winners
    (they carry a held-out `value` + neural_lift) rank first by that value; archive-
    only elites (a bare selection `fitness`) fill the remaining slots. Pure - no
    training. archive_raw is the raw JSON dict {cell: {"genome": <dict>, "fitness"}}."""
    by_sig = {}  # sig - (Genome, value, neural_lift)
    for rec in findings or []:
        for w in rec.get("winners", []):
            gd, v = w.get("genome"), w.get("value")
            if not isinstance(gd, dict) or v is None:
                continue
            try:
                g = Genome(**gd)
            except (TypeError, ValueError):
                continue
            sig = genome_sig(g)
            if sig not in by_sig or v > by_sig[sig][1]:
                by_sig[sig] = (g, v, w.get("neural_lift"))
    found = set(by_sig)
    arch = []
    for cell in (archive_raw or {}).values():
        gd = cell.get("genome") if isinstance(cell, dict) else None
        f = cell.get("fitness") if isinstance(cell, dict) else None
        if not isinstance(gd, dict) or f is None:
            continue
        try:
            g = Genome(**gd)
        except (TypeError, ValueError):
            continue
        if genome_sig(g) not in found:
            arch.append((g, f, None))
    findings_ranked = sorted(by_sig.values(),
                             key=lambda c: (c[1], c[2] if c[2] is not None else -1e9),
                             reverse=True)
    arch_ranked = sorted(arch, key=lambda c: c[1], reverse=True)
    return (findings_ranked + arch_ranked)[:k]


def _regate_load_archive_raw():
    if not os.path.exists(_QD_ARCHIVE_PATH):
        return {}
    try:
        with open(_QD_ARCHIVE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


_REGATE_PROGRESS_PATH = os.path.join(BASE, "_regate_progress.json")


def _say(msg):
    """Print + mirror into the shared log file, so a lost console (the way the
    2026-07-13 re-gate died silently) never loses the run trail."""
    print(msg)
    logger.info(msg)


def _regate_progress_load(base_sig):
    """Per-candidate checkpoint of a re-gate run: {gsig: result fields}. Discarded
    (fresh start) when the stored base signature no longer matches - results are only
    comparable within one data snapshot + objective/basis/screen config."""
    try:
        with open(_REGATE_PROGRESS_PATH, encoding="utf-8") as fh:
            saved = json.load(fh)
    except Exception:
        return {}
    return saved.get("done", {}) if saved.get("base_sig") == base_sig else {}


def _regate_progress_save(base_sig, done):
    with open(_REGATE_PROGRESS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"base_sig": base_sig, "done": done}, fh, ensure_ascii=True, indent=2)


def _regate_progress_clear():
    try:
        os.remove(_REGATE_PROGRESS_PATH)
    except OSError:
        pass


def _candidate_train_cached(subset, env, gsig):
    """train_env with a cross-run cache keyed by the genome signature (the env embeds
    temp spec-file paths, so the raw env cannot key the cache). The CB-only screen
    train and the full train cache under different kinds; the screen's CB train is
    therefore REUSED by the held-out eval instead of being trained twice. A resumed
    or repeated re-gate reuses every finished candidate train (hours each)."""
    kind = "cb" if env.get("GTRADE_SCREEN_ONLY") else "full"
    key = ar_memory.genome_key(subset, gsig, kind)
    rows = ar_memory.cache_get(key)
    if rows is not None:
        _say("[regate] candidate cache hit (%s): %s" % (kind, gsig[:12]))
        return rows
    rows = train_env(subset, env)
    if rows:
        ar_memory.cache_put(key, rows)
    return rows


def regate(k=8, screen=False):
    """Re-evaluate the best already-found candidate genomes under the CURRENT gate
    (Wilcoxon + enlarged held-out), reusing the 900+ prior experiments instead of
    re-running the search. Trains only the top-k on the held-out set. Adopts nothing;
    journals mode='regate' and feeds the same replication gate.

    Crash-safe: every finished candidate checkpoints to _regate_progress.json and its
    trains cache by genome signature, so an interrupted run RESUMES (same data + gate
    config) instead of restarting from zero."""
    cands = _regate_candidates(_regate_load_archive_raw(), ar_memory.findings_all(), k)
    if not cands:
        _say("[regate] no stored candidates to re-gate.")
        return
    obj, basis = _objective(), _score_basis()
    _say("[regate] %d candidate(s) | objective=%s basis=%s screen=%s"
         % (len(cands), obj, basis, screen))
    ho_base_full, ho_base_contrib = _heldout_eval(HELDOUT_ASSETS, {}, train_base_cached)
    base_contrib = {r["Asset"]: r["Score"] for r in ho_base_contrib}
    base_sig = "|".join([ar_memory.base_key(HELDOUT_ASSETS, {}), obj, basis,
                         "screen" if screen else "noscreen"])
    done = _regate_progress_load(base_sig)
    if done:
        _say("[regate] resuming: %d candidate(s) already evaluated." % len(done))
    if screen:
        screen_base = {r["Asset"]: r["Score"]
                       for r in train_base_cached(HELDOUT_ASSETS, screen_env({}))}
        cands = [c for c in cands
                 if genome_sig(c[0]) in done or _regate_passes_screen(c[0], screen_base)]
        _say("[regate] %d candidate(s) after the CB-only screen." % len(cands))
    results = []
    for i, (g, old_score, old_nl) in enumerate(cands, start=1):
        gsig = genome_sig(g)
        if gsig in done:
            r = done[gsig]
            results.append((g, old_score, r["p"], r["value"], r["tag"], r["nl"]))
            continue
        _say("[regate] %d/%d evaluating %s (stored score %.2f)..."
             % (i, len(cands), gsig[:12], old_score))

        def _fn(s, e, _sig=gsig):
            return _candidate_train_cached(s, e, _sig)

        var_full, var_contrib = _heldout_eval(HELDOUT_ASSETS, genome_to_env(g), _fn)
        nl, _d = _objective_delta(var_contrib, base_contrib, "mean")
        nl = round(nl, 4) if _d else None
        if basis == "neural":
            p, value, _d, tag = holdout_stats(ho_base_contrib, var_contrib, obj)
        else:
            p, value, _d, tag = holdout_stats(ho_base_full, var_full, obj)
        results.append((g, old_score, p, value, tag, nl))
        done[gsig] = {"p": p, "value": value, "tag": tag, "nl": nl}
        _regate_progress_save(base_sig, done)
        _say("[regate] %d/%d done: %s" % (i, len(cands), tag))
    flags = benjamini_hochberg([r[2] for r in results])
    ts = datetime.utcnow().isoformat()
    finding_winners = []
    for (g, old_score, p, value, tag, nl), s in zip(results, flags):
        ok = bool(s and value > _adopt_floor(obj))
        replicated = clears = None
        if ok:
            gsig = genome_sig(g)
            replicated = ar_memory.replication_seen(gsig)
            clears = ar_memory.replication_add(gsig, ts)
        finding_winners.append({"axis": "regate", "genome": asdict(g), "p": p,
                                "value": value, "tag": tag, "adoptable": ok,
                                "neural_lift": nl, "replicated": bool(replicated),
                                "clears": clears or 0})
        nl_str = "" if nl is None else " | neural_lift %+.2f" % nl
        _say("[regate] old %.2f - %s | %s%s" % (
            old_score, _gate_verdict(ok, bool(replicated), clears), tag, nl_str))
    ar_memory.findings_append({"ts": ts, "mode": "regate", "k": k,
                               "screen": bool(screen), "winners": finding_winners})
    _regate_progress_clear()
    _say("[regate] re-gated %d candidate(s); nothing adopted automatically." % len(results))


def _regate_passes_screen(g, screen_base):
    """Optional cheap CB-only prefilter: keep the candidate unless its CB-only held-out
    mean delta vs the CB-only base is clearly negative (below the existing SCREEN_MIN
    floor). `screen_base` is a {asset: CB-only Score} dict. The train is cached by
    genome signature, so the held-out eval reuses it as its CB side."""
    try:
        rows = _candidate_train_cached(HELDOUT_ASSETS, screen_env(genome_to_env(g)),
                                       genome_sig(g))
        d, deltas = _objective_delta(rows, screen_base, "mean")
        return (not deltas) or d >= SCREEN_MIN
    except Exception:
        return True


_SAMPLE_WINDOWS = [5, 10, 20, 50]
_SAMPLE_KS = [1, 2, 3, 5, 10]
_SAMPLE_HORIZONS = [1, 2, 5]
_SINGLE_OPS = ["zscore", "lag", "diff", "rolling"]
_PAIR_OPS = ["ratio", "interaction"]
_AGGS = ["mean", "std", "sum"]
_LEAD_LEADERS = ["sp500", "vix", "btc", "gold", "dxy", "tnx"]


def _random_spec(base_features, name, prefer):
    """One spec sampled from the DSL space. prefer biases the input choice toward
    columns that have shown positive deltas; it falls back to base_features."""
    pool = prefer if prefer else base_features
    if len(base_features) >= 2 and random.random() < 0.35:
        op = random.choice(_PAIR_OPS)
        a = random.choice(pool)
        b = random.choice([f for f in base_features if f != a])
        return {"name": name, "op": op, "inputs": [a, b], "params": {}}
    if random.random() < 0.12:
        return {"name": name, "op": "lead_lag",
                "inputs": [random.choice(_LEAD_LEADERS)],
                "params": {"horizon": random.choice(_SAMPLE_HORIZONS)}}
    op = random.choice(_SINGLE_OPS)
    params = {}
    if op in ("zscore", "rolling"):
        params["window"] = random.choice(_SAMPLE_WINDOWS)
        if op == "rolling":
            params["agg"] = random.choice(_AGGS)
    else:
        params["k"] = random.choice(_SAMPLE_KS)
    return {"name": name, "op": op, "inputs": [random.choice(pool)], "params": params}


def _mutate(spec, name):
    """A small variation of a past spec: tweak its one numeric param, new name."""
    out = {"name": name, "op": spec["op"],
           "inputs": list(spec.get("inputs") or []),
           "params": dict(spec.get("params") or {})}
    p = out["params"]
    if "window" in p:
        p["window"] = random.choice(_SAMPLE_WINDOWS)
    elif "k" in p:
        p["k"] = random.choice(_SAMPLE_KS)
    elif "horizon" in p:
        p["horizon"] = random.choice(_SAMPLE_HORIZONS)
    if out["op"] == "rolling":
        p["agg"] = random.choice(_AGGS)
    return out


def propose_evolutionary(log, base_features, avoid=None):
    """Autonomous (no LLM) proposer. Explores the DSL early; once a past spec shows
    a positive mean selection delta it biases input choice toward the good inputs and
    mutates the best spec. Reads the log, dedups against it, returns one valid spec.
    avoid is accepted for a uniform proposer signature but unused (this path already
    dedups against the log and the tried registry)."""
    seed = os.getenv("GTRADE_AR_SEED")
    if seed:
        random.seed(int(seed) + len(log))
    cols = set(base_features)
    seen = set()
    scored = []
    good_inputs = []
    for e in log:
        for s in (e.get("spec") or []):
            seen.add(_spec_signature(s))
        sc = e.get("score")
        if sc is not None and e.get("spec"):
            scored.append((sc, e["spec"]))
            if sc > 0:
                for s in e["spec"]:
                    good_inputs += [c for c in (s.get("inputs") or []) if c in cols]
    best = max(scored, key=lambda x: x[0]) if scored else None
    for attempt in range(30):
        name = "ar_%d_%d" % (len(log), attempt)
        if best and best[0] > 0 and random.random() < 0.6:
            spec = _mutate(random.choice(best[1]), name)
        else:
            spec = _random_spec(base_features, name, good_inputs)
        if (validate_spec(spec, cols) and _spec_signature(spec) not in seen
                and not ar_memory.tried_seen("spec", json.dumps(_spec_signature(spec)))):
            return [spec]
    return []


def _select_proposer():
    """Evolutionary (no LLM) by default; the LLM proposer when GTRADE_AR_PROPOSER=llm."""
    if llm_proposer.llm_selected():
        return llm_proposer.propose_specs
    return propose_evolutionary


PRESCREEN_MIN_ABS_CORR = float(os.getenv("AR_PRESCREEN_MIN", "0.02"))


def _prescreen_ok(spec, df, target_col="target", threshold=None):
    """Cheap univariate screen run BEFORE the expensive training: keep a spec only
    if its materialized feature has at least a small absolute correlation with the
    target. lead_lag (an engine op), a missing target, or any error passes through,
    to be judged by the full A/B."""
    if threshold is None:
        threshold = PRESCREEN_MIN_ABS_CORR
    if spec.get("op") == "lead_lag" or target_col not in getattr(df, "columns", []):
        return True
    try:
        import pandas as pd
        from core.feature_dsl import materialize
        c = pd.Series(materialize(df, spec)).corr(df[target_col])
        if c != c:  # NaN correlation (constant feature)
            return True
        return abs(c) >= threshold
    except Exception:
        return True


SCREEN_MIN = float(os.getenv("GTRADE_AR_SCREEN_MIN", "0.0"))


def _screen_on():
    """Whether the cheap CatBoost-only screen runs before the full eval (default on)."""
    return (os.getenv("GTRADE_AR_SCREEN", "1") or "1").strip() not in ("0", "false", "False", "")


def screen_env(env):
    """A copy of the candidate env with the CB-only screen flag set."""
    return {**env, "GTRADE_SCREEN_ONLY": "1"}


def _passes_screen(axis, selected, train_fn, screen_base, screen_min, objective="mean"):
    """Cheap CB-only screen of one candidate. Returns (passed, proxy_delta). A failed
    screen train (empty rows) returns (True, 0.0) - never drop a candidate on a screen
    infra failure, fall through to the full eval."""
    srows = train_fn(SELECTION_ASSETS, screen_env(axis.to_env(selected)))
    if not srows:
        return True, 0.0
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in screen_base}
    delta, _ = _objective_delta(srows, base_score, objective)
    return delta > screen_min, delta


_STATE_PATH = os.path.join(BASE, "_auto_research_state.json")
_LOG_PATH = os.path.join(BASE, "_auto_research_log.json")


def load_state():
    """The persisted loop state (cached base, kept set, log) for resume, or {}."""
    if not os.path.exists(_STATE_PATH):
        return {}
    try:
        with open(_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)
    # keep the human-readable log file in sync
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(state.get("log", []), f, ensure_ascii=True, indent=2)


@dataclass
class Axis:
    """One search axis. `propose(log)` returns candidate dicts; `to_env(selected)`
    turns the selected candidate (additive: the kept+new list; select_best: a single
    candidate) into training env overrides; `kind` is "additive" or "select_best";
    `validate(cand, selected)` and `prescreen(cand, screen_df)` default to accepting;
    `sig(cand)` returns (kind, signature) for the permanent tried-registry; None = not registered."""
    name: str
    propose: object
    to_env: object
    kind: str = "additive"
    validate: object = None
    prescreen: object = None
    sig: object = None

    def ok(self, cand, selected, screen_df):
        v = self.validate is None or self.validate(cand, selected)
        p = self.prescreen is None or self.prescreen(cand, screen_df)
        return v and p


def run_axis(axis, budget, base_rows, train_fn, screen_df=None, prior_log=None, persist=None,
             screen_base=None, screen_min=0.0):
    """Generalized search over one axis, sharing the mean-delta gate.

    additive: forward-selection - keep the running list when the cumulative mean
    Score delta improves. select_best: evaluate each proposed candidate against the
    base and keep the single best whose delta beats the base. `persist(log)` is
    called after each iteration (None = no-op).
    budget counts NEW iterations for THIS run (the persisted log only sets the
    resume point, it does not consume budget)."""
    def _mark_tried(cands):
        if axis.sig is None:
            return
        for c in (cands if isinstance(cands, list) else [cands]):
            kind, s = axis.sig(c)
            ar_memory.tried_add(kind, s)

    base_score = {r["Asset"]: r.get("Score", 0.0) for r in base_rows}
    objective = _objective()
    log = list(prior_log or [])
    persist = persist or (lambda _log: None)
    # Per-axis entries always carry axis == axis.name; filter explicitly rather than
    # relying incidentally on the caller (_persist) bucketing the log per axis.
    axis_log = [e for e in log if e.get("axis") in (None, axis.name)]

    if axis.kind == "additive":
        kept = [c for e in axis_log if e.get("accepted") for c in e.get("cand", [])]
        kept_delta = max((e.get("cand_mean_delta", 0.0) for e in axis_log if e.get("accepted")), default=0.0)
        start = len(axis_log)
        for i in range(start, start + budget):
            proposed = axis.propose(log)
            new = [c for c in proposed if axis.ok(c, kept, screen_df)]
            if not new:
                log.append({"axis": axis.name, "iter": i, "cand": [], "note": "no valid/screened candidate"})
                persist(log)
                continue
            cand = kept + new
            if screen_base is not None:
                passed, sdelta = _passes_screen(axis, cand, train_fn, screen_base, screen_min, objective)
                if not passed:
                    entry = {"axis": axis.name, "iter": i, "cand": new,
                             "screen_delta": sdelta, "note": "screened out"}
                    log.append(entry)
                    _mark_tried(new)
                    persist(log)
                    continue
            rows = score_rows(SELECTION_ASSETS, axis.to_env(cand), train_fn)
            delta, _ = _objective_delta(rows, base_score, objective)
            entry = {"axis": axis.name, "iter": i, "cand": new,
                     "cand_mean_delta": delta, "score": delta - kept_delta}
            if delta - kept_delta > 1e-9:
                kept, kept_delta = cand, delta
                entry["accepted"] = True
            log.append(entry)
            _mark_tried(new)
            persist(log)
        return {"axis": axis.name, "kept": kept, "kept_delta": kept_delta, "log": log}

    # select_best
    tried = {json.dumps(e["cand"], sort_keys=True) for e in axis_log
             if "cand" in e and isinstance(e["cand"], dict)}
    accepted = [e for e in axis_log if e.get("accepted")]
    if accepted:
        _be = max(accepted, key=lambda e: e.get("cand_mean_delta", 0.0))
        best, best_delta = _be["cand"], _be.get("cand_mean_delta", 0.0)
    else:
        best, best_delta = None, 0.0
    proposed = [c for c in axis.propose(log) if axis.ok(c, best, screen_df)]
    i = len([e for e in axis_log if "iter" in e])
    stop = i + budget
    for cand in proposed:
        if i >= stop:
            break
        if json.dumps(cand, sort_keys=True) in tried:
            continue
        if screen_base is not None:
            passed, sdelta = _passes_screen(axis, cand, train_fn, screen_base, screen_min, objective)
            if not passed:
                log.append({"axis": axis.name, "iter": i, "cand": cand,
                            "screen_delta": sdelta, "note": "screened out"})
                _mark_tried(cand)
                persist(log)
                i += 1
                continue
        rows = score_rows(SELECTION_ASSETS, axis.to_env(cand), train_fn)
        delta, _ = _objective_delta(rows, base_score, objective)
        entry = {"axis": axis.name, "iter": i, "cand": cand, "cand_mean_delta": delta}
        if delta > best_delta + 1e-9:
            best, best_delta = cand, delta
            entry["accepted"] = True
        log.append(entry)
        _mark_tried(cand)
        persist(log)
        i += 1
    return {"axis": axis.name, "best": best, "best_delta": best_delta, "log": log}


def make_features_axis(base_features):
    """The existing feature-DSL search as an axis (behavior-preserving). validate
    grows the available-column set with the names of already-kept specs, matching the
    old forward-selection."""
    proposer = _select_proposer()

    def _validate(cand, kept):
        cols = set(base_features) | {s["name"] for s in (kept or [])}
        return validate_spec(cand, cols)

    return Axis(
        name="features",
        propose=lambda log: proposer(log, base_features,
                                     avoid=ar_memory.tried_recent("spec", 30)),
        to_env=lambda selected: _feature_env(selected, [s["name"] for s in selected]),
        kind="additive",
        validate=_validate,
        prescreen=lambda cand, screen_df: True if screen_df is None else _prescreen_ok(cand, screen_df),
        sig=lambda c: ("spec", json.dumps(_spec_signature(c))),
    )


LABEL_WINDOWS = (20, 30, 60)


def _label_env(cand):
    """Env overrides for one labeling candidate. triple_barrier's window is its
    horizon H (same convention as the genome's label_window)."""
    if cand["mode"] == "triple_barrier":
        return {"GTRADE_LABEL_MODE": "triple_barrier",
                "GTRADE_LABEL_HORIZON": str(cand["window"])}
    return {"GTRADE_LABEL_MODE": cand["mode"],
            "GTRADE_LABEL_WINDOW": str(cand["window"])}


def make_labeling_axis():
    """Sweep the alternative label modes (direction is the base): rel_median windows
    and triple_barrier horizons. select_best: keep the single best candidate whose
    mean Score delta beats the base. Unlike the 2026-07-11 manual A/B, a
    triple_barrier candidate here does NOT bundle uniqueness weighting - that is a
    separate net-hygiene gene/axis, so the two effects are measured apart."""
    def _propose(log):
        tried = {(e["cand"].get("mode"), e["cand"].get("window")) for e in log
                 if isinstance(e.get("cand"), dict) and "window" in e["cand"]}
        cands = ([{"mode": "rel_median", "window": w} for w in LABEL_WINDOWS]
                 + [{"mode": "triple_barrier", "window": h} for h in TB_HORIZONS])
        cands = [c for c in cands if (c["mode"], c["window"]) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "label", json.dumps(c, sort_keys=True))]

    return Axis(
        name="labeling",
        propose=_propose,
        to_env=_label_env,
        kind="select_best",
        sig=lambda c: ("label", json.dumps(c, sort_keys=True)),
    )


# --- model-hyperparameter axis (relative deltas/multipliers, see Genome) -----

HYPER_CANDIDATES = (
    {"cb_depth_delta": -1}, {"cb_depth_delta": 1},
    {"cb_lr_mult": 0.5}, {"cb_lr_mult": 2.0},
    {"cb_iter_mult": 1.5}, {"cb_iter_mult": 0.7},
    {"lookback_delta": -5}, {"lookback_delta": 5}, {"lookback_delta": 10},
)

_HYPER_ENV_KEYS = {
    "cb_depth_delta": "GTRADE_CB_DEPTH_DELTA",
    "cb_lr_mult": "GTRADE_CB_LR_MULT",
    "cb_iter_mult": "GTRADE_CB_ITER_MULT",
    "lookback_delta": "GTRADE_LOOKBACK_DELTA",
}


def hyper_env(cand):
    """Env overrides for one hyperparameter candidate dict (one or more genes)."""
    return {_HYPER_ENV_KEYS[k]: str(v) for k, v in cand.items()}


def make_hyper_axis():
    """One-gene-at-a-time sweep of the relative model-hyperparameter overrides
    (train_hybrid applies them on top of each asset's optuna baseline).
    select_best: keep the single best override whose delta beats the base."""
    def _propose(log):
        tried = {json.dumps(e["cand"], sort_keys=True) for e in log
                 if isinstance(e.get("cand"), dict)}
        cands = [c for c in HYPER_CANDIDATES
                 if json.dumps(c, sort_keys=True) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "hyper", json.dumps(c, sort_keys=True))]

    return Axis(
        name="hyper",
        propose=_propose,
        to_env=hyper_env,
        kind="select_best",
        sig=lambda c: ("hyper", json.dumps(c, sort_keys=True)),
    )


# --- net-hygiene axis (SP-2 levers as searchable candidates) ------------------

NETS_CANDIDATES = (
    {"seeds": 3},
    {"calibrate": 1},
    {"seeds": 3, "calibrate": 1},
)

_NETS_ENV_KEYS = {"seeds": "GTRADE_NET_SEEDS", "uniqueness": "GTRADE_NET_UNIQUENESS",
                  "calibrate": "GTRADE_NET_CALIBRATE"}


def nets_env(cand):
    return {_NETS_ENV_KEYS[k]: str(v) for k, v in cand.items()}


def make_nets_axis():
    """Sweep the net-hygiene levers (seed-averaging, per-net calibration).
    Uniqueness weighting is intentionally NOT proposed alone - it is a no-op under
    the next-bar base label (needs GTRADE_LABEL_HORIZON > 1); the QD genome can
    still combine it with a triple_barrier label."""
    def _propose(log):
        tried = {json.dumps(e["cand"], sort_keys=True) for e in log
                 if isinstance(e.get("cand"), dict)}
        cands = [c for c in NETS_CANDIDATES
                 if json.dumps(c, sort_keys=True) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "nets", json.dumps(c, sort_keys=True))]

    return Axis(
        name="nets",
        propose=_propose,
        to_env=nets_env,
        kind="select_best",
        sig=lambda c: ("nets", json.dumps(c, sort_keys=True)),
    )


def make_pruning_axis(base_features):
    """Backward-elimination over the active candidate features. select-drop additive:
    propose dropping one feature at a time, keep a drop when it lifts the cumulative
    mean Score delta. The active set and the floor are read once at build time."""
    from core.features import active_candidate_features
    active = list(active_candidate_features())
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))

    def _propose(log):
        # one candidate per round: the next droppable feature not yet tried in any
        # prior round (accepted or rejected), matching the additive contract that
        # run_axis treats a proposer's whole result as ONE increment.
        tried = {c["drop"] for e in log for c in e.get("cand", [])
                 if isinstance(c, dict) and "drop" in c}
        remaining = [f for f in active if f not in tried
                     and not ar_memory.tried_seen("drop", f)]
        return [{"drop": remaining[0]}] if remaining else []

    def _validate(cand, kept):
        # dropping one more must leave at least prune_min active features
        return len(active) - len(kept or []) - 1 >= prune_min

    return Axis(
        name="pruning",
        propose=_propose,
        to_env=lambda selected: {"GTRADE_DROP_FEATURES": ",".join(c["drop"] for c in selected)},
        kind="additive",
        validate=_validate,
        sig=lambda c: ("drop", c["drop"]),
    )


def _try_sample_frame():
    """Best-effort: build one asset's engineered frame so proposals can be
    univariate-screened before training. Returns None on any problem (screening off)."""
    try:
        import pandas as pd
        from sqlalchemy import create_engine
        from core.features import (engineer_features, add_weekly_features,
                                   add_crossasset_features, add_macro_features,
                                   add_cross_lag_features, add_chronos_features)
        from core.track_record import _table_name
        engine = create_engine("sqlite:///" + os.path.join(BASE, "market.db"))
        table = _table_name(SELECTION_ASSETS.split(",")[0])
        df = pd.read_sql("SELECT * FROM %s" % table, engine)
        df = engineer_features(df)
        df = add_weekly_features(df, table, engine)
        df = add_crossasset_features(df, table, engine)
        df = add_macro_features(df, engine)
        df = add_cross_lag_features(df, engine)
        df = add_chronos_features(df, table, engine)
        return df
    except Exception:
        return None


def build_axes(names, base_features):
    """Map axis names to Axis objects; unknown names are skipped with a warning."""
    builders = {
        "features": lambda: make_features_axis(base_features),
        "labeling": make_labeling_axis,
        "pruning": lambda: make_pruning_axis(base_features),
        "hyper": make_hyper_axis,
        "nets": make_nets_axis,
    }
    axes = []
    for n in names:
        n = n.strip()
        if n in builders:
            axes.append(builders[n]())
        elif n:
            logger.warning("unknown auto-research axis skipped: %s", n)
    return axes


def _persist(axis_name, log):
    """Persist a per-axis log into the shared state, keyed by axis."""
    state = load_state()
    by_axis = state.get("by_axis", {})
    by_axis[axis_name] = log
    state["by_axis"] = by_axis
    state["log"] = [e for entries in by_axis.values() for e in entries]
    save_state(state)


def _gate_verdict(ok, replicated, clears):
    """Console verdict string shared by the axis and QD gates."""
    if not ok:
        return "not adoptable"
    if replicated:
        return "REPLICATED-ADOPTABLE (%d clears)" % clears
    return "ADOPTABLE (BH), 1st clear - awaiting replication"


def _winner_sig(axis_name, winner):
    """Stable cross-run signature of an axis winner, independent of temp-file env
    paths. A features winner is a list of spec dicts (name-agnostic); a pruning
    winner is a list of {'drop': f}; a labeling winner is a dict."""
    def _item(it):
        if isinstance(it, dict) and "drop" in it:
            return "drop:" + it["drop"]
        if isinstance(it, dict) and "op" in it:
            return "spec:" + json.dumps(_spec_signature(it))
        return json.dumps(it, sort_keys=True)
    if isinstance(winner, list):
        body = ",".join(sorted(_item(it) for it in winner))
    else:
        body = json.dumps(winner, sort_keys=True)
    return axis_name + ":" + body


def main():
    import argparse
    p = argparse.ArgumentParser(description="auto-research")
    p.add_argument("--regate", action="store_true",
                   help="re-gate stored candidate genomes under the current gate")
    p.add_argument("--regate-k", type=int, default=8)
    p.add_argument("--regate-screen", action="store_true",
                   help="optional cheap CB-only prefilter before the full re-gate eval")
    # parse_known_args (not parse_args): main() is called directly by the existing
    # test suite under pytest, whose own argv (test paths, -k, -o, ...) must not
    # make an un-flagged `ar.main()` call SystemExit(2). Real CLI invocations are
    # unaffected since there are no extra args to discard.
    args, _ = p.parse_known_args()
    if args.regate:
        regate(k=args.regate_k, screen=args.regate_screen)
        return
    base_features = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_z", "rsi",
                     "macd_hist", "bb_pos", "trend_strength", "atr"]
    names = os.getenv("GTRADE_AR_AXES", "features,labeling").split(",")
    if "qd" in [n.strip() for n in names]:
        run_qd()
        return
    axes = build_axes(names, base_features)
    # Only additive axes (e.g. features) consult screen_df for prescreening; skip the
    # DB read + feature pipeline entirely when no such axis is selected.
    screen_df = _try_sample_frame() if any(a.kind == "additive" for a in axes) else None
    print("[auto-research] axes: %s | budget: %d | prescreen: %s" % (
        ",".join(a.name for a in axes), BUDGET, "on" if screen_df is not None else "off"))

    base_rows = score_rows(SELECTION_ASSETS, {}, train_base_cached)  # shared base
    screen_base = train_base_cached(SELECTION_ASSETS, {"GTRADE_SCREEN_ONLY": "1"}) if _screen_on() else None
    obj = _objective()
    basis = _score_basis()
    winners = []   # (axis_name, winner, p, value, tag, neural_lift)
    ho_base_full = ho_base_contrib = None
    for axis in axes:
        try:
            prior = load_state().get("by_axis", {}).get(axis.name)
            res = run_axis(axis, BUDGET, base_rows, train_env, screen_df=screen_df,
                           prior_log=prior, persist=lambda log, a=axis.name: _persist(a, log),
                           screen_base=screen_base, screen_min=SCREEN_MIN)
            winner = res.get("kept") or res.get("best")
            if not winner:
                print("[auto-research] axis %s: nothing beat the base." % axis.name)
                continue
            winner_env = axis.to_env(winner)
            if ho_base_full is None:
                ho_base_full, ho_base_contrib = _heldout_eval(
                    HELDOUT_ASSETS, {}, train_base_cached)
            var_full, var_contrib = _heldout_eval(HELDOUT_ASSETS, winner_env, train_env)
            base_contrib = {r["Asset"]: r["Score"] for r in ho_base_contrib}
            nl, _d = _objective_delta(var_contrib, base_contrib, "mean")
            nl = round(nl, 4) if _d else None
            if basis == "neural":
                p, value, _d, tag = holdout_stats(ho_base_contrib, var_contrib, obj)
            else:
                p, value, _d, tag = holdout_stats(ho_base_full, var_full, obj)
            winners.append((axis.name, winner, p, value, tag, nl))
        except RuntimeError as exc:
            print("[auto-research] axis %s: LLM proposer unavailable, skipping (%s)"
                  % (axis.name, exc))
            continue

    flags = benjamini_hochberg([w[2] for w in winners])
    ts = datetime.utcnow().isoformat()
    finding_winners = []
    for (name, winner, p, value, tag, nl), s in zip(winners, flags):
        ok = bool(s and value > _adopt_floor(obj))
        replicated = clears = None
        if ok:
            wsig = _winner_sig(name, winner)
            replicated = ar_memory.replication_seen(wsig)
            clears = ar_memory.replication_add(wsig, ts)
            if ar_wiki.wiki_on() and clears >= 2:
                ar_wiki.note_replicated(wsig, "replicated (%d clears)" % clears)
        finding_winners.append({"axis": name, "p": p, "value": value, "tag": tag,
                                "adoptable": ok, "neural_lift": nl,
                                "replicated": bool(replicated), "clears": clears or 0})
        verdict = _gate_verdict(ok, bool(replicated), clears)
        nl_str = "" if nl is None else " | neural_lift %+.2f" % nl
        print("[auto-research] axis %s: %s | %s%s" % (name, verdict, tag, nl_str))
    ar_memory.findings_append({
        "ts": ts, "mode": "axes",
        "axes": [a.name for a in axes], "budget": BUDGET,
        "winners": finding_winners})
    if ar_wiki.wiki_on():
        ar_wiki.compile_wiki()
    mem = ar_memory.findings_summary()
    print("[auto-research] memory: %d experiments tried, %d adoptable, %d replicated so far."
          % (mem["experiments"], mem["adoptable"], mem["replicated"]))
    print("[auto-research] nothing adopted automatically; review _auto_research_log.json.")


if __name__ == "__main__":
    main()
