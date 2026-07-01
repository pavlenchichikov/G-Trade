"""Auto-research loop (local). The agent proposes feature variants
from the DSL, the harness A/B-tests them base vs variant on a selection subset,
and winners are checked on a held-out subset and flagged for a human. It never
retrains production. 

Proposer is autonomous (evolutionary search, no LLM) by default. Set
GTRADE_AR_PROPOSER=llm to use the Claude proposer instead (needs the anthropic SDK
and ANTHROPIC_API_KEY). GTRADE_AR_SEED makes the evolutionary search reproducible.

Run:  python auto_research.py
      GTRADE_AR_PROPOSER=llm python auto_research.py
"""
import bisect
import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field

from core.feature_dsl import validate_spec
from core.logger import get_logger

logger = get_logger("auto_research")

BASE = os.path.dirname(os.path.abspath(__file__))
SELECTION_ASSETS = "SP500,NVDA,BTC,ETH,EURUSD,GBPJPY,GAS,AAPL,SBER,DAX"
HELDOUT_ASSETS = "MSFT,GOLD,USDJPY,ADA,CAC40,XOM"
BUDGET = int(os.getenv("AR_BUDGET", "15"))
ADOPT_MEAN_SCORE_DELTA = 0.5

LIGHT_ENV = {
    "GTRADE_WORKERS": "4", "GTRADE_NEURAL_SLOTS": "4", "GTRADE_TF_THREADS": "2",
    "GTRADE_CB_THREADS": "1", "GTRADE_MAX_FOLDS": "5", "GTRADE_ADAPTIVE_NETS": "1",
    "GTRADE_NET_CAP": "80", "GTRADE_EPOCHS_LSTM": "90", "GTRADE_EPOCHS_TF": "60",
    "GTRADE_EPOCHS_TCN": "50", "GTRADE_FORCE_PROMOTE": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
}


def _objective_delta(var_rows, base_score, objective="mean"):
    """Paired (variant minus base) Score deltas over shared assets, reduced by the
    objective: 'mean' = average delta, 'min' = the worst asset's delta (lift-the-floor).
    Returns (value, deltas)."""
    e = {r["Asset"]: r.get("Score", 0.0) for r in var_rows}
    common = sorted(set(e) & set(base_score))
    if not common:
        return 0.0, []
    deltas = [e[a] - base_score[a] for a in common]
    value = min(deltas) if objective == "min" else sum(deltas) / len(deltas)
    return value, deltas


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


def _objective():
    """The search/gate objective: 'mean' (default) or 'min' (lift-the-floor)."""
    o = (os.getenv("GTRADE_AR_OBJECTIVE") or "mean").strip().lower()
    if o not in ("mean", "min"):
        logger.warning("unknown GTRADE_AR_OBJECTIVE %r, using mean", o)
        return "mean"
    return o


def holdout_stats(base_rows, ext_rows, objective="mean"):
    """Raw held-out stats for a variant: (sign-test p, objective value, deltas, tag).
    No adoption decision - main applies BH across the axis-winners."""
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in base_rows}
    value, deltas = _objective_delta(ext_rows, base_score, objective)
    if not deltas:
        return 1.0, 0.0, [], "no common held-out assets"
    p = _sign_test_p(deltas)
    up = sum(1 for d in deltas if d > 0)
    tag = "%s dScore %.2f, sign-test p=%.3f (%d/%d up)" % (objective, value, p, up, len(deltas))
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
    if p < alpha and value > ADOPT_MEAN_SCORE_DELTA:
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
    quality_report rows ([] on failure). The generic primitive every axis uses."""
    tmp = tempfile.mkdtemp(prefix="ar_")
    return _train(subset, dict(env_overrides), os.path.join(tmp, "run"))


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


@dataclass
class Genome:
    """A composable cross-axis experiment: feature drops + extra DSL specs + labeling.
    The objective is NOT here (it is the QD fitness). Empty == production default."""
    drops: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    label_mode: str = "direction"
    label_window: int = 30


def genome_to_env(g):
    """Compose the genome into one training env-override dict (empty -> no overrides)."""
    env = dict(_feature_env(g.extra, [s["name"] for s in g.extra]))
    if g.drops:
        env["GTRADE_DROP_FEATURES"] = ",".join(g.drops)
    if g.label_mode != "direction":
        env["GTRADE_LABEL_MODE"] = g.label_mode
        env["GTRADE_LABEL_WINDOW"] = str(g.label_window)
    return env


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
    if g.label_mode not in ("direction", "rel_median"):
        return False
    if g.label_window <= 0:
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
    if random.random() < 0.5:
        g.label_mode = "rel_median"
        g.label_window = random.choice([20, 30, 60])
    return g if valid(g, active, prune_min) else Genome()


def mutate(g, active, base_features):
    """One random gene change; always returns a valid genome (or the input if no valid
    single change exists)."""
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    ops = ["add_drop", "rm_drop", "add_extra", "rm_extra", "flip_label", "win"]
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
            ng.label_mode = "rel_median" if ng.label_mode == "direction" else "direction"
        elif op == "win":
            ng.label_window = random.choice([w for w in (20, 30, 60) if w != ng.label_window] or [30])
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


def run_qd(train_fn=None):
    """MAP-Elites: illuminate an archive of diverse genomes via the cheap CB screen,
    then full-evaluate + honest-gate the top elites. Returns the archive."""
    from core.features import active_candidate_features

    train_fn = train_fn or train_env
    base_features = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_z", "rsi",
                     "macd_hist", "bb_pos", "trend_strength", "atr"]
    active = active_candidate_features()
    init = int(os.getenv("GTRADE_AR_QD_INIT", "8"))
    n_final = int(os.getenv("GTRADE_AR_QD_FINAL", "3"))

    screen_base = train_fn(SELECTION_ASSETS, {"GTRADE_SCREEN_ONLY": "1"})
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in screen_base}

    def _canon(g):
        if g.label_mode == "direction":
            g.label_window = 30
        return g

    def _screen_eval(g):
        return train_fn(SELECTION_ASSETS, screen_env(genome_to_env(g)))

    archive = _qd_load()
    if not archive:
        for _ in range(init):
            g = _canon(random_genome(active, base_features))
            archive_put(archive, g, _screen_eval(g), base_score, active)
        _qd_save(archive)

    for _ in range(BUDGET):
        elites = list(archive.values())
        if not elites:
            break
        parent = random.choice(elites)["genome"]
        if len(elites) >= 2 and random.random() < 0.5:
            child = crossover(parent, random.choice(elites)["genome"], active)
        else:
            child = mutate(parent, active, base_features)
        child = _canon(child)
        archive_put(archive, child, _screen_eval(child), base_score, active)
        _qd_save(archive)

    elites = sorted(archive.values(), key=lambda e: e["fitness"], reverse=True)[:n_final]
    if not elites:
        print("[qd] no elites in the archive.")
        return archive
    ho_base = train_fn(HELDOUT_ASSETS, {})
    obj = _objective()
    results = []
    for e in elites:
        g = e["genome"]
        ho_var = train_fn(HELDOUT_ASSETS, genome_to_env(g))
        p, value, _d, tag = holdout_stats(ho_base, ho_var, obj)
        results.append((g, p, value, tag))
    sig = benjamini_hochberg([r[1] for r in results])
    for (g, p, value, tag), s in zip(results, sig):
        ok = s and value > ADOPT_MEAN_SCORE_DELTA
        print("[qd] elite drops=%s label=%s/%d extra=%d: %s | %s" % (
            g.drops, g.label_mode, g.label_window, len(g.extra),
            "ADOPTABLE (BH)" if ok else "not adoptable", tag))
    print("[qd] %d niches illuminated; review _qd_archive.json; nothing auto-adopted." % len(archive))
    return archive


DSL_MENU = (
    "ops: zscore(window 2-200), ratio(a,b), lag(k 1-20), diff(k 1-20), "
    "rolling(window,agg in mean|std|sum), interaction(a,b), lead_lag(leader in "
    "sp500|vix|btc|gold|dxy|tnx, horizon 1-20). Each spec: "
    '{"name": lower_snake, "op": ..., "inputs": [...], "params": {...}}.'
)


def _proposer_prompt(log, base_features):
    """The shared prompt for any LLM provider."""
    history = json.dumps(log[-8:], ensure_ascii=True)
    return (
        "You are proposing engineered features for a trading model to revive weak "
        "neural members. Use ONLY this DSL.\n" + DSL_MENU +
        "\nBase columns you can reference: " + ",".join(base_features) +
        "\nPast experiments (spec + held-back selection Score deltas):\n" + history +
        "\nReturn STRICT JSON: a list of 1-2 new spec dicts, no prose."
    )


def _parse_specs(text):
    """Extract the JSON list of specs from a model reply, tolerant of stray prose."""
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        specs = json.loads(text[start:end + 1])
    except Exception:
        return []
    return specs if isinstance(specs, list) else []


def _call_anthropic(prompt):
    """Anthropic SDK. Model via GTRADE_AR_LLM_MODEL (default claude-opus-4-8)."""
    import anthropic
    client = anthropic.Anthropic()
    model = os.getenv("GTRADE_AR_LLM_MODEL") or "claude-opus-4-8"
    last_err = None
    for _attempt in range(3):
        try:
            msg = client.messages.create(
                model=model, max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text.strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError("anthropic proposer failed after 3 attempts: %s" % last_err)


def _call_openai(prompt):
    """OpenAI-compatible chat API. Works with OpenAI and any compatible endpoint
    (Mistral, a local Ollama/LM Studio, etc.) via GTRADE_AR_LLM_BASE_URL. Model via
    GTRADE_AR_LLM_MODEL (default gpt-4o)."""
    import openai
    client = openai.OpenAI(base_url=os.getenv("GTRADE_AR_LLM_BASE_URL") or None)
    model = os.getenv("GTRADE_AR_LLM_MODEL") or "gpt-4o"
    last_err = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError("openai proposer failed after 3 attempts: %s" % last_err)


_LLM_BACKENDS = {"anthropic": _call_anthropic, "openai": _call_openai}


def propose_next(log, base_features):
    """Ask an LLM for the next 1-2 feature specs. Provider via GTRADE_AR_LLM:
    'anthropic' (default) or 'openai' (also any OpenAI-compatible endpoint via
    GTRADE_AR_LLM_BASE_URL). The chosen backend is retried a few times, then the
    loop stops cleanly; a non-JSON reply yields no specs (that iteration is skipped)."""
    provider = (os.getenv("GTRADE_AR_LLM") or "anthropic").strip().lower()
    backend = _LLM_BACKENDS.get(provider)
    if backend is None:
        raise RuntimeError("unknown GTRADE_AR_LLM %r (use anthropic or openai)" % provider)
    return _parse_specs(backend(_proposer_prompt(log, base_features)))


_SAMPLE_WINDOWS = [5, 10, 20, 50]
_SAMPLE_KS = [1, 2, 3, 5, 10]
_SAMPLE_HORIZONS = [1, 2, 5]
_SINGLE_OPS = ["zscore", "lag", "diff", "rolling"]
_PAIR_OPS = ["ratio", "interaction"]
_AGGS = ["mean", "std", "sum"]
_LEAD_LEADERS = ["sp500", "vix", "btc", "gold", "dxy", "tnx"]


def _spec_signature(spec):
    """Identity of a spec ignoring its name, used to dedup against the log."""
    return (spec["op"], tuple(spec.get("inputs") or []),
            tuple(sorted((spec.get("params") or {}).items())))


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


def propose_evolutionary(log, base_features):
    """Autonomous (no LLM) proposer. Explores the DSL early; once a past spec shows
    a positive mean selection delta it biases input choice toward the good inputs and
    mutates the best spec. Reads the log, dedups against it, returns one valid spec."""
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
        if validate_spec(spec, cols) and _spec_signature(spec) not in seen:
            return [spec]
    return []


def _select_proposer():
    """Evolutionary (no LLM) by default; the Claude proposer when GTRADE_AR_PROPOSER=llm."""
    if (os.getenv("GTRADE_AR_PROPOSER") or "evolutionary").strip().lower() == "llm":
        return propose_next
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
    `validate(cand, selected)` and `prescreen(cand, screen_df)` default to accepting."""
    name: str
    propose: object
    to_env: object
    kind: str = "additive"
    validate: object = None
    prescreen: object = None

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
    called after each iteration (None = no-op)."""
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
        for i in range(len(axis_log), budget):
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
                    persist(log)
                    continue
            rows = train_fn(SELECTION_ASSETS, axis.to_env(cand))
            delta, _ = _objective_delta(rows, base_score, objective)
            entry = {"axis": axis.name, "iter": i, "cand": new,
                     "cand_mean_delta": delta, "score": delta - kept_delta}
            if delta - kept_delta > 1e-9:
                kept, kept_delta = cand, delta
                entry["accepted"] = True
            log.append(entry)
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
    for cand in proposed:
        if i >= budget:
            break
        if json.dumps(cand, sort_keys=True) in tried:
            continue
        if screen_base is not None:
            passed, sdelta = _passes_screen(axis, cand, train_fn, screen_base, screen_min, objective)
            if not passed:
                log.append({"axis": axis.name, "iter": i, "cand": cand,
                            "screen_delta": sdelta, "note": "screened out"})
                persist(log)
                i += 1
                continue
        rows = train_fn(SELECTION_ASSETS, axis.to_env(cand))
        delta, _ = _objective_delta(rows, base_score, objective)
        entry = {"axis": axis.name, "iter": i, "cand": cand, "cand_mean_delta": delta}
        if delta > best_delta + 1e-9:
            best, best_delta = cand, delta
            entry["accepted"] = True
        log.append(entry)
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
        propose=lambda log: proposer(log, base_features),
        to_env=lambda selected: _feature_env(selected, [s["name"] for s in selected]),
        kind="additive",
        validate=_validate,
        prescreen=lambda cand, screen_df: True if screen_df is None else _prescreen_ok(cand, screen_df),
    )


LABEL_WINDOWS = (20, 30, 60)


def make_labeling_axis():
    """Sweep the rel_median label window (direction is the base). select_best: keep
    the single best window whose mean Score delta beats the base."""
    def _propose(log):
        tried = {e["cand"]["window"] for e in log
                 if isinstance(e.get("cand"), dict) and "window" in e["cand"]}
        return [{"mode": "rel_median", "window": w} for w in LABEL_WINDOWS if w not in tried]

    return Axis(
        name="labeling",
        propose=_propose,
        to_env=lambda cand: {"GTRADE_LABEL_MODE": cand["mode"],
                             "GTRADE_LABEL_WINDOW": str(cand["window"])},
        kind="select_best",
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
        remaining = [f for f in active if f not in tried]
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
    )


def _try_sample_frame():
    """Best-effort: build one asset's engineered frame so proposals can be
    univariate-screened before training. Returns None on any problem (screening off)."""
    try:
        import pandas as pd
        from sqlalchemy import create_engine
        from core.features import (engineer_features, add_weekly_features,
                                   add_crossasset_features, add_macro_features,
                                   add_cross_lag_features)
        from core.track_record import _table_name
        engine = create_engine("sqlite:///" + os.path.join(BASE, "market.db"))
        table = _table_name(SELECTION_ASSETS.split(",")[0])
        df = pd.read_sql("SELECT * FROM %s" % table, engine)
        df = engineer_features(df)
        df = add_weekly_features(df, table, engine)
        df = add_crossasset_features(df, table, engine)
        df = add_macro_features(df, engine)
        df = add_cross_lag_features(df, engine)
        return df
    except Exception:
        return None


def build_axes(names, base_features):
    """Map axis names to Axis objects; unknown names are skipped with a warning."""
    builders = {
        "features": lambda: make_features_axis(base_features),
        "labeling": make_labeling_axis,
        "pruning": lambda: make_pruning_axis(base_features),
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


def main():
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

    base_rows = train_env(SELECTION_ASSETS, {})  # shared base, trained once
    screen_base = train_env(SELECTION_ASSETS, {"GTRADE_SCREEN_ONLY": "1"}) if _screen_on() else None
    obj = _objective()
    winners = []   # (axis_name, p, value, tag)
    ho_base = None
    for axis in axes:
        prior = load_state().get("by_axis", {}).get(axis.name)
        res = run_axis(axis, BUDGET, base_rows, train_env, screen_df=screen_df,
                       prior_log=prior, persist=lambda log, a=axis.name: _persist(a, log),
                       screen_base=screen_base, screen_min=SCREEN_MIN)
        winner = res.get("kept") or res.get("best")
        if not winner:
            print("[auto-research] axis %s: nothing beat the base." % axis.name)
            continue
        if ho_base is None:
            ho_base = train_env(HELDOUT_ASSETS, {})
        ho_var = train_env(HELDOUT_ASSETS, axis.to_env(winner))
        p, value, _deltas, tag = holdout_stats(ho_base, ho_var, obj)
        winners.append((axis.name, p, value, tag))

    sig = benjamini_hochberg([w[1] for w in winners])
    for (name, p, value, tag), s in zip(winners, sig):
        ok = s and value > ADOPT_MEAN_SCORE_DELTA
        print("[auto-research] axis %s: %s | %s" % (
            name, "ADOPTABLE (BH)" if ok else "not adoptable", tag))
    print("[auto-research] nothing adopted automatically; review _auto_research_log.json.")


if __name__ == "__main__":
    main()
