"""A learned surrogate for the MAP-Elites QD search: featurize a genome, fit a small
regressor on the archive's (genome, observed fitness) pairs, and predict a candidate's
fitness so mutation can be biased toward predicted-good children. Pure and testable;
sklearn is already a dependency. Everything here is opt-in (GTRADE_AR_SURROGATE) and
returns safe defaults so the QD loop never depends on it succeeding.
"""

import os

_DSL_OPS = ["zscore", "ratio", "lag", "diff", "rolling", "interaction", "lead_lag"]


def surrogate_on():
    """GTRADE_AR_SURROGATE: bias QD mutation with the learned surrogate (default OFF)."""
    return (os.getenv("GTRADE_AR_SURROGATE") or "").strip() in ("1", "true", "True")


def n_candidates():
    """GTRADE_AR_SURROGATE_K: how many candidate children to score per step (default 6)."""
    try:
        return max(1, int(os.getenv("GTRADE_AR_SURROGATE_K", "6")))
    except ValueError:
        return 6


def genome_vector(genome, active, base_features):
    """Fixed-length numeric featurization of a Genome, deterministic given `active`:
    [n_drops, <one flag per active feature = 1 if dropped>, n_extra,
     <count per DSL op>, is_direction, is_rel_median, label_window].

    base_features: reserved for a future DSL-input featurization of the extra specs;
    unused today (the op-histogram already captures the extra specs coarsely)."""
    active = list(active)
    drop_set = set(genome.drops)
    vec = [float(len(genome.drops))]
    vec += [1.0 if f in drop_set else 0.0 for f in active]
    vec.append(float(len(genome.extra)))
    counts = {op: 0 for op in _DSL_OPS}
    for s in genome.extra:
        op = s.get("op")
        if op in counts:
            counts[op] += 1
    vec += [float(counts[op]) for op in _DSL_OPS]
    vec.append(1.0 if genome.label_mode == "direction" else 0.0)
    vec.append(1.0 if genome.label_mode == "rel_median" else 0.0)
    vec.append(float(genome.label_window))
    return vec


def fit_surrogate(samples, min_samples=8):
    """Fit a small regressor on [(vector, fitness), ...]. None when there are fewer
    than `min_samples` or the fitnesses are all equal (nothing to learn) or sklearn
    errors - callers then skip the surrogate."""
    if len(samples) < min_samples:
        return None
    y = [f for _, f in samples]
    if len(set(y)) < 2:
        return None
    x = [v for v, _ in samples]
    try:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=0)
        model.fit(x, y)
        return model
    except Exception:
        return None


def predict(model, vector):
    """Predicted fitness for a genome vector; 0.0 when there is no model or on error."""
    if model is None:
        return 0.0
    try:
        return float(model.predict([vector])[0])
    except Exception:
        return 0.0
