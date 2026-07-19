"""RL search controller for auto_research: emitter/scheduler architecture.

Spec: docs/superpowers/specs/2026-07-18-rl-search-controller-design.md
Discounted Thompson sampling over child-generation arms (two-phase context),
curiosity-based parent selection, sep-CMA-ES-lite emitter over the numeric
genes, empty-cell-targeting novelty emitter, and QC (exploration floor,
floor-draw-vs-scheduler auto-fallback, telemetry snapshot).

This module imports ONLY stdlib + core.ar_memory. Anything that requires
project knowledge (Genome fields, mutate ops, descriptor bins) is injected
by the caller as plain callables - ar_rl never imports auto_research.
"""
import copy
import math
import os
import random

from core import ar_memory  # noqa: F401

ARMS = ("feat", "hyper", "nets", "tuning", "cross", "llm", "surr", "cma", "novelty")
PHASES = ("fill", "refine")
GAMMA = 0.99
FLOOR_PER_ARM = 0.03
ADOPT_BONUS = 5
STATE_KEY = "rl_scheduler_v1"
STATE_VERSION = 1
RECENT_CAP = 200
ORIGIN_CAP = 500


def rl_on():
    return os.getenv("GTRADE_AR_RL") == "1"


def phase_of(occupancy):
    """Search phase from archive occupancy (filled cells / total cells)."""
    return "fill" if occupancy < 0.5 else "refine"


class Scheduler:
    """Discounted Thompson sampling over ARMS with a per-phase context."""

    def __init__(self, state=None, rng=None):
        self.rng = rng or random.Random()
        self.posteriors = {p: {a: [1.0, 1.0] for a in ARMS} for p in PHASES}
        if isinstance(state, dict) and state.get("version") == STATE_VERSION:
            try:
                for p in PHASES:
                    for a in ARMS:
                        pair = state["posteriors"][p][a]
                        self.posteriors[p][a] = [float(pair[0]), float(pair[1])]
            except (KeyError, TypeError, ValueError, IndexError):
                self.posteriors = {p: {a: [1.0, 1.0] for a in ARMS}
                                   for p in PHASES}

    def choose(self, available, phase):
        """Pick an arm: uniform floor draw with prob 0.03*len(available),
        else Thompson (argmax of one Beta sample per arm).
        Returns (arm, was_floor_draw)."""
        if self.rng.random() < FLOOR_PER_ARM * len(available):
            return self.rng.choice(available), True
        best, best_v = None, -1.0
        for a in available:
            al, be = self.posteriors[phase][a]
            v = self.rng.betavariate(al, be)
            if v > best_v:
                best, best_v = a, v
        return best, False

    def update(self, arm, phase, success):
        al, be = self.posteriors[phase][arm]
        # exponential forgetting of the evidence counts (keep the +1 prior)
        al = 1.0 + (al - 1.0) * GAMMA
        be = 1.0 + (be - 1.0) * GAMMA
        if success:
            al += 1.0
        else:
            be += 1.0
        self.posteriors[phase][arm] = [al, be]

    def bonus(self, arm, phase, n=ADOPT_BONUS):
        for _ in range(n):
            self.update(arm, phase, True)

    def posterior_mean(self, arm, phase):
        al, be = self.posteriors[phase][arm]
        return al / (al + be)

    def halve(self):
        """Halve evidence counts (base-model generation changed)."""
        for p in PHASES:
            for a in ARMS:
                al, be = self.posteriors[p][a]
                self.posteriors[p][a] = [1.0 + (al - 1.0) / 2.0,
                                         1.0 + (be - 1.0) / 2.0]

    def to_state(self):
        return {"version": STATE_VERSION, "posteriors": self.posteriors}


class CuriosityMap:
    """Curiosity-based parent selection (Cully et al.): elites whose children
    succeed get picked more; keyed by genome_sig."""

    START, STEP_UP, STEP_DOWN, FLOOR, TEMP = 1.0, 1.0, 0.5, 0.1, 1.0

    def __init__(self, state=None):
        self.scores = {}
        if isinstance(state, dict):
            try:
                self.scores = {str(k): float(v)
                               for k, v in state.get("scores", {}).items()}
            except (TypeError, ValueError):
                self.scores = {}

    def score(self, sig):
        return self.scores.get(sig, self.START)

    def reward(self, sig):
        self.scores[sig] = self.score(sig) + self.STEP_UP

    def penalize(self, sig):
        self.scores[sig] = max(self.FLOOR, self.score(sig) - self.STEP_DOWN)

    def pick(self, sigs, rng):
        weights = [math.exp(self.score(s) / self.TEMP) for s in sigs]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for s, w in zip(sigs, weights):
            acc += w
            if r <= acc:
                return s
        return sigs[-1]

    def prune(self, live_sigs):
        self.scores = {s: v for s, v in self.scores.items() if s in live_sigs}

    def top(self, n=5):
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def to_state(self):
        return {"scores": self.scores}


class FallbackMonitor:
    """Floor draws are uniform-random, so their rolling hit-rate is an unbiased
    live baseline. Trip when the scheduler's window underperforms it."""

    FLOOR_WIN, SCHED_WIN, MIN_FLOOR, MIN_SCHED, RATIO = 30, 50, 20, 50, 0.8

    def __init__(self, state=None):
        self.floor_hits, self.sched_hits = [], []
        if isinstance(state, dict):
            self.floor_hits = [bool(x) for x in state.get("floor", [])][-self.FLOOR_WIN:]
            self.sched_hits = [bool(x) for x in state.get("sched", [])][-self.SCHED_WIN:]

    def record(self, was_floor, hit):
        if was_floor:
            self.floor_hits = (self.floor_hits + [hit])[-self.FLOOR_WIN:]
        else:
            self.sched_hits = (self.sched_hits + [hit])[-self.SCHED_WIN:]

    def tripped(self):
        if len(self.floor_hits) < self.MIN_FLOOR or len(self.sched_hits) < self.MIN_SCHED:
            return False
        floor_rate = sum(self.floor_hits) / len(self.floor_hits)
        sched_rate = sum(self.sched_hits) / len(self.sched_hits)
        return sched_rate < self.RATIO * floor_rate

    def to_state(self):
        return {"floor": self.floor_hits, "sched": self.sched_hits}


CMA_DIMS = (
    ("thr_margin", 0.0, 0.05, False),
    ("band_delta", -0.01, 0.02, False),
    ("cb_lr_mult", 0.5, 2.0, False),
    ("cb_iter_mult", 0.7, 1.5, False),
    ("cb_depth_delta", -2, 2, True),
    ("lookback_delta", -10, 10, True),
)
_CMA_LAMBDA = 6
_CMA_MU = 3


class CmaEmitter:
    """Separable (diagonal) evolution strategy over the numeric genes.
    ask(): sample N(mean, diag(sigma^2)), clip to the palette hull, round.
    tell(): after _CMA_LAMBDA evals, rank-mu update of mean and sigma."""

    def __init__(self, state=None, rng=None, dims=None):
        self.rng = rng or random.Random()
        self.dims = tuple(dims) if dims is not None else CMA_DIMS
        spans = [hi - lo for _, lo, hi, _ in self.dims]
        self.mean = [lo + (hi - lo) / 2.0 for _, lo, hi, _ in self.dims]
        self.sigma = [0.25 * s for s in spans]
        self.evals = []
        if isinstance(state, dict):
            try:
                if (len(state["mean"]) == len(self.dims)
                        and len(state["sigma"]) == len(self.dims)):
                    self.mean = [float(v) for v in state["mean"]]
                    self.sigma = [float(v) for v in state["sigma"]]
                    self.evals = [(list(map(float, x)), float(f))
                                  for x, f in state.get("evals", [])]
            except (KeyError, TypeError, ValueError):
                pass

    def vector_of(self, genome):
        return [float(getattr(genome, name)) for name, _, _, _ in self.dims]

    def seed_from(self, genome):
        self.mean = self.vector_of(genome)

    def ask(self, parent_genome):
        child = copy.deepcopy(parent_genome)
        for i, (name, lo, hi, is_int) in enumerate(self.dims):
            v = self.rng.gauss(self.mean[i], self.sigma[i])
            v = min(hi, max(lo, v))
            v = int(round(v)) if is_int else round(v, 4)
            setattr(child, name, v)
        return child

    def tell(self, x, fit):
        self.evals.append((list(x), float(fit)))
        if len(self.evals) < _CMA_LAMBDA:
            return
        ranked = sorted(self.evals, key=lambda e: e[1], reverse=True)[:_CMA_MU]
        weights = [math.log(_CMA_MU + 0.5) - math.log(i + 1.0)
                   for i in range(_CMA_MU)]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        new_mean = [sum(w * e[0][j] for w, e in zip(weights, ranked))
                    for j in range(len(self.dims))]
        for j, (_, lo, hi, _) in enumerate(self.dims):
            span = hi - lo
            var = sum(w * (e[0][j] - new_mean[j]) ** 2
                      for w, e in zip(weights, ranked))
            self.sigma[j] = min(0.5 * span,
                                max(0.01 * span, math.sqrt(var) * 1.1))
            self.mean[j] = min(hi, max(lo, new_mean[j]))
        self.evals = []

    def to_state(self):
        return {"mean": self.mean, "sigma": self.sigma, "evals": self.evals,
                "dims_n": len(self.dims)}


class NoveltyEmitter:
    """Target the genome-derived projection (count_bin, gene_group) of the
    descriptor: pick an unoccupied projection cell, take the elite nearest in
    count (ties: higher fitness) as parent, ask the caller to mutate toward
    the target. Archive keys are "floor_count_group" strings."""

    ATTEMPTS = 5

    def __init__(self, count_bins, groups, rng=None):
        self.count_bins = count_bins
        self.groups = groups
        self.rng = rng or random.Random()

    def emit(self, archive_keys, elites, count_of, group_of, count_bin_of,
             mutate_toward):
        occupied = set()
        for k in archive_keys:
            parts = k.split("_")
            if len(parts) == 3:
                occupied.add((int(parts[1]), int(parts[2])))
        empty = [(cb, gr) for cb in range(self.count_bins)
                 for gr in range(self.groups) if (cb, gr) not in occupied]
        if not empty or not elites:
            return None
        target_bin, target_group = self.rng.choice(empty)
        parent = min(
            elites,
            key=lambda e: (abs(count_bin_of(count_of(e["genome"])) - target_bin),
                           -e["fitness"]))["genome"]
        for _ in range(self.ATTEMPTS):
            child = mutate_toward(parent, target_bin, target_group)
            if child is not None:
                return child
        return None
