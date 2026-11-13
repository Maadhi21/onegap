"""Wilson intervals, paired and cluster bootstrap over seed_id, exact McNemar (spec 8.3)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class Rate:
    k: int
    n: int
    lo: float
    hi: float

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def fmt(self, pct: bool = True) -> str:
        if not self.n:
            return "n/a (n=0)"
        s = 100 if pct else 1
        return f"{self.p * s:.1f} [{self.lo * s:.1f}, {self.hi * s:.1f}] ({self.k}/{self.n})"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Rate:
    """Wilson score interval for k successes out of n."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid k={k}, n={n}")
    if n == 0:
        return Rate(0, 0, float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return Rate(k, n, lo, hi)


def _paired(a: Mapping[str, bool], b: Mapping[str, bool]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no shared seed_id between the two conditions")
    return (np.array([bool(a[k]) for k in keys], dtype=float),
            np.array([bool(b[k]) for k in keys], dtype=float))


@dataclass(frozen=True)
class Diff:
    n: int
    diff: float  # mean(b) - mean(a)
    lo: float
    hi: float

    def fmt(self) -> str:
        return f"{self.diff * 100:+.1f} pts [{self.lo * 100:+.1f}, {self.hi * 100:+.1f}] (n={self.n} paired)"


def paired_bootstrap(a: Mapping[str, bool], b: Mapping[str, bool],
                     n_resamples: int = 10_000, seed: int = 13, alpha: float = 0.05) -> Diff:
    """Percentile CI for mean(b) - mean(a), resampling seed_ids with replacement."""
    xa, xb = _paired(a, b)
    d = xb - xa
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Diff(len(d), float(d.mean()), float(lo), float(hi))


def cluster_bootstrap(a: Mapping[tuple[str, str], bool], b: Mapping[tuple[str, str], bool],
                      n_resamples: int = 10_000, seed: int = 13, alpha: float = 0.05) -> Diff:
    """Percentile CI for mean(b) - mean(a) over shared (language, seed_id) items, resampling seed_ids.

    A behaviour asked in three languages is one cluster, not three independent pairs. With one language this is
    the same as `paired_bootstrap`. Each resample is a ratio (sum of differences / number of items), because
    clusters can differ in size when a language failed the fidelity gate for a seed.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no shared (language, seed_id) between the two conditions")
    seeds = sorted({k[1] for k in keys})
    pos = {s: i for i, s in enumerate(seeds)}
    d_sum, n = np.zeros(len(seeds)), np.zeros(len(seeds))
    for k in keys:
        d_sum[pos[k[1]]] += float(bool(b[k])) - float(bool(a[k]))
        n[pos[k[1]]] += 1
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)))
    boots = d_sum[pick].sum(axis=1) / n[pick].sum(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Diff(len(keys), float(d_sum.sum() / n.sum()), float(lo), float(hi))


@dataclass(frozen=True)
class McNemar:
    n: int
    b01: int  # a False, b True
    b10: int  # a True, b False
    p: float

    def fmt(self) -> str:
        return f"McNemar exact p={self.p:.4g} (discordant {self.b10}->0, {self.b01}->1; n={self.n})"


def mcnemar(a: Mapping[str, bool], b: Mapping[str, bool]) -> McNemar:
    """Exact (binomial) McNemar test on paired per-conversation outcomes."""
    xa, xb = _paired(a, b)
    b01 = int(((xa == 0) & (xb == 1)).sum())
    b10 = int(((xa == 1) & (xb == 0)).sum())
    m = b01 + b10
    p = 1.0 if m == 0 else float(binomtest(b01, m, 0.5, alternative="two-sided").pvalue)
    return McNemar(len(xa), b01, b10, p)


def cohen_kappa(x: list, y: list) -> float:
    """Cohen's kappa for two raters over the same items."""
    if len(x) != len(y) or not x:
        raise ValueError("need two equal-length, non-empty label lists")
    labels = sorted(set(x) | set(y))
    n = len(x)
    po = sum(i == j for i, j in zip(x, y, strict=True)) / n
    pe = sum((x.count(lab) / n) * (y.count(lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
