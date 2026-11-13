import math

import pytest

from onegap.stats import cohen_kappa, mcnemar, paired_bootstrap, wilson


def test_wilson_known_value():
    # 14.3% of 119 (public HarmBench anchor): 17/119
    r = wilson(17, 119)
    assert r.p == pytest.approx(0.1429, abs=1e-4)
    assert r.lo == pytest.approx(0.0912, abs=1e-3)
    assert r.hi == pytest.approx(0.2166, abs=1e-3)


def test_wilson_edges():
    assert wilson(0, 10).lo == 0.0
    assert wilson(10, 10).hi == 1.0
    assert math.isnan(wilson(0, 0).p)
    with pytest.raises(ValueError):
        wilson(5, 3)


def test_paired_bootstrap_direction_and_ci():
    a = {f"s{i}": True for i in range(50)}
    b = {f"s{i}": i < 10 for i in range(50)}
    d = paired_bootstrap(a, b, n_resamples=2000)
    assert d.n == 50
    assert d.diff == pytest.approx(-0.8)
    assert d.lo < d.diff < d.hi < 0


def test_paired_bootstrap_requires_overlap():
    with pytest.raises(ValueError):
        paired_bootstrap({"a": True}, {"b": True})


def test_mcnemar():
    a = {f"s{i}": True for i in range(30)}
    b = {f"s{i}": i < 5 for i in range(30)}
    m = mcnemar(a, b)
    assert (m.b10, m.b01) == (25, 0)
    assert m.p < 1e-6
    assert mcnemar(a, a).p == 1.0


def test_kappa():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("aabb")) == pytest.approx(0.0)
