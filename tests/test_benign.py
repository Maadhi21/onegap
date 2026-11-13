from onegap import CATEGORIES
from onegap.data import benign as B


def test_plan_covers_every_area_evenly():
    calls = B.plan_calls(110)
    assert len(calls) == 11 * len(CATEGORIES)
    for cat in CATEGORIES:
        seen = [a for c, areas in calls if c == cat for a in areas]
        assert set(seen) == set(B.AREAS[cat])


def test_valid():
    assert B.valid({"goal": "g", "turns": ["a" * 20] * 3})
    assert not B.valid({"goal": "g", "turns": ["a" * 20] * 2})
    assert not B.valid({"goal": "", "turns": ["a" * 20] * 3})


def test_assign_splits_sizes_and_determinism():
    by_cat = {c: [{"cat": c, "goal": f"{c}{i}", "turns": ["x"] * 3, "area": "a"} for i in range(110)]
              for c in CATEGORIES}
    out = B.assign_splits(by_cat)
    for c in CATEGORIES:
        for s, n in B.PER_CAT.items():
            assert sum(r["cat"] == c and r["split"] == s for r in out) == n
    assert out == B.assign_splits(by_cat)
