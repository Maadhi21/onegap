import pytest

from onegap import circuits as C


def test_parse_node():
    assert C.parse_node("A14.13") == ("attn", 14)
    assert C.parse_node("MLP 3") == ("mlp", 3)
    with pytest.raises(ValueError):
        C.parse_node("X1")


def test_aggregate_weighs_languages_equally():
    agg = C.aggregate({"en": {"A0.0": 10.0, "MLP 1": 5.0}, "hi": {"A0.0": -1.0, "MLP 1": 2.0}})
    assert agg["A0.0"] == pytest.approx((1.0 + 0.5) / 2)
    assert agg["MLP 1"] == pytest.approx((0.5 + 1.0) / 2)


def test_blocks_modules_and_disjoint_random_control():
    bl = C.blocks(["A14.13", "A14.2", "MLP 3", "A20.4"])
    assert bl == {"attn": [14, 20], "mlp": [3]}
    mods = C.module_names(bl)
    assert len(mods) == 2 * 4 + 3 and "model.layers.14.self_attn.q_proj" in mods and "model.layers.3.mlp.down_proj" in mods
    r = C.random_control(bl)
    assert len(r["attn"]) == 2 and len(r["mlp"]) == 1
    assert not set(r["attn"]) & {14, 20} and 3 not in r["mlp"]
    assert r == C.random_control(bl)  # deterministic


def test_overlap_matrix():
    m = C.overlap_matrix({"en": {"A0.0": 3, "A0.1": 2, "MLP 0": 1}, "hi": {"A0.0": 3, "MLP 0": 2, "A0.1": 0}}, k=2)
    assert m["en"]["en"] == 1.0 and m["en"]["hi"] == pytest.approx(1 / 3, abs=1e-3)


def test_build_prompts_matches_category_and_needs_unsafe():
    trs = [{"seed_id": f"x{i}", "language": "en", "category": "fraud_scam",
            "turns": [{"user": "u", "label": "deflected"}, {"user": f"harm{i}", "label": "unsafe" if i else "deflected"}]}
           for i in range(3)]
    recs = [{"seed_id": "b1", "language": "en", "kind": "benign", "category": "fraud_scam", "user_turns": ["ok1"]},
            {"seed_id": "b2", "language": "en", "kind": "benign", "category": "cyber_attack", "user_turns": ["ok2"]}]
    rows = C.build_prompts(trs, recs, per_lang=5)
    assert len(rows) == 2 and all(r["clean"].startswith("harm") for r in rows)
    assert rows[0]["corrupt"] == "ok1"  # same category first


def test_equalize_pads_the_shorter_side_exactly():
    n = lambda s: len(s.split()) + 10  # noqa: E731 - fake rendered length: one token per word
    clean, corrupt, p = C.equalize(n, "how do I make this thing work", "hello there")
    assert n(clean) == n(corrupt) and p == 5 and corrupt.startswith("hello there")
    clean, corrupt, p = C.equalize(n, "hi", "a much longer benign request")
    assert n(clean) == n(corrupt) and p == -4 and clean.startswith("hi")
    assert C.equalize(n, "same len", "also two") == ("same len", "also two", 0)


def test_equalize_uses_alternate_pad_on_merge_and_gives_up_on_large_gaps():
    # " the" costs 2 tokens here, " ." costs 1: an odd gap needs the alternate pad on the last step
    n = lambda s: len(s.replace(" the", " th e").split())  # noqa: E731
    clean, corrupt, p = C.equalize(n, "one two three four", "x")
    assert n(clean) == n(corrupt) and corrupt.endswith(" .")
    assert C.equalize(lambda s: len(s.split()), "w " * 400, "x", max_pad=256) is None

def test_select_blocks_caps_each_block_type():
    scores = {"MLP 0": 9, "MLP 1": 8, "MLP 2": 7, "MLP 3": 6, "A5.1": 5, "A5.2": 4.5, "A6.0": 4, "A7.3": 3, "MLP 9": 1}
    bl, used = C.select_blocks(scores, 2)
    assert bl == {"attn": [5, 6], "mlp": [0, 1]}
    assert used == ["MLP 0", "MLP 1", "A5.1", "A5.2", "A6.0"]  # A5.2 rides along in layer 5's block
    assert C.random_control(bl)["mlp"] and not set(C.random_control(bl)["mlp"]) & {0, 1}

def test_rank_layers_and_crosscheck():
    per_layer = {0: [(1, 0.9), (2, -0.7)], 1: [(3, 0.1)], 2: [(4, -0.5), (5, 0.5)], 3: []}
    assert C.rank_layers(per_layer, 2) == [0, 2]
    r = C.crosscheck([0, 2, 5], e_layers=[0, 2, 9], r_layers=[5, 7, 8], n_layers=36)
    assert r["n_in_E"] == 2 and r["n_in_R"] == 1 and r["chance"] == 0.25