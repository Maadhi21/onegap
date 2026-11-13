import pytest

from onegap import figures as F


def tr(seed, lang, labels):
    return {"seed_id": seed, "language": lang, "kind": "attack",
            "turns": [{"turn": i, "label": lab} for i, lab in enumerate(labels, start=1)]}


def test_by_seed_and_language_gaps_are_paired_on_seed():
    trs = [tr(f"s{i}", "en", ["refused"]) for i in range(10)]
    trs += [tr(f"s{i}", "hi", ["deflected"]) for i in range(10)]
    trs += [tr("only-hi", "hi", ["refused"])]  # no English partner: not paired
    assert F.by_seed(trs, "en", "refused") == {f"s{i}": True for i in range(10)}
    g = F.language_gaps(trs, "refused", langs=("hi",))
    assert g["hi"]["n"] == 10 and g["hi"]["diff"] == -1.0 and g["hi"]["p"] < 0.01


def _rate(k, n):
    return {"k": k, "n": n, "p": k / n, "lo": max(0, k / n - 0.1), "hi": min(1, k / n + 0.1), "text": ""}


@pytest.fixture
def reports():
    langs = ["en", "hi", "hinglish", "bn", "pooled"]
    conds = ["S0", "A", "B", "D", "E", "R"]
    dev = {"asr": {c: {x: _rate(10, 40) for x in langs} for c in conds},
           "overrefusal": {c: {x: _rate(2, 50) for x in langs} for c in conds},
           "first_failure": {c: {x: {"2": 1, "3": 5, "4": 5, "held": 29} for x in langs} for c in conds},
           "edit_size": {c: {"trainable_params": p, "model_params": 3_349_227_520}
                         for c, p in (("D", 30_228_480), ("E", 7_557_120), ("R", 7_557_120))}}
    test = {"asr": {"S0": {x: _rate(3, 100) for x in langs}},
            "final_turn_refused": {"S0": {x: _rate(14, 100) for x in langs}}}
    return dev, test


def test_video_numbers_cover_every_value_with_counts(reports):
    dev, test = reports
    n = F.video_numbers(dev, test, faithfulness={"en": 0.6908})
    assert n["dev_a_en"] == "25%" and n["dev_a_en__n"] == "10/40"
    assert n["test_refused_en"] == "14%" and n["n_test_attack"] == "400"
    assert n["modal_fail_turn"] == "3"  # tie 3 vs 4 goes to the earlier turn
    assert n["params_E"] == "7.6M" and n["share_E"] == "0.23%" and n["faith_en"] == "0.69"


def test_all_figures_render(reports, tmp_path):
    pytest.importorskip("matplotlib")
    dev, test = reports
    paths = F.save_all(dev, test, tmp_path)
    assert len(paths) == len(F.FIGURES) and all(p.stat().st_size > 5000 for p in paths)
