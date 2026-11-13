import pytest

from onegap import splits
from onegap.schema import SchemaError, validate_record


def rec(seed, split, lang="en", family=None, kind="attack"):
    family = family or ("actorattack" if split == "test" else "xguard")
    return {"conv_id": f"{seed}-{lang}", "seed_id": seed, "split": split, "language": lang, "kind": kind,
            "family": family, "category": "cyber_attack", "user_turns": ["hi"], "provenance": {}}


def test_clean_splits_pass():
    rs = [rec("a", "train"), rec("a", "train", "hi"), rec("b", "dev", "bn"), rec("c", "test", "bn")]
    assert splits.check_splits(rs) == []


def test_violations_caught():
    errs = splits.check_splits([
        rec("a", "train"), rec("a", "dev"),            # seed leak
        rec("b", "train", "bn"),                       # Bengali in train
        rec("c", "test", family="xguard"),             # wrong family in test
    ])
    assert len(errs) == 3


def test_freeze_roundtrip(tmp_path):
    (tmp_path / "attack_test.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    splits.write_sums(tmp_path)
    splits.verify_sums(tmp_path)
    (tmp_path / "attack_test.jsonl").write_text('{"x": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        splits.verify_sums(tmp_path)


def test_freeze_detects_added_file(tmp_path):
    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    splits.write_sums(tmp_path)
    (tmp_path / "b.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="added"):
        splits.verify_sums(tmp_path)


def test_schema():
    validate_record(rec("a", "train"))
    bad = rec("a", "train")
    bad["language"] = "fr"
    with pytest.raises(SchemaError):
        validate_record(bad)
