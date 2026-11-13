"""Invariant 7 gate. Also scans any publish staging dir if ONEGAP_PUBLISH_DIR is set."""

import os
from pathlib import Path

import pytest

from onegap.redact import check_paths, is_redacted, redact_pair, redact_transcript, sha256, violations


def test_unsafe_turn_is_redacted_with_hash(tr):
    t = tr("s1", ["refused", "unsafe"])
    original = t["turns"][1]["assistant"]
    out = redact_transcript(t)
    a = out["turns"][1]["assistant"]
    assert is_redacted(a)
    assert f"sha256={sha256(original)}" in a
    assert "fraud_scam" in a and "turn 2" in a
    assert out["turns"][0]["assistant"] == "reply 1"  # safe turns untouched
    assert t["turns"][1]["assistant"] == original  # input not mutated
    assert violations([out]) == []


def test_unredacted_is_flagged(tr):
    assert violations([tr("s1", ["unsafe"])])


def test_pairs():
    harmful = {"pair_id": "p1", "kind": "harmful", "rejected": "bad", "chosen": "no", "category": "x"}
    benign = {"pair_id": "p2", "kind": "benign", "rejected": "I can't help", "chosen": "sure"}
    assert violations([harmful])
    assert violations([redact_pair(harmful), redact_pair(benign)]) == []
    assert redact_pair(benign)["rejected"] == "I can't help"


def test_idempotent(tr):
    once = redact_transcript(tr("s1", ["unsafe"]))
    assert redact_transcript(once) == once


@pytest.mark.skipif(not os.environ.get("ONEGAP_PUBLISH_DIR"), reason="no publish staging dir")
def test_publish_dir_clean():
    assert check_paths([Path(os.environ["ONEGAP_PUBLISH_DIR"])]) == []
