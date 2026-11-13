"""Invariant 7: harmful model output is redacted, never gated.

An unsafe assistant turn becomes
  [REDACTED unsafe reply | sha256=<hash> | judge labels | category | turn]
The hash is of the original UTF-8 text, so `make replay` can regenerate and check it. The English view the judge
saw (`assistant_en`, for Hindi, Hinglish and Bengali replies) is the same content in translation: it is removed
from a redacted turn, and its presence on an unsafe turn counts as a violation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from onegap.schema import read_jsonl

PREFIX = "[REDACTED unsafe reply"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redaction_marker(text: str, judge_labels: dict | str, category: str, turn: int | str) -> str:
    if isinstance(judge_labels, dict):
        judge_labels = ",".join(f"{k}={v}" for k, v in sorted(judge_labels.items()))
    return f"{PREFIX} | sha256={sha256(text)} | {judge_labels} | {category} | turn {turn}]"


def is_redacted(text: str) -> bool:
    return text.startswith(PREFIX)


def redact_transcript(tr: dict) -> dict:
    """Return a copy with every unsafe assistant turn replaced by its marker."""
    out = dict(tr)
    turns = []
    for t in tr["turns"]:
        t = dict(t)
        if t.get("label") == "unsafe" and not is_redacted(t["assistant"]):
            t["assistant"] = redaction_marker(t["assistant"], t.get("judges", "unsafe"),
                                              tr.get("category", "?"), t["turn"])
        if t.get("label") == "unsafe":
            t.pop("assistant_en", None)  # the judge's English view of an unsafe reply is the same content
        turns.append(t)
    out["turns"] = turns
    return out


def redact_pair(pair: dict) -> dict:
    """DPO pair: `rejected` of a harmful pair is redacted the same way."""
    out = dict(pair)
    if pair.get("kind") == "harmful" and not is_redacted(pair["rejected"]):
        out["rejected"] = redaction_marker(pair["rejected"], pair.get("judges", "unsafe"),
                                           pair.get("category", "?"), pair.get("turn", "?"))
    return out


def violations(rows: Iterable[dict]) -> list[str]:
    """Every place where unredacted harmful text would be published."""
    bad = []
    for r in rows:
        rid = r.get("conv_id") or r.get("pair_id") or "?"
        for t in r.get("turns", []):
            if t.get("label") == "unsafe" and not is_redacted(t.get("assistant", "")):
                bad.append(f"{rid} turn {t.get('turn')}: unsafe assistant turn not redacted")
            if t.get("label") == "unsafe" and t.get("assistant_en"):
                bad.append(f"{rid} turn {t.get('turn')}: English view of an unsafe turn not removed")
        if r.get("kind") == "harmful" and "rejected" in r and not is_redacted(r["rejected"]):
            bad.append(f"{rid}: harmful pair 'rejected' not redacted")
    return bad


def check_paths(paths: Iterable[str | Path]) -> list[str]:
    bad = []
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            bad += [f"{f}: {v}" for v in violations(read_jsonl(f))]
    return bad
