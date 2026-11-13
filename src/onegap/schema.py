"""Conversation record (spec 7.4) and transcript record validation. Plain dicts, no pydantic."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from onegap import CATEGORIES, CONDITIONS, LANGUAGES, OUTCOMES

SPLITS = ("train", "dev", "test")
KINDS = ("attack", "benign", "overrefusal")
FAMILIES = ("xguard", "actorattack", "generated", "xstest")


class SchemaError(ValueError):
    pass


def _req(rec: dict, key: str, typ: type | tuple[type, ...]) -> object:
    if key not in rec:
        raise SchemaError(f"{rec.get('conv_id', '?')}: missing '{key}'")
    if not isinstance(rec[key], typ):
        raise SchemaError(f"{rec.get('conv_id', '?')}: '{key}' has type {type(rec[key]).__name__}")
    return rec[key]


def validate_record(rec: dict) -> dict:
    """Validate a conversation record (input data). Returns the record unchanged."""
    _req(rec, "conv_id", str)
    _req(rec, "seed_id", str)
    if _req(rec, "split", str) not in SPLITS:
        raise SchemaError(f"{rec['conv_id']}: bad split {rec['split']!r}")
    if _req(rec, "language", str) not in LANGUAGES:
        raise SchemaError(f"{rec['conv_id']}: bad language {rec['language']!r}")
    if _req(rec, "kind", str) not in KINDS:
        raise SchemaError(f"{rec['conv_id']}: bad kind {rec['kind']!r}")
    if _req(rec, "family", str) not in FAMILIES:
        raise SchemaError(f"{rec['conv_id']}: bad family {rec['family']!r}")
    if rec["kind"] == "attack" and _req(rec, "category", str) not in CATEGORIES:
        raise SchemaError(f"{rec['conv_id']}: bad category {rec.get('category')!r}")
    turns = _req(rec, "user_turns", list)
    if not turns or not all(isinstance(t, str) and t.strip() for t in turns):
        raise SchemaError(f"{rec['conv_id']}: user_turns must be non-empty strings")
    _req(rec, "provenance", dict)
    return rec


def validate_transcript(tr: dict) -> dict:
    """Validate a transcript line written by the runner and labelled by the judges."""
    _req(tr, "conv_id", str)
    _req(tr, "seed_id", str)
    if _req(tr, "condition", str) not in CONDITIONS:
        raise SchemaError(f"{tr['conv_id']}: bad condition {tr['condition']!r}")
    turns = _req(tr, "turns", list)
    for i, t in enumerate(turns, start=1):
        if t.get("turn") != i:
            raise SchemaError(f"{tr['conv_id']}: turn numbering broken at {i}")
        for k in ("user", "assistant"):
            if not isinstance(t.get(k), str):
                raise SchemaError(f"{tr['conv_id']}: turn {i} missing '{k}'")
        if "label" in t and t["label"] not in OUTCOMES:
            raise SchemaError(f"{tr['conv_id']}: turn {i} bad label {t['label']!r}")
    return tr


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise SchemaError(f"{path}:{n}: {e}") from e


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n
