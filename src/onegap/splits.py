"""Split checks, freeze and hash verification (spec 7.3, Invariants 1-2)."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

FROZEN = Path("data/frozen")
SUMS = "SHA256SUMS"


def check_splits(records: Iterable[dict]) -> list[str]:
    """Return a list of invariant violations (empty list = OK)."""
    errors = []
    seed_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        seed_splits[r["seed_id"]].add(r["split"])
        if r["split"] == "train" and r["language"] == "bn":
            errors.append(f"{r['conv_id']}: Bengali in train")
        if r["kind"] == "attack":
            if r["split"] == "test" and r["family"] != "actorattack":
                errors.append(f"{r['conv_id']}: test attack not from Attack_600")
            if r["split"] in ("train", "dev") and r["family"] != "xguard":
                errors.append(f"{r['conv_id']}: train/dev attack not from XGuard-Train")
    for seed, splits in seed_splits.items():
        if len(splits) > 1:
            errors.append(f"seed {seed} appears in splits {sorted(splits)}")
    return errors


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sums(directory: Path = FROZEN) -> Path:
    files = sorted(p for p in directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl files in {directory}")
    out = directory / SUMS
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for p in files:
            f.write(f"{_file_hash(p)}  {p.name}\n")
    return out


def verify_sums(directory: Path = FROZEN) -> str:
    """Raise if any frozen file changed, is missing, or was added. Returns a digest of SHA256SUMS."""
    sums = directory / SUMS
    if not sums.exists():
        raise FileNotFoundError(f"{sums} missing — test set not frozen yet")
    expected = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        expected[name.strip()] = digest
    actual = {p.name for p in directory.glob("*.jsonl")}
    if extra := actual - set(expected):
        raise RuntimeError(f"files added after freeze: {sorted(extra)}")
    for name, digest in expected.items():
        p = directory / name
        if not p.exists():
            raise RuntimeError(f"frozen file missing: {name}")
        if _file_hash(p) != digest:
            raise RuntimeError(f"frozen file changed: {name}")
    return _file_hash(sums)
