"""Build the HUMAN spot-check sheets (spec M1): 30 Hindi + 30 Hinglish TEST attack conversations.

    uv run python scripts/spot_check.py [--n 30] [--seed 11]

Stratified by category (round-robin), same seed_ids for hi and Hinglish where both passed the gate, so reviewers
can compare. Writes data/generated/spot_check_{hi,hinglish}.csv (UTF-8 with BOM so Excel shows Devanagari).
One row per user turn; reviewers fill `faithful` (y/n), `natural` (y/n), `notes` on the first row of each
conversation. Files hold attack prompts: they stay local (data/generated is gitignored) and go only to reviewers.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from onegap.schema import read_jsonl

GEN = Path(__file__).resolve().parents[1] / "data" / "generated"


def pick(recs: list[dict], n: int, seed: int) -> list[str]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["seed_id"]):
        by_cat[r["category"]].append(r["seed_id"])
    rng = random.Random(seed)
    for v in by_cat.values():
        rng.shuffle(v)
    out: list[str] = []
    while len(out) < n and any(by_cat.values()):
        for c in sorted(by_cat):
            if by_cat[c] and len(out) < n:
                out.append(by_cat[c].pop())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    recs = [r for r in read_jsonl(GEN / "conversations.jsonl") if r["split"] == "test" and r["kind"] == "attack"]
    idx = {(r["seed_id"], r["language"]): r for r in recs}
    both = [r for r in recs if r["language"] == "en" and (r["seed_id"], "hi") in idx
            and (r["seed_id"], "hinglish") in idx]
    chosen = pick(both, a.n, a.seed)
    for lang in ("hi", "hinglish"):
        ids = chosen + pick([r for r in recs if r["language"] == lang and r["seed_id"] not in chosen],
                            a.n - len(chosen), a.seed)
        path = GEN / f"spot_check_{lang}.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conv", "seed_id", "category", "turn", "english_source", lang, "fidelity_min_cos",
                        "faithful (y/n)", "natural (y/n)", "notes"])
            for k, sid in enumerate(ids, start=1):
                en, tr = idx[(sid, "en")], idx[(sid, lang)]
                for t, (s, x) in enumerate(zip(en["user_turns"], tr["user_turns"], strict=True), start=1):
                    w.writerow([k, sid, en["category"], t, s, x,
                                tr["provenance"]["fidelity_min_cos"] if t == 1 else "", "", "", ""])
        print(f"wrote {path} ({len(ids)} conversations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
