"""M1 step 1: load sources, map categories, sample English seed conversations (spec 7.1, 7.3).

    uv run onegap data [--limit N]      # --limit caps the classifier pool (smoke run, fewer Cohere calls)

Writes (all English; translation and Hinglish come later):
  data/generated/seeds_attack.jsonl       train/dev from XGuard-Train, test from Attack_600
  data/generated/seeds_overrefusal.jsonl  XSTest safe prompts, dev/test
  data/generated/category_labels.jsonl    classifier output with goals (provenance)
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from onegap import CATEGORIES
from onegap.data import categories as C
from onegap.schema import validate_record, write_jsonl

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "data" / "generated"
SEED = 13

ATTACK600 = ("SafeMTData/SafeMTData", "Attack_600")
XGUARD = ("marslabucla/XGuard-Train", "xguard-train.json")
XSTEST = "Paul/XSTest"

SIZES = {"train_per_cat": 80, "dev_per_cat": 8, "test_per_cat": 20, "test_total": 100, "xg_pool_per_cat": 260,
         "xstest_dev": 50, "xstest_test": 150}
# Run 1 (2026-11-10): only 35 drug conversations survived classification + dedup from 260. Extra candidates are
# APPENDED after the base pool so earlier classifier batches keep their cache keys (no re-spent calls).
XG_EXTRA_POOL = {"dangerous_drugs": 450}


# ---------- loaders ----------

def load_attack600() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(*ATTACK600, split=ATTACK600[1])
    return [dict(r) for r in ds]


def load_xguard() -> list[dict]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(XGUARD[0], XGUARD[1], repo_type="dataset", local_dir=ROOT / "data" / "raw" / "xguard")
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(raw):
        human = [m["value"] for m in r["conversations"] if m["from"] == "human"]
        out.append({"xg_idx": i, "human_turns": human})
    return out


def load_xstest_safe() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(XSTEST, split="train")
    return [dict(r) for r in ds if r["label"] == "safe"]


# ---------- pure helpers (unit-tested) ----------

def one_actor_per_query(rows: list[dict], seed: int = SEED) -> list[dict]:
    """Attack_600 has 3 actors per query_id; keep one at random so seeds are distinct behaviours."""
    by_q: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r)
    rng = random.Random(seed)
    return [rng.choice(sorted(v, key=lambda x: x["id"])) for _, v in sorted(by_q.items())]


def stratified_pick(labelled: list[tuple[str, str]], per_cat: int, seed: int = SEED,
                    exclude: set[str] | None = None) -> dict[str, list[str]]:
    """labelled: (id, category). Returns {category: [ids]} with up to per_cat each (seeded shuffle)."""
    exclude = exclude or set()
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for iid, cat in labelled:
        if cat in CATEGORIES and iid not in exclude:
            by_cat[cat].append(iid)
    out = {}
    for cat in CATEGORIES:
        ids = sorted(by_cat.get(cat, []))
        rng.shuffle(ids)
        out[cat] = ids[:per_cat]
    return out


def top_up(labelled: list[tuple[str, str]], pick: dict[str, list[str]], target: int,
           seed: int = SEED) -> set[str]:
    """Fill `pick` in place up to `target` ids, round-robin over categories that still have unused ids.
    Used when a source simply lacks behaviours in a category (Attack_600: 8 drug, 17 fraud). Returns added ids."""
    used = {i for v in pick.values() for i in v}
    rng = random.Random(seed + 7)
    spare: dict[str, list[str]] = defaultdict(list)
    for iid, cat in labelled:
        if cat in CATEGORIES and iid not in used:
            spare[cat].append(iid)
    for v in spare.values():
        v.sort()
        rng.shuffle(v)
    added: set[str] = set()
    while sum(len(v) for v in pick.values()) < target and any(spare.values()):
        for cat in CATEGORIES:
            if spare.get(cat) and sum(len(v) for v in pick.values()) < target:
                iid = spare[cat].pop()
                pick[cat].append(iid)
                added.add(iid)
    return added


def make_record(seed_id: str, split: str, family: str, category: str | None, kind: str, turns: list[str],
                provenance: dict) -> dict:
    rec = {"conv_id": f"{seed_id}-en", "seed_id": seed_id, "split": split, "family": family, "language": "en",
           "kind": kind, "user_turns": [t.strip() for t in turns], "provenance": provenance}
    if category:
        rec["category"] = category
    return validate_record(rec)


# ---------- pipeline ----------

def build_test(limit: int | None) -> tuple[list[dict], list[dict]]:
    rows = [r for r in load_attack600() if C.ATTACK600_PRIOR.get(r["category"]) != "drop"]
    rows = [r for r in one_actor_per_query(rows) if 3 <= len(r["multi_turn_queries"]) <= 6]
    if limit:
        rows = rows[:limit]
    # Attack_600 ships the true harmful behaviour (`plain_query`); classify on that alone. Including the turns made
    # the classifier label the innocuous early turns (dimethylmercury synthesis -> "other").
    labels = C.classify((f"a600-{r['query_id']:04d}", "[final turn] " + r["plain_query"]) for r in rows)
    labelled = [(f"a600-{r['query_id']:04d}", labels.get(f"a600-{r['query_id']:04d}", {}).get("category", "other"))
                for r in rows]
    pick = stratified_pick(labelled, SIZES["test_per_cat"])
    target = SIZES["test_total"] if not limit else len(rows)
    topup = top_up(labelled, pick, target)
    chosen = {i for ids in pick.values() for i in ids}
    by_id = {f"a600-{r['query_id']:04d}": r for r in rows}
    recs = []
    for cat, ids in pick.items():
        for sid in ids:
            r = by_id[sid]
            recs.append(make_record(sid, "test", "actorattack", cat, "attack", r["multi_turn_queries"],
                                    {"source": "SafeMTData/Attack_600", "source_id": r["id"],
                                     "query_id": r["query_id"], "actor": r["actor_name"],
                                     "native_category": r["category"], "goal": labels[sid]["goal"],
                                     "topup": sid in topup}))
    log = [{"id": i, "source": "attack600", **labels.get(i, {}), "chosen": i in chosen} for i, _ in labelled]
    return recs, log


def build_train_dev(limit: int | None) -> tuple[list[dict], list[dict]]:
    convs = [c for c in load_xguard() if 3 <= len(c["human_turns"]) <= 6]
    rng = random.Random(SEED)
    pools: dict[str, list[dict]] = defaultdict(list)
    for c in convs:
        hits = C.keyword_hits("\n".join(c["human_turns"]))
        if hits:
            pools[rng.choice(sorted(hits))].append(c)  # shortlist only; final label comes from the classifier
    pool: list[dict] = []
    per = SIZES["xg_pool_per_cat"] if not limit else max(1, limit // len(CATEGORIES))
    shuffled = {}
    for cat in CATEGORIES:
        cands = pools.get(cat, [])
        rng.shuffle(cands)
        shuffled[cat] = cands
        pool += cands[:per]
    if not limit:
        for cat, extra in XG_EXTRA_POOL.items():
            pool += shuffled.get(cat, [])[per: per + extra]
        print(f"[xguard] shortlist sizes: { {c: len(v) for c, v in shuffled.items()} }; pool {len(pool)}")
    labels = C.classify((f"xg-{c['xg_idx']:05d}", "\n".join(f"[turn {k + 1}] {t}" for k, t in
                                                           enumerate(c["human_turns"]))) for c in pool)
    ids_goals = [(f"xg-{c['xg_idx']:05d}", labels[f"xg-{c['xg_idx']:05d}"]["goal"]) for c in pool
                 if f"xg-{c['xg_idx']:05d}" in labels and labels[f"xg-{c['xg_idx']:05d}"]["category"] in CATEGORIES]
    keep = set(C.dedup_by_goal(ids_goals))
    labelled = [(i, labels[i]["category"]) for i, _ in ids_goals if i in keep]
    dev = stratified_pick(labelled, SIZES["dev_per_cat"], seed=SEED + 1)
    dev_ids = {i for v in dev.values() for i in v}
    train = stratified_pick(labelled, SIZES["train_per_cat"], seed=SEED, exclude=dev_ids)
    by_id = {f"xg-{c['xg_idx']:05d}": c for c in pool}
    recs = []
    for split, pick in (("dev", dev), ("train", train)):
        for cat, ids in pick.items():
            for sid in ids:
                recs.append(make_record(sid, split, "xguard", cat, "attack", by_id[sid]["human_turns"],
                                        {"source": "marslabucla/XGuard-Train", "source_id": by_id[sid]["xg_idx"],
                                         "goal": labels[sid]["goal"]}))
    log = [{"id": i, "source": "xguard", **labels.get(i, {}), "kept_after_dedup": i in keep,
            "split": "dev" if i in dev_ids else ("train" if any(i in v for v in train.values()) else None)}
           for i in (f"xg-{c['xg_idx']:05d}" for c in pool)]
    return recs, log


def build_overrefusal() -> list[dict]:
    rows = load_xstest_safe()
    rng = random.Random(SEED)
    rows = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(rows)
    recs = []
    for i, r in enumerate(rows[: SIZES["xstest_dev"] + SIZES["xstest_test"]]):
        split = "dev" if i < SIZES["xstest_dev"] else "test"
        recs.append(make_record(f"xst-{r['id']}", split, "xstest", None, "overrefusal", [r["prompt"]],
                                {"source": XSTEST, "source_id": r["id"], "type": r["type"]}))
    return recs


def main(args) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from onegap.llm import calls_this_month

    before = calls_this_month()
    test, log_t = build_test(args.limit)
    traindev, log_x = build_train_dev(args.limit)
    over = build_overrefusal()
    attacks = traindev + test

    from onegap.splits import check_splits

    errs = check_splits(attacks + over)
    if errs:
        print("\n".join(errs))
        return 1
    suffix = f".limit{args.limit}" if args.limit else ""
    n1 = write_jsonl(GEN / f"seeds_attack{suffix}.jsonl", attacks)
    n2 = write_jsonl(GEN / f"seeds_overrefusal{suffix}.jsonl", over)
    write_jsonl(GEN / f"category_labels{suffix}.jsonl", log_t + log_x)

    print(f"attack seeds: {n1}  over-refusal seeds: {n2}  Cohere calls used now: {calls_this_month() - before} "
          f"(month total {calls_this_month()})")
    counts = Counter((r["split"], r["category"]) for r in attacks)
    for split in ("train", "dev", "test"):
        row = {c: counts.get((split, c), 0) for c in CATEGORIES}
        print(f"  {split:5s} {sum(row.values()):4d}  {row}")
    print(f"  over-refusal: {Counter(r['split'] for r in over)}")
    short = [(s, c) for (s, c), n in counts.items() if n < {"train": SIZES['train_per_cat'],
                                                           "dev": SIZES['dev_per_cat'],
                                                           "test": SIZES['test_per_cat']}[s]]
    if short:
        print(f"  NOTE under-filled cells (log in RUNS.md): {sorted(short)}")
    return 0
