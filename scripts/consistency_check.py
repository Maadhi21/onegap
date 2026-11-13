"""Mechanical consistency checks on translated test conversations (pre-freeze worklist for the HUMAN reviewers).

    uv run python scripts/consistency_check.py [--split test] [--csv data/generated/spot_check_hi.csv ...]

Default input: data/generated/conversations.jsonl, the given split, Hindi and Hinglish records paired with their
English record by seed_id. With --csv, reads the spot-check sheets instead (english_source + hi/hinglish columns).
Output: data/generated/consistency_worklist.csv with one row per finding: language, seed_id, turn, code, detail.
`detail` names only the entity or marker involved, never the request text.

Codes
  NAME_DRIFT        hi: two near-identical spellings of a rare word (a name) in one conversation (मैलन / मलोन)
  NAME_MISSING      an English name used in several turns has no shared rendering in this turn (hi), or does not
                    appear in this turn at all (Hinglish): translated as a common word, dropped, or misspelled
  MARKER_DROPPED    the English turn has a specificity marker ("specific", "in detail", "step by step",
                    "theoretically", "allegedly", ...) and the translation has none: possible softening
  BACKREF_MAYBE_LOST  Hinglish: the English turn points back ("those", "such", "you mentioned") and the Hinglish turn
                    has no demonstrative or back-reference word: possible loss of the multi-turn link
  WRONG_SCRIPT      Devanagari in a Hinglish turn, or under 60% Devanagari in a Hindi turn
These are heuristics that build a worklist; every finding is decided by a human reviewer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "generated"

DEV = re.compile(r"[ऀ-ॿ]")
DEV_WORD = re.compile(r"[ऀ-ॿ]+")
LAT_WORD = re.compile(r"[A-Za-z][A-Za-z.'-]*")

# Capitalised words that are not names when they start a sentence or are generic.
NOT_NAMES = {"can", "could", "what", "how", "who", "why", "when", "where", "which", "are", "is", "do", "does", "did",
             "please", "in", "on", "for", "the", "a", "an", "and", "or", "if", "based", "thank", "thanks", "given",
             "describe", "explain", "summarize", "connect", "applying", "according", "as", "i", "my", "you", "your",
             "would", "should", "with", "from", "this", "that", "these", "those", "it", "its", "of", "to", "by"}

MARKERS_EN = {"specific": r"\bspecific", "detail": r"\bdetail", "step": r"\bstep[- ]by[- ]step|\bsteps\b",
              "theoretical": r"\btheoretical", "alleged": r"\balleged", "exact": r"\bexact", "precise": r"\bprecise",
              "practical": r"\bpractical", "example": r"\bexamples?\b"}
MARKERS_HI = {"specific": r"विशिष्ट|विशेष|खास|ख़ास|specific", "detail": r"विस्तार|विस्तृत|ब्योरा|विवरण|detail",
              "step": r"चरण|कदम|step", "theoretical": r"सैद्धांतिक|theoretical", "alleged": r"कथित|alleged",
              "exact": r"सटीक|ठीक[- ]ठीक|exact", "precise": r"सटीक|precise", "practical": r"व्यावहारिक|practical",
              "example": r"उदाहरण|example"}
MARKERS_HG = {"specific": r"specific|khas|khaas|vishesh", "detail": r"detail|vistar",
              "step": r"step|kadam|charan", "theoretical": r"theoretical|siddhant", "alleged": r"alleged|kathit",
              "exact": r"exact|sateek|bilkul", "precise": r"precise|exact|sateek",
              "practical": r"practical|vyavaharik", "example": r"example|udaharan|misaal"}
BACKREF_EN = re.compile(r"\b(those|these|such|you mentioned|mentioned|the same|that speech|those strategies|"
                        r"this approach|the above|earlier)\b", re.IGNORECASE)
BACKREF_HG = re.compile(r"\b(un|in|unhi|inhi|unke|inke|unka|inka|unki|inki|is|us|isi|usi|aise|aisa|aisi|waise|"
                        r"wahi|wohi|vahi|ye|yeh|wo|woh|aapne|jo|uss|iss|upar|pehle|same)\b", re.IGNORECASE)


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def english_names(turn: str) -> set[str]:
    """Proper-noun guesses: runs of capitalised words that are not sentence-initial and not generic, joined with
    spaces ("Mary Mallon", "Unmandated Lab Workers"); possessive 's removed."""
    out = set()
    for sent in re.split(r"(?<=[.?!])\s+", turn):
        words = [re.sub(r"'s$", "", w.strip(".'-")) for w in LAT_WORD.findall(sent)]
        run: list[str] = []
        for k, w in enumerate(words + [""]):
            is_name = (len(w) >= 3 and w.lower() not in NOT_NAMES
                       and ((w[:1].isupper() and k > 0) or (w.isupper() and len(w) >= 3)))
            if is_name:
                run.append(w)
            elif run:
                out.add(" ".join(run))
                run = []
    return out


SUFFIXES = ("ाओं", "ओं", "ों", "ें", "एं", "एँ", "ाएं", "ाएँ", "ियों", "ने", "नी", "ना", "ती", "ते", "ता", "ाने", "ाती",
            "्स", "ी", "े", "ा", "ो", "ि", "ु", "ू", "ं")


def stem(w: str) -> str:
    for suf in sorted(SUFFIXES, key=len, reverse=True):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def check_conversation(lang: str, seed_id: str, en: list[str], tr: list[str], rare: set[str]) -> list[dict]:
    rows: list[dict] = []

    def add(turn: int, code: str, detail: str) -> None:
        rows.append({"language": lang, "seed_id": seed_id, "turn": turn, "code": code, "detail": detail})

    n = min(len(en), len(tr))
    for t in range(n):
        letters = [c for c in tr[t] if c.isalpha()]
        dev = sum(bool(DEV.match(c)) for c in letters) / max(1, len(letters))
        if (lang == "hinglish" and dev > 0) or (lang == "hi" and dev < 0.6):
            add(t + 1, "WRONG_SCRIPT", f"devanagari share {dev:.2f}")
        markers = MARKERS_HI if lang == "hi" else MARKERS_HG
        for name, rx in MARKERS_EN.items():
            if re.search(rx, en[t], re.IGNORECASE) and not re.search(markers[name], tr[t], re.IGNORECASE):
                add(t + 1, "MARKER_DROPPED", name)
        if lang == "hinglish" and BACKREF_EN.search(en[t]) and not BACKREF_HG.search(tr[t]):
            add(t + 1, "BACKREF_MAYBE_LOST", BACKREF_EN.search(en[t]).group(0).lower())

    # names used in 2+ English turns
    name_turns: dict[str, set[int]] = defaultdict(set)
    for t in range(n):
        for w in english_names(en[t]):
            name_turns[w].add(t)
    for name, ts in name_turns.items():
        if len(ts) < 2:
            continue
        if lang == "hinglish":
            for t in sorted(ts):
                if not any(part.lower() in tr[t].lower() for part in name.split()):
                    add(t + 1, "NAME_MISSING", name)
            continue
        # hi: the rendering = rare Devanagari words shared by most of the turns that use the name
        cnt = Counter(w for t in ts for w in set(DEV_WORD.findall(tr[t])) if w in rare)
        rendering = {w for w, c in cnt.items() if c >= max(2, (len(ts) + 1) // 2)}
        for t in sorted(ts):
            if any(part in tr[t] for part in name.split()):
                continue  # kept in Latin script
            if not rendering or not (rendering & set(DEV_WORD.findall(tr[t]))):
                add(t + 1, "NAME_MISSING", name)

    if lang == "hi":  # spelling drift of a name: compare each name's main rendering with near-variants in its turns
        seen = set()
        for name, ts in name_turns.items():
            cnt = Counter(w for t in ts for w in set(DEV_WORD.findall(tr[t])) if w in rare and len(w) >= 4)
            for main, c in cnt.items():
                if c < 2:
                    continue  # the main rendering is used in at least two of the name's turns
                for t in sorted(ts):
                    for w in set(DEV_WORD.findall(tr[t])):
                        if w == main or w not in rare or len(w) < 4 or stem(w) == stem(main) or (w, main) in seen:
                            continue
                        if w[0] == main[0] and lev(w, main) <= max(2, int(0.45 * max(len(w), len(main)))):
                            seen.add((w, main))
                            tw = [k + 1 for k in range(n) if w in DEV_WORD.findall(tr[k])]
                            add(t + 1, "NAME_DRIFT", f"{name}: {main} / {w} (turns {tw})")
    return rows


def load_pairs(a) -> list[tuple[str, str, list[str], list[str]]]:
    pairs = []
    if a.csv:
        for path in a.csv:
            rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
            lang = "hi" if "hi" in rows[0] else "hinglish"
            by: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by[r["seed_id"]].append(r)
            for sid, rs in by.items():
                rs.sort(key=lambda r: int(r["turn"]))
                pairs.append((lang, sid, [r["english_source"] for r in rs], [r[lang] for r in rs]))
        return pairs
    recs = [r for r in (json.loads(x) for x in open(GEN / "conversations.jsonl", encoding="utf-8") if x.strip())
            if r["split"] == a.split]
    en = {r["seed_id"]: r["user_turns"] for r in recs if r["language"] == "en"}
    for r in recs:
        if r["language"] in ("hi", "hinglish") and r["seed_id"] in en:
            pairs.append((r["language"], r["seed_id"], en[r["seed_id"]], r["user_turns"]))
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--csv", nargs="*", type=Path)
    p.add_argument("--out", type=Path, default=GEN / "consistency_worklist.csv")
    a = p.parse_args()
    pairs = load_pairs(a)
    # "rare" Devanagari words = used in at most 2 Hindi conversations (names and unusual terms, not function words)
    df = Counter(w for lang, _, _, tr in pairs if lang == "hi" for w in {x for t in tr for x in DEV_WORD.findall(t)})
    rare = {w for w, c in df.items() if c <= 2}
    rows = [r for lang, sid, en, tr in pairs for r in check_conversation(lang, sid, en, tr, rare)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["language", "seed_id", "turn", "code", "detail", "reviewer_decision"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "reviewer_decision": ""})
    convs = Counter((r["language"], r["seed_id"]) for r in rows)
    print(f"{len(pairs)} conversations checked, {len(rows)} findings in {len(convs)} conversations -> {a.out}")
    for (lang, code), c in sorted(Counter((r["language"], r["code"]) for r in rows).items()):
        print(f"  {lang:9s} {code:20s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
