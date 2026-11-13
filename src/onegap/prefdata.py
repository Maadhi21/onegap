"""On-policy multi-turn DPO correction pairs (spec 8.6).

    uv run onegap prefdata [--limit N]            (local; teacher = Cohere command-a through onegap.llm)

Inputs: labelled train transcripts of base model A, listed in data/generated/prefdata_runs.txt (one run dir per
line; attack and benign runs), plus data/generated/conversations.jsonl for English sources.
Output: data/generated/prefpairs_train.jsonl in TRL's conversational DPO format
  {"prompt": [messages...], "chosen": [{"role": "assistant", ...}], "rejected": [...], "meta": {...}}
and data/generated/prefpairs_report.json (counts, gate failures, 20 random pairs for the HUMAN spot check).

Pairs:
  harmful       first unsafe turn: rejected = base model's unsafe reply, chosen = teacher refusal that names the
                concern and redirects (same language and script, length-matched)
  harmful_final conversation never failed: pair at the final turn, rejected = base model's (non-operational)
                compliant reply, chosen = teacher refusal; capped at 30% of harmful pairs
  benign        base model refused unnecessarily: rejected = that refusal, chosen = teacher helpful reply;
                otherwise chosen = base model's own good reply, rejected = teacher-written unnecessary refusal
Gates: language/script, refusal present (or absent) where required, length within +-25% (words), dedup.
Unsafe `rejected` texts never leave the private work repo; publishing goes through onegap.redact (Invariant 7).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from onegap.judges import looks_like_refusal
from onegap.schema import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "data" / "generated"
OUT = GEN / "prefpairs_train.jsonl"
LENGTH_TOL = 0.25
FINAL_CAP = 0.30
BENIGN_RATIO = 2.0
PER_CALL_CHARS = 15000  # ~5-6 replies per call: the full plan fits the trial key (~217 calls; 9000 needed ~381)
LANG_NAME = {"en": "English", "hi": "Hindi in Devanagari script",
             "hinglish": "Hinglish (romanized Hindi-English code-mixing, Latin script only, WhatsApp style)"}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_HINDI_WORDS = re.compile(r"\b(?:hai|hain|nahi|nahin|ka|ki|ke|ko|main|aap|aapko|yeh|ye|kar|karna|mein|bhi|toh|"
                          r"lekin|agar|hoon|ho)\b", re.IGNORECASE)

REFUSAL_SYSTEM = (
    "You write training data that teaches a small assistant to decline harmful requests well. For each item you "
    "get the conversation's recent user messages, the harm category and a target length. Write the assistant's "
    "reply to the LAST user message: decline clearly, name the specific concern in one sentence (taking the whole "
    "conversation into account: earlier harmless-looking steps may build toward harm), then be genuinely helpful "
    "with legitimate, safe alternatives (for example how to recognise and report scams, security best practices "
    "for defenders, legal and historical context, health and safety resources). Never include operational details "
    "that would help cause harm. Write in the requested language and script only, in a warm, non-preachy tone, at "
    'about the target length. Return JSON {"items": [{"id": str, "reply": str}]}.'
)
HELPFUL_SYSTEM = (
    "You write training data for a helpful assistant. For each item you get a conversation's recent user "
    "messages about a sensitive but legitimate topic (fraud awareness, defensive security, history, medicine "
    "safety, supporting victims), and a target length. Write a helpful, accurate, complete answer to the LAST "
    "user message, in the requested language and script only, at about the target length. Do not refuse, do not "
    'lecture. Return JSON {"items": [{"id": str, "reply": str}]}.'
)
OVERREFUSAL_SYSTEM = (
    "You write training data showing what an assistant should NOT do: refuse a legitimate request. For each item "
    "you get the recent user messages of a harmless conversation and a target length. Write an over-cautious "
    "reply that refuses or deflects the LAST user message, citing safety or policy concerns, without answering it. "
    "Use the requested language and script only, at about the target length. "
    'Return JSON {"items": [{"id": str, "reply": str}]}.'
)
# Appended to every teacher prompt: in the first smoke the teacher wrote ~75-word refusals for ~250-word targets
# (all 8 failed the +-25% length gate). Length matching is a spec requirement (8.6.5), so it is a hard rule.
LENGTH_RULE = (
    " LENGTH IS A HARD REQUIREMENT: each reply must reach the item's minimum word count. Long targets "
    "need several paragraphs; for refusals, spend the extra length on concrete, safe help (steps, resources, "
    "context), never on repetition."
)
REFUSAL_SYSTEM += LENGTH_RULE
HELPFUL_SYSTEM += LENGTH_RULE
OVERREFUSAL_SYSTEM += LENGTH_RULE
SCHEMA = {
    "type": "object", "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["id", "reply"],
        "properties": {"id": {"type": "string"}, "reply": {"type": "string"}}}}},
}


# ------------------------------------------------------------------ pure helpers (tested)

def words(text: str) -> int:
    return len(text.split())


def length_ok(a: str, b: str, tol: float = LENGTH_TOL) -> bool:
    wa, wb = words(a), words(b)
    return wa > 0 and wb > 0 and abs(wa - wb) <= tol * max(wa, wb)


def script_ok(text: str, language: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    dev = sum(bool(_DEVANAGARI.match(c)) for c in letters) / len(letters)
    if language == "hi":
        return dev >= 0.6
    if language == "hinglish" and len(_HINDI_WORDS.findall(text)) < 0.03 * max(1, len(text.split())):
        return False  # the teacher fell back to plain English (50/393 Hinglish replies in the first build)
    return dev == 0 and bool(_LATIN.search(text))  # en, hinglish: Latin only


# Over-refusal side of benign_ok pairs: the teacher mostly writes soft deflections ("discussing this could lead to
# misuse; consult official sources") rather than "I can't" refusals, which the judge's refusal phrases miss
# (507/1003 failed on that in the first full build). The harmful side accepts it too: its Hinglish declines read
# "ye bahut khatarnak hai, galat fayda utha sakta hai; instead...". Used only for these gates, not for eval labels.
DEFLECTION_PATTERNS = {
    "en": r"\b(?:could|might|may) (?:potentially )?(?:lead to|pose|result in) (?:\w+ )?(?:misuse|harm|risks?|"
          r"misinterpretation|confusion)|\bsensitive (?:topic|subject|matter|area)|\bunable to (?:provide|discuss|"
          r"share|offer)|\bnot (?:appropriate|advisable) (?:for me )?to (?:discuss|provide|share)|"
          r"\b(?:policy|safety) (?:and policy )?(?:restrictions|concerns|reasons)",
    "hi": r"संवेदनशील|दुरुपयोग|गलत जानकारी|चर्चा (?:करना )?(?:उचित|ठीक) नहीं",
    "hinglish": r"\b(?:sensitive|risky|khatarnak|dangerous)\b|\bmisuse\b|galat jaank?aari|discuss (?:karna|nahi)|"
                r"galat (?:fayda|haath)",
}
_DEFLECT_RX = {k: re.compile(v, re.IGNORECASE) for k, v in DEFLECTION_PATTERNS.items()}


def looks_like_deflection(reply: str, language: str, head_chars: int = 600) -> bool:
    head = reply[:head_chars]
    return any(_DEFLECT_RX[k].search(head) for k in {language, "en"} if k in _DEFLECT_RX)


def prompt_messages(system: str, tr: dict, turn: int) -> list[dict]:
    """History up to and including user turn `turn` (1-based), with the model's own earlier replies."""
    msgs = [{"role": "system", "content": system}]
    for t in tr["turns"][: turn]:
        msgs.append({"role": "user", "content": t["user"]})
        if t["turn"] < turn:
            msgs.append({"role": "assistant", "content": t["assistant"]})
    return msgs


def plan_harmful(trs: list[dict], cap: float = FINAL_CAP, seed: int = 13) -> list[dict]:
    """One candidate per attack conversation: first unsafe turn, else (capped) the final turn."""
    first, final = [], []
    for tr in trs:
        if tr.get("kind", "attack") != "attack":
            continue
        f = next((t["turn"] for t in tr["turns"] if t["label"] == "unsafe"), None)
        if f is not None:
            first.append({"tr": tr, "turn": f, "kind": "harmful"})
        elif tr["turns"][-1]["label"] != "refused":
            final.append({"tr": tr, "turn": len(tr["turns"]), "kind": "harmful_final"})
    n_final = min(len(final), int(cap / (1 - cap) * len(first)))
    random.Random(seed).shuffle(final)
    return first + final[:n_final]


def plan_benign(trs: list[dict], n_max: int, seed: int = 13) -> list[dict]:
    """One candidate per benign conversation: first unnecessary refusal, else the final turn."""
    out = []
    for tr in trs:
        if tr.get("kind") != "benign":
            continue
        r = next((t["turn"] for t in tr["turns"] if t["label"] == "refused"), None)
        out.append({"tr": tr, "turn": r or len(tr["turns"]), "kind": "benign_refused" if r else "benign_ok"})
    out.sort(key=lambda c: (c["kind"] != "benign_refused", c["tr"]["conv_id"]))  # keep real over-refusals first
    keep, rest = [c for c in out if c["kind"] == "benign_refused"], [c for c in out if c["kind"] != "benign_refused"]
    random.Random(seed).shuffle(rest)
    return (keep + rest)[:n_max]


def teacher_task(c: dict) -> tuple[str, str]:
    """(system prompt, which side the teacher writes) for a candidate."""
    if c["kind"] in ("harmful", "harmful_final"):
        return REFUSAL_SYSTEM, "chosen"
    if c["kind"] == "benign_refused":
        return HELPFUL_SYSTEM, "chosen"
    return OVERREFUSAL_SYSTEM, "rejected"


def gate(c: dict, teacher_reply: str) -> str | None:
    """None if the pair passes, else the reason."""
    base = c["tr"]["turns"][c["turn"] - 1]["assistant"]
    if not teacher_reply.strip():
        return "empty"
    why = content_gate(c["kind"], teacher_reply, base, c["tr"]["language"])
    if why:
        return why
    if match_lengths(base, teacher_reply) is None:
        return "length"
    return None


def content_gate(kind: str, teacher: str, base: str, lang: str) -> str | None:
    """Script and refusal checks on both replies (shared by the build and `prefdata-refilter`)."""
    if not script_ok(teacher, lang):
        return "script"
    refusal = looks_like_refusal(teacher, lang, head_chars=600)
    if kind in ("harmful", "harmful_final") and not (refusal or looks_like_deflection(teacher, lang)):
        return "no_refusal"
    if kind == "benign_refused" and refusal:
        return "helpful_reply_refuses"
    if kind == "benign_ok" and (not (refusal or looks_like_deflection(teacher, lang)) or looks_like_refusal(base, lang)):
        return "refusal_pair_unclear"
    return base_gate(kind, base, lang)


def base_gate(kind: str, base: str, lang: str) -> str | None:
    """Checks on the base model's side of the pair (added 2026-11-12 after the HUMAN spot check).

    - base_script (benign_ok only, where the base reply is CHOSEN): 55/167 Hinglish and 9/291 Hindi chosen replies
      were in the wrong script, which would train script switching. Hinglish chosen may be plain English (Latin)
      but never Devanagari. Where the base reply is REJECTED its script does not matter.
    - base_not_refusal (benign_refused): the base reply must open with a refusal; the turn label said "refused"
      for 18/25 replies that answer the question (Hindi refusal-phrase false positives, fixed in judges).
    """
    if kind == "benign_ok" and not base_script_ok(base, lang):
        return "base_script"
    if kind == "benign_refused" and not looks_like_refusal(base, lang, head_chars=200):
        return "base_not_refusal"
    return None


def base_script_ok(text: str, language: str) -> bool:
    if language == "hinglish":
        return not _DEVANAGARI.search(text) and bool(_LATIN.search(text))
    return script_ok(text, language)


EXCLUDE = GEN / "prefpairs_exclude.txt"


def load_exclusions(path: Path = EXCLUDE) -> dict[tuple[str, int], str]:
    """HUMAN spot-check rejections: lines `conv_id turn reason...` (# comments allowed)."""
    out: dict[tuple[str, int], str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                conv, turn, *why = line.split()
                out[(conv, int(turn))] = " ".join(why) or "excluded"
    return out


def pair_gate(p: dict, exclusions: dict[tuple[str, int], str]) -> str | None:
    """Re-check a built pair on its stored text: HUMAN exclusions, then the same content gates as the build."""
    m = p["meta"]
    if (m["conv_id"], m["turn"]) in exclusions:
        return "human_excluded"
    teacher_is_chosen = m["teacher_side"] == "chosen"
    teacher = (p["chosen"] if teacher_is_chosen else p["rejected"])[0]["content"]
    base = (p["rejected"] if teacher_is_chosen else p["chosen"])[0]["content"]
    return content_gate(m["kind"], teacher, base, m["language"])


_SENT_END = re.compile(r"(?<=[.!?।])\s+|\n+")


def trim_to_words(text: str, max_words: int) -> str | None:
    """Longest prefix of whole sentences with at most max_words words; None if the first sentence is too long."""
    out, n = [], 0
    for s in (x for x in _SENT_END.split(text.strip()) if x.strip()):
        w = words(s)
        if n + w > max_words:
            break
        out.append(s.strip())
        n += w
    return " ".join(out) if out else None


def match_lengths(base: str, teacher: str, tol: float = LENGTH_TOL) -> tuple[str, str, str | None] | None:
    """(base, teacher, trimmed side) with lengths within tol (spec 8.6.5). The batched teacher writes ~60% of the
    target length, so the longer reply is cut back to whole sentences; None if that cannot reach the band."""
    if length_ok(base, teacher, tol):
        return base, teacher, None
    longer_is_base = words(base) > words(teacher)
    longer, shorter = (base, teacher) if longer_is_base else (teacher, base)
    cut = trim_to_words(longer, int(words(shorter) / (1 - tol)))
    if cut is None or not length_ok(cut, shorter, tol):
        return None
    return (cut, teacher, "base") if longer_is_base else (base, cut, "teacher")


def make_pair(c: dict, system: str, teacher_reply: str) -> dict:
    tr, turn = c["tr"], c["turn"]
    base, teacher_reply, trimmed = match_lengths(tr["turns"][turn - 1]["assistant"], teacher_reply)
    _, side = teacher_task(c)
    chosen, rejected = (teacher_reply, base) if side == "chosen" else (base, teacher_reply)
    return {"prompt": prompt_messages(system, tr, turn),
            "chosen": [{"role": "assistant", "content": chosen.strip()}],
            "rejected": [{"role": "assistant", "content": rejected.strip()}],
            "meta": {"seed_id": tr["seed_id"], "conv_id": tr["conv_id"], "language": tr["language"],
                     "category": tr.get("category"), "kind": c["kind"], "turn": turn,
                     "teacher": "cohere/command-a-03-2025", "teacher_side": side,
                     "rejected_unsafe": c["kind"] == "harmful",
                     "trimmed": {"base": "chosen" if side == "rejected" else "rejected",
                                 "teacher": side}.get(trimmed)}}


# ------------------------------------------------------------------ teacher calls

def _render(c: dict, k: int) -> str:
    tr, turn = c["tr"], c["turn"]
    users = [t["user"] for t in tr["turns"][:turn]][-3:]
    base = tr["turns"][turn - 1]["assistant"]
    hist = "\n".join(f"  [user] {u[:1200]}" for u in users)
    cat = f", category: {tr.get('category')}" if tr.get("category") else ""
    n = words(base)
    return (f"### id={k}\nlanguage: {LANG_NAME[tr['language']]}{cat}\n"
            f"required length: AT LEAST {round(n * 1.1)} words, aim for {round(n * 1.3)} words\n"
            f"recent user messages (last one is the one to answer):\n{hist}")


def run_teacher(cands: list[dict], log=print) -> dict[int, str]:
    from onegap.llm import BudgetExceeded, chat_json

    out: dict[int, str] = {}
    by_task: dict[str, list[int]] = {}
    for k, c in enumerate(cands):
        by_task.setdefault(teacher_task(c)[0], []).append(k)
    for system, ks in by_task.items():
        batch, size = [], 0
        batches = []
        for k in ks:
            n = len(_render(cands[k], k)) + 4 * words(cands[k]["tr"]["turns"][cands[k]["turn"] - 1]["assistant"])
            if batch and size + n > PER_CALL_CHARS:
                batches.append(batch)
                batch, size = [], 0
            batch.append(k)
            size += n
        if batch:
            batches.append(batch)
        log(f"[prefdata] {len(ks)} items in {len(batches)} teacher calls ({system[:40]}...)")

        def call(b: list[int], system: str = system) -> list[dict]:
            """One teacher call; on failure (timeout, truncated JSON) split in half, down to single items."""
            user = "\n\n".join(_render(cands[k], k) for k in b)
            try:
                return chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 SCHEMA, purpose="prefdata").get("items", [])
            except BudgetExceeded:
                raise
            except RuntimeError as e:
                if len(b) == 1:
                    log(f"[prefdata] giving up on item {b[0]}: {e}")
                    return []
                log(f"[prefdata] batch of {len(b)} failed; splitting")
                return call(b[: len(b) // 2], system) + call(b[len(b) // 2:], system)

        for i, b in enumerate(batches):
            try:
                items = call(b)
            except BudgetExceeded as e:
                log(f"[prefdata] STOP: {e}")
                return out
            for it in items:
                if str(it.get("id", "")).isdigit():
                    out[int(it["id"])] = it.get("reply", "")
            if (i + 1) % 20 == 0:
                log(f"[prefdata] {i + 1}/{len(batches)} calls, {len(out)} replies")
    return out


def near_duplicates(texts: list[str], threshold: float = 0.8) -> set[int]:
    """Indices of later texts that are near-duplicates (word-bigram Jaccard) of an earlier kept text."""
    from onegap.data.categories import jaccard

    kept: list[str] = []
    dup: set[int] = set()
    for i, t in enumerate(texts):
        if any(jaccard(t, k) >= threshold for k in kept):
            dup.add(i)
        else:
            kept.append(t)
    return dup


def main(args) -> int:
    runs = [ROOT / p.strip() for p in (GEN / "prefdata_runs.txt").read_text().splitlines() if p.strip()]
    trs = [t for r in runs for t in read_jsonl(r / "transcripts.jsonl") if t.get("split", "train") == "train"]
    system = (ROOT / "prompts" / "system_A.txt").read_text(encoding="utf-8").strip()
    harmful = plan_harmful(trs)
    benign = plan_benign(trs, int(BENIGN_RATIO * len(harmful)))
    cands = harmful + benign
    if args.limit:
        cands = cands[: args.limit]
    print(f"[prefdata] {len(trs)} transcripts -> candidates {Counter(c['kind'] for c in cands)}")
    replies = run_teacher(cands)
    exclusions = load_exclusions()
    pairs, fails = [], Counter()
    for k, c in enumerate(cands):
        why = "missing" if k not in replies else gate(c, replies[k])
        if why:
            fails[f"{c['kind']}:{why}"] += 1
            continue
        pair = make_pair(c, system, replies[k])
        why = pair_gate(pair, exclusions)
        if why:
            fails[f"{c['kind']}:{why}"] += 1
            continue
        pairs.append(pair)
    dup = near_duplicates([p["chosen"][0]["content"] for p in pairs])
    pairs = [p for i, p in enumerate(pairs) if i not in dup]
    write_jsonl(OUT, pairs)
    sample = random.Random(13).sample(pairs, min(20, len(pairs)))
    write_report(pairs, fails, len(dup), [spot_item(p) for p in sample])
    return 0


def spot_item(p: dict) -> dict:
    return {"meta": p["meta"], "last_user": p["prompt"][-1]["content"][:400],
            "chosen": p["chosen"][0]["content"][:600],
            "rejected": "[REDACTED unsafe reply]" if p["meta"]["rejected_unsafe"] else p["rejected"][0]["content"][:600]}


def write_report(pairs: list[dict], fails: Counter, n_dup: int, spot: list[dict], **extra) -> dict:
    counts = Counter((p["meta"]["kind"], p["meta"]["language"]) for p in pairs)
    report = {"pairs": len(pairs), "by_kind_language": {f"{k}/{lang}": v for (k, lang), v in sorted(counts.items())},
              "by_category": dict(Counter(p["meta"]["category"] for p in pairs)), "gate_failures": dict(fails),
              "dedup_removed": n_dup, **extra, "spot_check": spot}
    (GEN / "prefpairs_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "spot_check"}, indent=2))
    return report


def main_refilter(_args=None) -> int:
    """Re-apply the base-side gates and HUMAN exclusions to the existing pair file (0 Cohere calls).

    Equivalent to a full rebuild from cache: the new checks only remove pairs, and the original build removed no
    near-duplicates, so no pair can be un-shadowed. The 20-pair HUMAN spot sample is kept as reviewed.
    """
    old = json.loads((GEN / "prefpairs_report.json").read_text(encoding="utf-8"))
    pairs = list(read_jsonl(OUT))
    exclusions = load_exclusions()
    fails = Counter(old.get("gate_failures", {}))
    kept = []
    for p in pairs:
        why = pair_gate(p, exclusions)
        if why:
            fails[f"{p['meta']['kind']}:{why}"] += 1
        else:
            kept.append(p)
    write_jsonl(OUT, kept)
    write_report(kept, fails, old.get("dedup_removed", 0), old.get("spot_check", []),
                 refiltered={"from": len(pairs), "to": len(kept), "exclusions": len(exclusions)})
    return 0
