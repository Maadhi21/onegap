"""Fidelity gate + record assembly (spec 7.2, 7.3, 7.4).

Two entry points:
  onegap score-hinglish   (job, cpu-upgrade is enough)  LaBSE cosine(src, v1|v2) for every Hinglish variant
                          inputs: data/generated/hinglish_raw.jsonl -> $RUN_DIR/hinglish_scored.jsonl
  onegap curate           (local)  apply the gate to translations + scored Hinglish, write per-language records:
                          data/generated/conversations.jsonl, data/generated/fidelity_report.json

Gate (every turn of a conversation must pass, else the conversation fails and is listed for regeneration):
  hi/bn    : cos(src, back-translation) >= 0.80 and target-script share >= 0.80
  hinglish : best of v1/v2 with no Devanagari, Hindi-lexicon share >= 0.25, not pure English, and
             cos(src, English back-translation) >= 0.80 (back-translation: `onegap codemix-back`, Cohere;
             scored by `onegap score-hinglish-back`, job)
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from onegap.schema import read_jsonl, validate_record, write_jsonl

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "data" / "generated"
MIN_COS = 0.80
MIN_SCRIPT = 0.80
MIN_LEXICON = 0.25

# Common romanized Hindi function/content words (spelling variants included). Used only for the Hinglish gate.
HINDI_LEXICON = set("""
hai hain ho hoon hu tha thi the thaa raha rahi rahe rha rhi kya kyaa kyun kyon kaise kaisa kaisi kab kahan kaha
kaun kitna kitne kitni koi kuch kuchh mai main mein me mujhe mujhko mera meri mere hum hamara hamari hamare humein
hume aap aapka aapki aapke aapko tum tumhara tumhari tumhe tu tera teri yeh ye yah woh wo vo vah unka unki unke
usko use uska uski uske inka inki inke isko ise iska iski iske ka ki ke ko se par pe tak liye lie wala wali wale
aur ya lekin magar par bhi toh to hi nahi nahin na mat bas sirf bahut bohot bahot zyada jyada kam thoda thodi
accha acha achha theek thik sahi galat bilkul zaroor jaroor shayad abhi ab phir fir pehle baad saath sath bina
karna karne karo karu karun karta karti karte kar kiya kiye kiyi karke ho hona hone hota hoti hote hua hui hue
dena dene do de diya diye dijiye dena lena lene lo le liya chahiye chahta chahti chahte sakta sakti sakte sake
batao bataiye bataye batana samjhao samjha samajh pata jaan jaana jaise waise aisa aise kaisa yahan wahan jab
tab agar matlab yaani bhai yaar ji haan han nahi kyunki isliye taaki jaisa sabse sab kuchh apna apni apne log
logon cheez cheezein tarika tareeka tarike baat baatein kaam paisa paise madad sawal jawab dost ghar zindagi
""".split())
# Romanized Hindi words that are also common English words would let plain English pass the gate
# ("I want to do this" -> to, do). Excluded; caught by tests/test_curate.py.
HINDI_LEXICON -= {"to", "do", "me", "main", "mai", "par", "hi", "ho", "use", "the", "tab", "log", "logon", "na",
                  "de", "le", "lo", "mat", "kam", "bas", "hum", "ye", "jaise"}
_WORD = re.compile(r"[A-Za-z']+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def lexicon_share(text: str) -> float:
    words = [w.lower() for w in _WORD.findall(text)]
    return sum(w in HINDI_LEXICON for w in words) / len(words) if words else 0.0


def hinglish_ok(text: str, cos: float) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty"
    if _DEVANAGARI.search(text):
        return False, "devanagari"
    share = lexicon_share(text)
    if share == 0:
        return False, "pure_english"
    if share < MIN_LEXICON:
        return False, "low_hindi_lexicon"
    if cos < MIN_COS:
        return False, "low_cos"
    return True, "ok"


def variants(row: dict) -> list[str]:
    """v1, v2 from the first generation; v3, v4 from a stricter regeneration of failing turns."""
    return sorted((k for k in row if re.fullmatch(r"v\d+", k)), key=lambda k: int(k[1:]))


def pick_hinglish(row: dict) -> tuple[str | None, dict]:
    """Choose the passing variant with the higher fidelity cosine. Fidelity (spec 7.3) is cos(source, English
    back-translation) = `cos_back_v*`; a variant never back-translated fails as `no_back`. The direct
    LaBSE cos(source, Hinglish) is kept only as a diagnostic: LaBSE is not trained on romanized Hindi."""
    cands = []
    for v in variants(row):
        cos = row.get(f"cos_back_{v}")
        ok, why = hinglish_ok(row[v], cos) if cos is not None else (False, "no_back")
        if why == "no_back":
            ok2, why2 = hinglish_ok(row[v], 1.0)
            why = why if ok2 else why2
        cands.append((ok, cos if cos is not None else 0.0, v, why))
    passing = sorted([c for c in cands if c[0]], key=lambda c: -c[1])
    diag = {c[2]: {"cos": c[1], "why": c[3], "lex": round(lexicon_share(row[c[2]]), 3)} for c in cands}
    return (row[passing[0][2]], {"chosen": passing[0][2], **diag}) if passing else (None, diag)


def translation_ok(row: dict) -> tuple[bool, str]:
    if row["cos"] < MIN_COS:
        return False, "low_cos"
    if row["script_share"] < MIN_SCRIPT:
        return False, "wrong_script"
    return True, "ok"


def assemble(seeds: list[dict], translations: list[dict], hinglish: list[dict]) -> tuple[list[dict], dict]:
    """Return (records for en/hi/hinglish/bn, report). Pure: no I/O."""
    tr: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for r in translations:
        tr[(r["seed_id"], r["lang"])][r["turn"]] = r
    hg: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in hinglish:
        hg[r["seed_id"]][r["turn"]] = r

    out: list[dict] = []
    report: dict = {"pass": Counter(), "fail": Counter(), "fail_reasons": Counter(), "failed": defaultdict(list)}
    for s in seeds:
        out.append(s)  # English source passes trivially
        report["pass"]["en"] += 1
        langs = ["hi", "hinglish"] + (["bn"] if s["split"] != "train" else [])
        for lang in langs:
            n = len(s["user_turns"])
            turns, cos_min, reasons = [], 1.0, []
            for t in range(1, n + 1):
                if lang == "hinglish":
                    row = hg.get(s["seed_id"], {}).get(t)
                    if row is None:
                        reasons.append("missing")
                        break
                    text, diag = pick_hinglish(row)
                    if text is None:
                        reasons.append(min((d["why"] for d in diag.values()), default="fail"))
                        break
                    turns.append(text)
                    cos_min = min(cos_min, diag[diag["chosen"]]["cos"])
                else:
                    row = tr.get((s["seed_id"], lang), {}).get(t)
                    if row is None:
                        reasons.append("missing")
                        break
                    ok, why = translation_ok(row)
                    if not ok:
                        reasons.append(why)
                        break
                    turns.append(row["text"])
                    cos_min = min(cos_min, row["cos"])
            if reasons:
                report["fail"][lang] += 1
                report["fail_reasons"][f"{lang}:{reasons[0]}"] += 1
                report["failed"][lang].append(s["seed_id"])
                continue
            rec = dict(s)
            rec.update(conv_id=f"{s['seed_id']}-{lang}", language=lang, user_turns=turns)
            rec["provenance"] = {**s["provenance"], "translator": ("cohere/command-a-03-2025" if lang == "hinglish"
                                                                  else "facebook/nllb-200-distilled-1.3B"),
                                 "fidelity_min_cos": round(cos_min, 4), "reviewed": False}
            out.append(validate_record(rec))
            report["pass"][lang] += 1
    return out, report


HUMAN_FIXES = "human_fixes.jsonl"


def apply_human_fixes(recs: list[dict], fixes: list[dict]) -> list[dict]:
    """Replace reviewer-approved user turns (spec 7.3 HUMAN spot check) before the freeze.

    Each fix: {seed_id, language, turn, old, new, reason, approved, reviewer}. Only `approved` fixes apply, and only
    if the current text equals `old` (a stale fix raises instead of silently editing the wrong turn). The record keeps
    an audit trail in provenance["human_fixes"]. Returns the approved fixes that were applied.
    """
    by_key = {(r["seed_id"], r["language"]): r for r in recs}
    applied = []
    for fx in fixes:
        if not fx.get("approved"):
            continue
        rec = by_key.get((fx["seed_id"], fx["language"]))
        if rec is None:
            raise ValueError(f"human fix for missing record {fx['seed_id']}/{fx['language']}")
        i = fx["turn"] - 1
        if rec["user_turns"][i] != fx["old"]:
            raise ValueError(f"stale human fix {fx['seed_id']}/{fx['language']} turn {fx['turn']}")
        rec["user_turns"][i] = fx["new"]
        rec["provenance"].setdefault("human_fixes", []).append(
            {"turn": fx["turn"], "reason": fx["reason"], "reviewer": fx.get("reviewer")})
        applied.append(fx)
    return applied


def pass_rates(report: dict) -> dict[str, str]:
    return {lang: f"{report['pass'][lang]}/{report['pass'][lang] + report['fail'][lang]} "
                  f"({100 * report['pass'][lang] / max(1, report['pass'][lang] + report['fail'][lang]):.1f}%)"
            for lang in ("en", "hi", "hinglish", "bn")}


def main(args) -> int:
    """Local: apply the gate. Expects runs/<translate run>/translations.jsonl and hinglish_scored.jsonl paths in
    data/generated/translations.jsonl and data/generated/hinglish_scored.jsonl (copied there after pull)."""
    seeds = [r for p in ("seeds_attack.jsonl", "seeds_overrefusal.jsonl", "seeds_benign.jsonl")
             if (GEN / p).exists() for r in read_jsonl(GEN / p)]
    translations = list(read_jsonl(GEN / "translations.jsonl"))
    hinglish = list(read_jsonl(GEN / "hinglish_scored.jsonl"))
    back_path = GEN / "hinglish_back_scored.jsonl"
    back: dict[tuple[str, int], dict] = {}
    for b in (read_jsonl(back_path) if back_path.exists() else []):
        back.setdefault((b["seed_id"], b["turn"]), {})[f"cos_back_{b['variant']}"] = b["cos_back"]
    for r in hinglish:
        r.update(back.get((r["seed_id"], r["turn"]), {}))
    recs, report = assemble(seeds, translations, hinglish)
    fixes = list(read_jsonl(GEN / HUMAN_FIXES)) if (GEN / HUMAN_FIXES).exists() else []
    applied = apply_human_fixes(recs, fixes)
    print(f"human fixes: {len(applied)} applied, {len(fixes) - len(applied)} not approved")
    write_jsonl(GEN / "conversations.jsonl", recs)
    rates = pass_rates(report)
    summary = {"pass_rates": rates, "fail_reasons": dict(report["fail_reasons"]),
               "failed": {k: v for k, v in report["failed"].items()},
               "by_split_lang": Counter(f"{r['split']}/{r['language']}/{r['kind']}" for r in recs)}
    (GEN / "fidelity_report.json").write_text(json.dumps(summary, indent=2, default=list), encoding="utf-8")
    print("fidelity-gate pass rate per language:", json.dumps(rates))
    print("fail reasons:", dict(report["fail_reasons"]))
    for k, v in sorted(summary["by_split_lang"].items()):
        print(f"  {k:32s} {v}")
    return 0


def main_score(args) -> int:  # pragma: no cover - job
    from sentence_transformers import SentenceTransformer

    rows = list(read_jsonl("data/generated/hinglish_raw.jsonl"))
    if args.limit:
        rows = rows[: args.limit]
    emb = SentenceTransformer("sentence-transformers/LaBSE")
    enc = lambda xs: emb.encode(xs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)  # noqa: E731
    s, a, b = enc([r["src"] for r in rows]), enc([r["v1"] for r in rows]), enc([r["v2"] for r in rows])
    for r, x, y, z in zip(rows, s, a, b, strict=True):
        r["cos_v1"] = round(float((x * y).sum()), 4)
        r["cos_v2"] = round(float((x * z).sum()), 4)
    out = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-score")) / "hinglish_scored.jsonl"
    print(f"[score-hinglish] wrote {write_jsonl(out, rows)} rows to {out}")
    return 0


def main_score_back(args) -> int:  # pragma: no cover - job
    """cos(source, English back-translation) for data/generated/hinglish_back.jsonl -> $RUN_DIR/hinglish_back_scored.jsonl"""
    from sentence_transformers import SentenceTransformer

    rows = list(read_jsonl("data/generated/hinglish_back.jsonl"))
    if args.limit:
        rows = rows[: args.limit]
    prev_path = Path("data/generated/hinglish_back_scored.jsonl")  # optional input: reuse earlier scores
    key = lambda r: (r["seed_id"], r["turn"], r["variant"], r["back"])  # noqa: E731
    prev = {key(r): r["cos_back"] for r in read_jsonl(prev_path)} if prev_path.exists() else {}
    for r in rows:
        if key(r) in prev:
            r["cos_back"] = prev[key(r)]
    todo = [r for r in rows if "cos_back" not in r]
    print(f"[score-hinglish-back] {len(rows)} rows, {len(todo)} to encode", flush=True)
    emb = SentenceTransformer("sentence-transformers/LaBSE")
    enc = lambda xs: emb.encode(xs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)  # noqa: E731
    if todo:
        s, b = enc([r["src"] for r in todo]), enc([r["back"] for r in todo])
        for r, x, y in zip(todo, s, b, strict=True):
            r["cos_back"] = round(float((x * y).sum()), 4)
    out = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-score")) / "hinglish_back_scored.jsonl"
    print(f"[score-hinglish-back] wrote {write_jsonl(out, rows)} rows to {out}")
    return 0
