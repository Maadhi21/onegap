"""M1 translation job (spec 7.2): NLLB-200 distilled 1.3B for Hindi and Bengali, user turns only.

    launch: scripts/launch_job.py translate --flavor l4x1 --venv core
            --inputs data/generated/seeds_attack.jsonl data/generated/seeds_overrefusal.jsonl [-- --limit N]

Plan per record (Invariant 1: Bengali never in train):
  train -> hi ; dev/test -> hi, bn ; over-refusal (dev/test) -> hi, bn
For each segment: forward translation, back-translation to English, LaBSE cosine(source, back), script share.
Output: $ONEGAP_RUN_DIR/translations.jsonl  rows {seed_id, lang, turn, src, text, back, cos, script_share}
The fidelity gate itself (curate.py) runs locally on this file so thresholds can be changed without a GPU.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from onegap.schema import read_jsonl, write_jsonl

MODEL = "facebook/nllb-200-distilled-1.3B"
EMBED = "sentence-transformers/LaBSE"
NLLB_CODE = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng"}
SCRIPT_RX = {"hi": re.compile(r"[ऀ-ॿ]"), "bn": re.compile(r"[ঀ-৿]")}
INPUTS = ("data/generated/seeds_attack.jsonl", "data/generated/seeds_overrefusal.jsonl",
          "data/generated/seeds_benign.jsonl")  # only the files uploaded with --inputs exist in the job


def plan_languages(rec: dict) -> list[str]:
    if rec["split"] == "train":
        return ["hi"]
    return ["hi", "bn"]


def split_sentences(text: str, max_chars: int = 600) -> list[str]:
    """NLLB degrades on long inputs: split on sentence ends / newlines into chunks <= max_chars."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+|\n+", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text]


def script_share(text: str, lang: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(bool(SCRIPT_RX[lang].match(c)) for c in letters) / len(letters) if letters else 0.0


class NLLB:  # pragma: no cover - GPU only
    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    def __call__(self, texts: list[str], src: str, tgt: str, batch_size: int = 32, num_beams: int = 4,
                 token_budget: int | None = None) -> list[str]:
        """Data translation uses beam 4, batch 32. The judge's English view (runner) uses greedy decoding with
        batches capped at `token_budget` source tokens x beams: beam 4 at batch 32 hit OOM retries on an L4
        and took ~60 min for one Hindi train replay (20261111T120623)."""
        self.tok.src_lang = NLLB_CODE[src]
        bos = self.tok.convert_tokens_to_ids(NLLB_CODE[tgt])
        lens = [len(self.tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
        out = [""] * len(texts)
        if token_budget:
            from onegap.judges import length_batches

            batches = length_batches([min(n, 400) * num_beams for n in lens], token_budget)
        else:
            order = sorted(range(len(texts)), key=lambda i: lens[i])
            batches = [order[b: b + batch_size] for b in range(0, len(order), batch_size)]
        for idx in batches:
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=400).to("cuda")
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=512, num_beams=num_beams)
            for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                out[i] = t
        return out


def translate_texts(nllb, texts: list[str], src: str, tgt: str, **kw) -> list[str]:  # pragma: no cover
    """Chunk long texts into sentences, translate all chunks in one batched pass, re-join."""
    chunks, owner = [], []
    for i, t in enumerate(texts):
        for c in split_sentences(t):
            chunks.append(c)
            owner.append(i)
    done = nllb(chunks, src, tgt, **kw)
    joined: list[list[str]] = [[] for _ in texts]
    for i, c in zip(owner, done, strict=True):
        joined[i].append(c)
    return [" ".join(j) for j in joined]


def main(args) -> int:  # pragma: no cover - GPU job
    from sentence_transformers import SentenceTransformer

    recs = [r for p in INPUTS if Path(p).exists() for r in read_jsonl(p)]
    if args.limit:
        recs = recs[: args.limit]
    segs = [(r["seed_id"], lang, t, turn) for r in recs for lang in plan_languages(r)
            for t, turn in enumerate(r["user_turns"], start=1)]
    print(f"[translate] {len(recs)} records -> {len(segs)} segments", flush=True)
    nllb = NLLB()
    rows = []
    t0 = time.time()
    for lang in ("hi", "bn"):
        sub = [s for s in segs if s[1] == lang]
        if not sub:
            continue
        fwd = translate_texts(nllb, [s[3] for s in sub], "en", lang)
        back = translate_texts(nllb, fwd, lang, "en")
        for (sid, _, t, src), f, b in zip(sub, fwd, back, strict=True):
            rows.append({"seed_id": sid, "lang": lang, "turn": t, "src": src, "text": f, "back": b,
                         "script_share": round(script_share(f, lang), 3)})
        print(f"[translate] {lang}: {len(sub)} segments, {time.time() - t0:.0f}s elapsed", flush=True)
    emb = SentenceTransformer(EMBED, device="cuda")
    a = emb.encode([r["src"] for r in rows], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    b = emb.encode([r["back"] for r in rows], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    for r, x, y in zip(rows, a, b, strict=True):
        r["cos"] = round(float((x * y).sum()), 4)
    out = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-translate")) / "translations.jsonl"
    n = write_jsonl(out, rows)
    print(f"[translate] wrote {n} rows to {out} in {time.time() - t0:.0f}s", flush=True)
    return 0
