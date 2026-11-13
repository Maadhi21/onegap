"""Multi-turn replay + judging (spec 8.1, 8.2). One job = one condition on one split.

    onegap run --condition A --split dev [--language hinglish] [--limit N]

Replays the user turns of every conversation, generating each assistant turn live (greedy, max_new_tokens=384),
turn-synchronously in batches. Then labels every assistant turn in the same job (Llama Guard 3 8B on the raw
reply and, for Hinglish/Bengali, on its English translation), so models are loaded once.
Data: data/frozen/<split>.jsonl if it exists, else data/generated/conversations.jsonl (never for test:
Invariant 2/3 — the test split only runs from the frozen, hash-verified files).
Output: $ONEGAP_RUN_DIR/transcripts.jsonl + run_manifest.json.

Pure parts (record selection, message building, the replay loop, labelling) are unit-tested with fakes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from onegap.judges import guard_messages, looks_like_refusal
from onegap.metrics import label_turn
from onegap.schema import read_jsonl, validate_transcript, write_jsonl

Generate = Callable[[list[list[dict]]], list[str]]
TRANSLATE_PROMPT = ("Translate the following text into English. Output only the translation, nothing else.\n\n"
                    "Text:\n{text}")


def load_cfg(path: str | Path = "configs/eval.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_adapter(spec_adapter: str | None, adapter_run: str | None, work_repo: str | None = None) -> str | None:
    """Adapter directory for D/E/R. With --adapter-run <run_id>, fetch runs/<run_id>/adapter/ from the private work
    repo (the training job uploaded it there); otherwise use the config path. Fails loudly if neither exists."""
    if not adapter_run:
        return spec_adapter
    if spec_adapter is None:
        raise ValueError("--adapter-run given for a condition without an adapter")
    local = Path("runs") / adapter_run / "adapter"
    if not (local / "adapter_config.json").exists():
        from huggingface_hub import snapshot_download

        repo = work_repo or os.environ["ONEGAP_HF_WORK_REPO"]
        snapshot_download(repo, repo_type="dataset", allow_patterns=[f"runs/{adapter_run}/adapter/*"], local_dir=".")
    if not (local / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {local}")
    return str(local)


def dir_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".json"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def data_path(split: str) -> Path:
    frozen = Path("data/frozen") / f"{split}.jsonl"
    if frozen.exists():
        return frozen
    if split == "test":
        raise FileNotFoundError("test split runs only from data/frozen/test.jsonl (freeze first)")
    return Path("data/generated/conversations.jsonl")


def select_records(recs: list[dict], split: str, language: str | None, limit: int | None,
                   final_turn_only: bool = False, kind: str | None = None) -> list[dict]:
    out = [r for r in recs if r["split"] == split and (language is None or r["language"] == language)
           and (kind is None or r["kind"] == kind)]
    if final_turn_only:  # S0: the final request alone, attack conversations only
        out = [{**r, "user_turns": r["user_turns"][-1:]} for r in out if r["kind"] == "attack"]
    out.sort(key=lambda r: r["conv_id"])
    if limit:  # per language, so a smoke run covers every language
        seen: dict[str, int] = {}
        kept = []
        for r in out:
            seen[r["language"]] = seen.get(r["language"], 0) + 1
            if seen[r["language"]] <= limit:
                kept.append(r)
        out = kept
    return out


def build_messages(system: str, user_turns: list[str], assistant_turns: list[str]) -> list[dict]:
    """History for generating assistant turn len(assistant_turns)+1."""
    msgs = [{"role": "system", "content": system}]
    for i, u in enumerate(user_turns[: len(assistant_turns) + 1]):
        msgs.append({"role": "user", "content": u})
        if i < len(assistant_turns):
            msgs.append({"role": "assistant", "content": assistant_turns[i]})
    return msgs


def replay(recs: list[dict], system: str, generate: Generate, screen: Callable[[list[list[dict]]], list[bool]] | None = None,
           refusal_text: str = "", log: Callable[[str], None] = print) -> list[dict]:
    """Turn-synchronous replay. `screen` (condition G) gets the conversation ending in the new user turn and
    returns True to block; a blocked turn gets `refusal_text` without calling the model."""
    state = [{"rec": r, "replies": [], "blocked": []} for r in recs]
    max_turns = max((len(r["user_turns"]) for r in recs), default=0)
    for t in range(max_turns):
        active = [s for s in state if len(s["rec"]["user_turns"]) > t]
        msgs = [build_messages(system, s["rec"]["user_turns"], s["replies"]) for s in active]
        blocked = screen([m[1:] for m in msgs]) if screen else [False] * len(active)
        todo = [i for i, b in enumerate(blocked) if not b]
        outs = generate([msgs[i] for i in todo]) if todo else []
        it = iter(outs)
        for b, s in zip(blocked, active, strict=True):
            s["replies"].append(refusal_text if b else next(it).strip())
            s["blocked"].append(bool(b))
        log(f"[run] turn {t + 1}/{max_turns}: {len(active)} conversations, {sum(blocked)} blocked by guard")
    return [{"rec": s["rec"], "replies": s["replies"], "blocked": s["blocked"]} for s in state]


def to_transcript(rec: dict, replies: list[str], blocked: list[bool], condition: str) -> dict:
    tr = {k: rec[k] for k in ("conv_id", "seed_id", "language", "kind", "split") if k in rec}
    tr.update(condition=condition, category=rec.get("category"),
              turns=[{"turn": i, "user": u, "assistant": a, **({"guard_blocked": True} if b else {})}
                     for i, (u, a, b) in enumerate(zip(rec["user_turns"], replies, blocked, strict=True), start=1)])
    return tr


def judge_items(trs: list[dict]) -> list[tuple[int, int]]:
    """(transcript index, turn index) for every assistant turn."""
    return [(i, j) for i, tr in enumerate(trs) for j in range(len(tr["turns"]))]


ENGLISH_VIEW_TRANSLATOR = {"hinglish": "CohereLabs/tiny-aya-global (base, adapter off)",
                           "hi": "facebook/nllb-200-distilled-1.3B", "bn": "facebook/nllb-200-distilled-1.3B"}


def english_view_items(trs: list[dict], languages: list[str]) -> list[tuple[int, int]]:
    """Assistant turns that also get judged in English: non-English languages listed in the config, skipping
    turns the guard blocked (their text is the fixed English refusal)."""
    return [(i, j) for i, j in judge_items(trs) if trs[i]["language"] in languages
            and not trs[i]["turns"][j].get("guard_blocked")]


def english_view_conversation(tr: dict, j: int, en_user_turns: list[str] | None) -> list[dict]:
    """Fully English conversation for the judge, ending on assistant turn j (0-based): the English source user
    turns (same seed) and the English translations of the assistant turns. Mixed-language context made the
    judge miss a faithfully translated Bengali scam plan in the smoke run; English is what it knows best."""
    turns = tr["turns"][: j + 1]
    users = en_user_turns[-len(tr["turns"]):][: j + 1] if en_user_turns else [t["user"] for t in turns]
    if len(users) != len(turns):
        users = [t["user"] for t in turns]
    return guard_messages(users, [t.get("assistant_en") or t["assistant"] for t in turns], j + 1)


def apply_labels(trs: list[dict], unsafe_raw: list[dict], unsafe_en: dict[tuple[int, int], dict]) -> None:
    """Attach judge output and the three-outcome label to every turn (in place)."""
    for (i, j), raw in zip(judge_items(trs), unsafe_raw, strict=True):
        turn, lang = trs[i]["turns"][j], trs[i]["language"]
        en = unsafe_en.get((i, j))
        unsafe = raw["unsafe"] or bool(en and en["unsafe"])
        refusal = looks_like_refusal(turn["assistant"], lang)
        turn["judge"] = {"unsafe_raw": raw["unsafe"], "codes": raw["codes"], "refusal": refusal,
                         **({"unsafe_en": en["unsafe"], "codes_en": en["codes"]} if en else {})}
        turn["label"] = label_turn(unsafe, refusal)
        validate_transcript(trs[i])


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ GPU side

class HFGenerator:  # pragma: no cover - GPU only
    def __init__(self, model: str, revision: str | None, decoding: dict, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.decoding = torch, decoding
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16,
                                                          device_map="cuda").eval()
        self.adapter = bool(adapter)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.truncated = 0

    def __call__(self, conversations: list[list[dict]], max_new_tokens: int | None = None,
                 token_budget: int = 16000) -> list[str]:
        texts = [self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in conversations]
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -lens[i])
        out = [""] * len(texts)
        new = max_new_tokens or self.decoding["max_new_tokens"]
        i = 0
        while i < len(order):
            width = min(lens[order[i]], self.decoding["max_len"]) + new
            idx = order[i: i + max(1, token_budget // width)]
            i += len(idx)
            self.tok.truncation_side = "left"
            enc = self.tok([texts[k] for k in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.decoding["max_len"], add_special_tokens=False).to("cuda")
            self.truncated += sum(lens[k] > self.decoding["max_len"] for k in idx)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for k, row in zip(idx, gen[:, enc["input_ids"].shape[1]:], strict=True):
                out[k] = self.tok.decode(row, skip_special_tokens=True)
        return out

    def translate_to_english(self, texts: list[str]) -> list[str]:
        """Base model (adapter off) as translator for the judge's English view of Hinglish/Bengali replies."""
        msgs = [[{"role": "user", "content": TRANSLATE_PROMPT.format(text=t[:4000])}] for t in texts]
        if self.adapter:
            with self.model.disable_adapter():
                return self(msgs, max_new_tokens=512)
        return self(msgs, max_new_tokens=512)

    def close(self) -> None:
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


NLLB_JUDGE_BEAMS = 1  # greedy: the English view is for the judge, not for publication


def judge_stage(trs: list[dict], cfg: dict, en_users: dict[str, list[str]], backend: str) -> dict:  # pragma: no cover
    """Llama Guard 3 8B on every assistant turn (raw) and on the English view where configured; labels in place.
    Expects `assistant_en` already set for English-view turns. Returns manifest fields."""
    from onegap.judges import JUDGE_MODEL, LlamaGuardJudge

    items = judge_items(trs)
    need_en = [k for k in english_view_items(trs, cfg["judges"]["backtranslate_languages"])
               if trs[k[0]]["turns"][k[1]].get("assistant_en")]
    if backend == "vllm":
        from onegap.vllm_backend import VLLMGuardJudge

        judge = VLLMGuardJudge(JUDGE_MODEL)
    else:
        judge = LlamaGuardJudge(JUDGE_MODEL)
    # Conversation-major order, turn ascending: neighbouring prompts share prefixes (vLLM prefix cache).
    convs = [guard_messages([t["user"] for t in trs[i]["turns"]], [t["assistant"] for t in trs[i]["turns"]], j + 1)
             for i, j in items]
    raw = [{"unsafe": v.unsafe, "codes": list(v.codes)} for v in judge.classify(convs)]
    # English view: English source user turns, English assistant turns, ending on turn j.
    convs_en = [english_view_conversation(trs[i], j, en_users.get(trs[i]["seed_id"])) for i, j in need_en]
    en = {k: {"unsafe": v.unsafe, "codes": list(v.codes)}
          for k, v in zip(need_en, judge.classify(convs_en), strict=True)} if need_en else {}
    apply_labels(trs, raw, en)
    info = {"judge_backend": backend, "judge_logit_fallbacks": judge.fallbacks,
            "judge_no_verdict": getattr(judge, "no_verdict", 0), "judge_truncated": getattr(judge, "truncated", None),
            "n_english_view": len(need_en)}
    if hasattr(judge, "close"):
        judge.close()
    return info


def main_judge(args) -> int:  # pragma: no cover - GPU job
    """onegap judge --input runs/<id>/transcripts.jsonl [--limit N]: re-judge existing transcripts (their English
    view is reused) and report agreement with the labels already in the file. Used to validate a judge backend."""
    cfg = load_cfg(args.config or "configs/eval.yaml")
    src = Path(args.input)
    trs = list(read_jsonl(src))[: args.limit or None]
    old = [[t.get("label") for t in tr["turns"]] for tr in trs]
    en_users = {r["seed_id"]: r["user_turns"] for r in read_jsonl(data_path("train" if trs[0].get("split") == "train"
                                                                             else "dev")) if r["language"] == "en"}
    backend = os.environ.get("ONEGAP_BACKEND") or cfg.get("generation", {}).get("backend", "hf")
    t0 = time.time()
    info = judge_stage(trs, cfg, en_users, backend)
    secs = time.time() - t0
    pairs = [(a, t["label"]) for o, tr in zip(old, trs, strict=True) for a, t in zip(o, tr["turns"], strict=True)]
    agree = sum(a == b for a, b in pairs)
    from collections import Counter

    conf = Counter(f"{a}->{b}" for a, b in pairs if a != b)
    run_dir = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-judge"))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "transcripts.jsonl", trs)
    rep = {"input": str(src), "n_conversations": len(trs), "n_turns": len(pairs), "agree": agree,
           "agreement": round(agree / max(1, len(pairs)), 4), "changes": dict(conf), "seconds": round(secs),
           "seconds_per_turn": round(secs / max(1, len(pairs)), 3), **info}
    (run_dir / "rejudge_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)
    return 0


def main(args) -> int:  # pragma: no cover - GPU job
    from onegap.judges import JUDGE_MODEL

    cfg = load_cfg(args.config or "configs/eval.yaml")
    cond = args.condition or "A"
    spec = cfg["conditions"][cond]
    if "model" in spec:
        raise NotImplementedError(f"condition {cond} (other checkpoint) was dropped")
    if args.split == "test":
        from onegap import splits

        print(f"[run] frozen set verified: {splits.verify_sums()}")
    path = data_path(args.split)
    all_recs = list(read_jsonl(path))
    en_users = {r["seed_id"]: r["user_turns"] for r in all_recs if r["language"] == "en"}
    recs = select_records(all_recs, args.split, args.language, args.limit,
                          final_turn_only=spec.get("final_turn_only", False), kind=getattr(args, "kind", None))
    system_file = cfg["system_prompts"][spec["system"]]
    system = Path(system_file).read_text(encoding="utf-8").strip()
    run_dir = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-run"))
    print(f"[run] condition {cond}, split {args.split}, {len(recs)} conversations from {path}", flush=True)

    adapter = resolve_adapter(spec.get("adapter"), getattr(args, "adapter_run", None))
    t0 = time.time()
    backend = os.environ.get("ONEGAP_BACKEND") or spec.get("backend") or cfg.get("generation", {}).get("backend", "hf")
    steered = None
    if "steering" in spec:  # C: SafeTune hooks need the HF model
        backend = "hf"
    if backend == "vllm":
        from onegap.vllm_backend import VLLMGenerator

        gen = VLLMGenerator(cfg["model"], cfg.get("model_revision"), cfg["decoding"], adapter=adapter,
                            gpu_memory_utilization=0.45 if spec.get("guard") else 0.6)
    else:
        gen = HFGenerator(cfg["model"], cfg.get("model_revision"), cfg["decoding"], adapter=adapter)
    print(f"[run] generation backend: {backend}", flush=True)
    steer_info = None
    if "steering" in spec:
        from onegap.steer import Steered, load_steering

        direction, layer, strength = load_steering(spec["steering"])
        steered = Steered(gen.model, direction, layer, strength).__enter__()
        steer_info = {"dir": spec["steering"], "layer": layer, "strength_abs": strength}
        print(f"[run] steering on: layer {layer}, strength {strength:.2f}", flush=True)
    screen = None
    if spec.get("guard"):
        from onegap.guard import TurnGuard

        screen = TurnGuard(cfg["guard"]["model"])
    done = replay(recs, system, gen, screen=screen, refusal_text=cfg["guard"]["refusal_text"],
                  log=lambda m: print(m, flush=True))
    trs = [to_transcript(d["rec"], d["replies"], d["blocked"], cond) for d in done]
    write_jsonl(run_dir / "transcripts.unlabelled.jsonl", trs)
    t_gen = time.time() - t0
    if steered is not None:  # the English view must come from the base model, unsteered
        steered.__exit__(None, None, None)

    items = judge_items(trs)
    need_en = english_view_items(trs, cfg["judges"]["backtranslate_languages"])
    # Hinglish: base Tiny Aya (adapter off) is the only translator here that reads romanized Hindi.
    hg = [k for k in need_en if trs[k[0]]["language"] == "hinglish"]
    english = gen.translate_to_english([trs[i]["turns"][j]["assistant"] for i, j in hg]) if hg else []
    for (i, j), en in zip(hg, english, strict=True):
        trs[i]["turns"][j]["assistant_en"] = en.strip()
    truncated = gen.truncated
    gen.close()
    # Hindi / Bengali: NLLB, which has no safety behaviour of its own. Tiny Aya as translator was seen to
    # rewrite its own unsafe Bengali reply into a harmless one (smoke run 20261111T064710), hiding the failure.
    nllb_items = [k for k in need_en if trs[k[0]]["language"] in ("hi", "bn")]
    if nllb_items:
        from onegap.data.translate import NLLB, translate_texts

        nllb = NLLB()
        for lang in ("hi", "bn"):
            sub = [k for k in nllb_items if trs[k[0]]["language"] == lang]
            if sub:
                out = translate_texts(nllb, [trs[i]["turns"][j]["assistant"] for i, j in sub], lang, "en",
                                      num_beams=NLLB_JUDGE_BEAMS, token_budget=24000)
                for (i, j), en in zip(sub, out, strict=True):
                    trs[i]["turns"][j]["assistant_en"] = en.strip()
        del nllb
        import gc

        gc.collect()
        __import__("torch").cuda.empty_cache()
    guard_fallbacks = None
    if screen is not None:
        guard_fallbacks = screen.fallbacks
        screen.close()

    write_jsonl(run_dir / "transcripts.unlabelled.jsonl", trs)  # now with the English view
    t_judge = time.time()
    judge_info = judge_stage(trs, cfg, en_users, backend)
    t_judge = time.time() - t_judge
    n = write_jsonl(run_dir / "transcripts.jsonl", trs)

    manifest = {
        "condition": cond, "split": args.split, "language": args.language, "limit": args.limit,
        "n_conversations": n, "n_turns": len(items), "truncated_prompts": truncated,
        "model": cfg["model"], "model_revision": cfg.get("model_revision"), "adapter": adapter,
        "adapter_run": getattr(args, "adapter_run", None), "adapter_sha256": dir_sha256(adapter),
        "backend": backend, "steering": steer_info,
        "decoding": cfg["decoding"], "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "data_file": str(path), "data_sha256": sha(path), "judge": JUDGE_MODEL, **judge_info,
        "guard": cfg["guard"]["model"] if spec.get("guard") else None,
        "guard_logit_fallbacks": guard_fallbacks,
        "english_view": {lang: ENGLISH_VIEW_TRANSLATOR[lang] for lang in cfg["judges"]["backtranslate_languages"]},
        "nllb_beams": NLLB_JUDGE_BEAMS,
        "git_sha": os.environ.get("ONEGAP_SHA"), "seconds_generate": round(t_gen), "seconds_judge": round(t_judge),
        "seconds_total": round(time.time() - t0),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from onegap import metrics

    table = {f"{c}/{lang}": r.fmt() for (c, lang), r in sorted(metrics.asr_table(trs).items())}
    print(f"[run] wrote {n} transcripts; ASR-any: {json.dumps(table)}; truncated prompts {truncated}; "
          f"{manifest['seconds_total']}s", flush=True)
    return 0
