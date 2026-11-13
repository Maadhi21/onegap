"""Locating refusal (spec 8.4): CircuitKIT custom contrastive refusal task per language, EAP-IG discovery,
faithfulness, export, cross-language overlap matrix, and the component sets for repairs E and R.

    onegap circuit-data            (local) choose prompts -> data/generated/circuit_prompts.jsonl
    onegap circuits [--limit N]    (job, circuit venv, l40sx1) -> $RUN_DIR/{scores_<lang>.json, overlap.json,
                                   selected_components.json, faithfulness_<lang>.json, ck_pairs_<lang>.csv}

Task (per language): clean = a harmful request from the train attack set (final user turn of a conversation on
which the base model went unsafe); corrupt = a benign sensitive request of the same category from the train
benign set. clean_answer = the first token of a refusal in that language; corrupt_answer = the model's own first
token on the corrupt prompt (measured in the job). Metric: logit difference. Histories are dropped (single user
turn) to fit memory (28 GB peak at ~430 tokens in the M0 smoke) — deviation from spec 8.4 "final two turns".

Component selection (E): per-language |node score| normalised by its max, averaged over en/hi/Hinglish, top-K
nodes. A head node A{L}.{h} selects layer L's q/k/v/o projections; an MLP node selects layer L's gate/up/down.
PEFT LoRA cannot train single heads, so a selected head trains its whole attention block (spec fallback E1).
R: the same number of attention blocks and MLP blocks, drawn at random (seed 13) from layers not in E's set.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
N_LAYERS = 36
LANGS = ("en", "hi", "hinglish")
# First token of a refusal, per language (what the model says when it does refuse; see dev S0 transcripts).
REFUSAL_START = {"en": "I", "hi": "मैं", "hinglish": "Sorry"}
_NODE = re.compile(r"^(?:A(\d+)\.(\d+)|MLP (\d+))$")


# ------------------------------------------------------------------ pure helpers (tested)

def parse_node(name: str) -> tuple[str, int]:
    """'A14.13' -> ('attn', 14); 'MLP 3' -> ('mlp', 3)."""
    m = _NODE.match(name)
    if not m:
        raise ValueError(f"unknown CircuitKIT node name {name!r}")
    return ("attn", int(m[1])) if m[1] is not None else ("mlp", int(m[3]))


def aggregate(scores_by_lang: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean over languages of |score| / max|score| (each language weighs the same)."""
    agg: dict[str, float] = {}
    for scores in scores_by_lang.values():
        top = max((abs(v) for v in scores.values()), default=0.0) or 1.0
        for k, v in scores.items():
            agg[k] = agg.get(k, 0.0) + abs(v) / top / len(scores_by_lang)
    return agg


def top_nodes(scores: dict[str, float], k: int) -> list[str]:
    return [n for n, _ in sorted(scores.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:k]]


def blocks(nodes: list[str]) -> dict[str, list[int]]:
    """Selected layers per block type."""
    out: dict[str, set[int]] = {"attn": set(), "mlp": set()}
    for n in nodes:
        kind, layer = parse_node(n)
        out[kind].add(layer)
    return {k: sorted(v) for k, v in out.items()}


def select_blocks(scores: dict[str, float], max_per_kind: int) -> tuple[dict[str, list[int]], list[str]]:
    """Walk nodes by descending score and add each node's block (attention or MLP of its layer) until each block
    type holds `max_per_kind` layers. MLP nodes are whole blocks and outscore single heads, so a plain top-K over
    nodes fills most MLP layers (smoke: top-36 -> 21/36 MLP layers) and leaves no room for a disjoint random
    control. Returns (blocks, the nodes that contributed)."""
    out: dict[str, list[int]] = {"attn": [], "mlp": []}
    used = []
    for n in top_nodes(scores, len(scores)):
        kind, layer = parse_node(n)
        if len(out[kind]) < max_per_kind and layer not in out[kind]:
            out[kind].append(layer)
            used.append(n)
        elif layer in out[kind]:
            used.append(n)
        if all(len(v) >= max_per_kind for v in out.values()):
            break
    return {k: sorted(v) for k, v in out.items()}, used


def rank_layers(per_layer: dict, k: int) -> list[int]:
    """SafeTune SafetyNeuronReport.per_layer {layer: [(unit, score), ...]} -> top-k layers by mean |score|."""
    mean = {int(L): sum(abs(s) for _, s in units) / len(units) for L, units in per_layer.items() if units}
    return sorted(sorted(mean, key=lambda L: (-mean[L], L))[:k])


def crosscheck(st_layers: list[int], e_layers: list[int], r_layers: list[int], n_layers: int = N_LAYERS) -> dict:
    """How many of SafeTune's top layers fall in E's vs R's blocks, against the chance expectation."""
    hit_e, hit_r = sorted(set(st_layers) & set(e_layers)), sorted(set(st_layers) & set(r_layers))
    return {"safetune_layers": st_layers, "in_E": hit_e, "in_R": hit_r, "n_in_E": len(hit_e), "n_in_R": len(hit_r),
            "chance": round(len(st_layers) * len(e_layers) / n_layers, 2)}


def module_names(bl: dict[str, list[int]]) -> list[str]:
    names = [f"model.layers.{L}.self_attn.{p}" for L in bl["attn"] for p in ATTN_PROJ]
    names += [f"model.layers.{L}.mlp.{p}" for L in bl["mlp"] for p in MLP_PROJ]
    return names


def random_control(bl: dict[str, list[int]], n_layers: int = N_LAYERS, seed: int = 13) -> dict[str, list[int]]:
    """Same number of attention and MLP blocks as `bl`, drawn from layers not used by `bl` for that block type."""
    rng = random.Random(seed)
    out = {}
    for kind in ("attn", "mlp"):
        pool = [L for L in range(n_layers) if L not in set(bl[kind])]
        if len(pool) < len(bl[kind]):
            raise ValueError(f"not enough free {kind} layers for a disjoint random control")
        out[kind] = sorted(rng.sample(pool, len(bl[kind])))
    return out


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def equalize(n_tokens, clean: str, corrupt: str, pads: tuple[str, ...] = (" the", " .", " a"),
             max_pad: int = 256) -> tuple[str, str, int] | None:
    """Pad the shorter user message at its end with neutral tokens until both rendered prompts have the same
    token count (CircuitKIT's 'filter' alignment needs equal lengths; its own pad_question stops at 50 tokens).
    `n_tokens(user_text)` = token length of the rendered prompt. Returns (clean, corrupt, signed pad count:
    + = corrupt padded, - = clean padded) or None if no exact match within max_pad tokens."""
    a, b = n_tokens(clean), n_tokens(corrupt)
    if a == b:
        return clean, corrupt, 0
    base, target = (corrupt, a) if b < a else (clean, b)
    if target - min(a, b) > max_pad:
        return None
    k = 0
    while n_tokens(base + pads[0] * (k + 1)) < target and k < max_pad:
        k += 1
    for tail in (pads[0], *pads[1:]):  # last step: try each pad, since merges can make one token two
        t = base + pads[0] * k + tail
        if n_tokens(t) == target:
            return (clean, t, k + 1) if b < a else (t, corrupt, -(k + 1))
    return None


def overlap_matrix(scores_by_lang: dict[str, dict[str, float]], k: int) -> dict[str, dict[str, float]]:
    tops = {lang: top_nodes(s, k) for lang, s in scores_by_lang.items()}
    return {a: {b: round(jaccard(tops[a], tops[b]), 3) for b in tops} for a in tops}


# ------------------------------------------------------------------ local: choose prompts

def build_prompts(attack_trs: list[dict], benign_recs: list[dict], per_lang: int, seed: int = 13) -> list[dict]:
    """Pairs (clean harmful request, corrupt benign request) per language, matched on category where possible."""
    rng = random.Random(seed)
    out = []
    for lang in LANGS:
        harm = [t for t in attack_trs if t["language"] == lang and t.get("kind", "attack") == "attack"
                and any(x["label"] == "unsafe" for x in t["turns"])]
        harm.sort(key=lambda t: t["seed_id"])
        rng.shuffle(harm)
        ben = [r for r in benign_recs if r["language"] == lang and r["kind"] == "benign"]
        by_cat: dict[str, list[dict]] = {}
        for r in sorted(ben, key=lambda r: r["seed_id"]):
            by_cat.setdefault(r.get("category", ""), []).append(r)
        for v in by_cat.values():
            rng.shuffle(v)
        used: set[str] = set()
        for t in harm:
            if sum(p["lang"] == lang for p in out) >= per_lang:
                break
            pool = [r for r in by_cat.get(t.get("category", ""), []) if r["seed_id"] not in used] or \
                   [r for v in by_cat.values() for r in v if r["seed_id"] not in used]
            if not pool:
                break
            b = pool[0]
            used.add(b["seed_id"])
            out.append({"lang": lang, "category": t.get("category"), "clean_seed": t["seed_id"],
                        "corrupt_seed": b["seed_id"], "clean": t["turns"][-1]["user"],
                        "corrupt": b["user_turns"][0]})
    return out


def main_data(args) -> int:
    from onegap.schema import read_jsonl, write_jsonl

    root = Path(__file__).resolve().parents[2]
    gen = root / "data" / "generated"
    runs = [root / p.strip() for p in (gen / "prefdata_runs.txt").read_text().splitlines() if p.strip()]
    trs = [t for r in runs for t in read_jsonl(r / "transcripts.jsonl")]
    recs = [r for r in read_jsonl(gen / "conversations.jsonl") if r["split"] == "train"]
    rows = build_prompts(trs, recs, per_lang=args.limit or 48)
    write_jsonl(gen / "circuit_prompts.jsonl", rows)
    from collections import Counter

    print(f"[circuit-data] {len(rows)} pairs: {dict(Counter(r['lang'] for r in rows))}")
    return 0


# ------------------------------------------------------------------ job: discovery

def main(args) -> int:  # pragma: no cover - GPU job (circuit venv)
    import csv

    import circuitkit as ck
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from onegap.schema import read_jsonl

    cfg = yaml.safe_load(Path("configs/circuits.yaml").read_text(encoding="utf-8"))
    ecfg = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))
    model_name, rev = ecfg["model"], ecfg.get("model_revision")
    run_dir = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-circuits"))
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl("data/generated/circuit_prompts.jsonl"))
    if args.limit:
        rows = [r for lang in LANGS for r in [x for x in rows if x["lang"] == lang][: args.limit]]
    system = Path("prompts/system_A.txt").read_text(encoding="utf-8").strip()
    tok = AutoTokenizer.from_pretrained(model_name, revision=rev)

    def render(user: str) -> str:
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       add_generation_prompt=True, tokenize=False)

    # 1. the model's own first token on each prompt (corrupt answer; and a check that clean prompts lean refusal)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for r in rows:
        for side in ("clean", "corrupt"):
            ids = tok(render(r[side]), return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                logits = model(**ids).logits[0, -1]
            r[f"{side}_first"] = tok.decode([int(logits.argmax())])
            ref = tok(REFUSAL_START[r["lang"]], add_special_tokens=False)["input_ids"][0]
            r[f"{side}_refusal_logit_gap"] = round(float(logits[ref] - logits.max()), 3)
    del model
    torch.cuda.empty_cache()
    (run_dir / "first_tokens.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    # 2. discovery per language
    # Equal token lengths are made here (equalize), so CircuitKIT's default 'filter' alignment keeps the pairs;
    # its pad_question inserts at most 50 tokens and dropped every pair in the smoke (run 20261111T185251).
    n_tok = lambda user: len(tok(render(user), add_special_tokens=False)["input_ids"])  # noqa: E731
    scores_by_lang: dict[str, dict[str, float]] = {}
    report: dict = {"languages": {}}
    for lang in LANGS:
        sub, pads, dropped = [], [], 0
        for r in (r for r in rows if r["lang"] == lang):
            eq = equalize(n_tok, r["clean"], r["corrupt"])
            if eq is None:
                dropped += 1
                continue
            r["clean_padded"], r["corrupt_padded"], p = eq
            pads.append(p)
            sub.append(r)
        path = run_dir / f"ck_pairs_{lang}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clean", "clean_answer", "corrupt", "corrupt_answer"])
            for r in sub:
                corrupt_answer = r["corrupt_first"] if r["corrupt_first"].strip() != REFUSAL_START[lang] else "**"
                w.writerow([render(r["clean_padded"]), REFUSAL_START[lang], render(r["corrupt_padded"]),
                            corrupt_answer])
        pipe = ck.Pipeline.from_custom_data(model_name, str(path), clean_prompt="{clean}",
                                            clean_answer="{clean_answer}", corrupt_prompt="{corrupt}",
                                            corrupt_answer="{corrupt_answer}", task_name=f"onegap_refusal_{lang}",
                                            output_dir=str(run_dir / f"ck_{lang}"))
        info: dict = {"pairs": len(sub), "dropped_unequalizable": dropped,
                      "pad_tokens": {"corrupt_padded": sum(p > 0 for p in pads), "clean_padded": sum(p < 0 for p in pads),
                                     "median_abs": sorted(abs(p) for p in pads)[len(pads) // 2] if pads else 0,
                                     "max_abs": max((abs(p) for p in pads), default=0)}}
        print(f"[circuits] {lang}: {json.dumps(info)}", flush=True)
        try:
            pipe.discover(algorithm=cfg["discover"]["algorithm"], level="node", sparsity=0.3, n_examples=len(sub),
                          batch_size=1, ig_steps=3, seed=13)
        except Exception as e:  # noqa: BLE001 - spec fallback: plain EAP
            info["eap_ig_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            pipe.discover(algorithm=cfg["discover"]["fallback"], level="node", sparsity=0.3, n_examples=len(sub),
                          batch_size=1, seed=13)
            info["algorithm"] = cfg["discover"]["fallback"]
        # (no pipe.export(): in CircuitKIT 0.2.2 it exports a pruned/quantized model, not the circuit)
        sj = next(Path(run_dir / f"ck_{lang}").rglob("*_scores.json"), None)
        data = json.loads(sj.read_text()) if sj else {}
        scores_by_lang[lang] = data.get("node_scores", {})
        (run_dir / f"scores_{lang}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        try:
            pipe.evaluate(pillars=cfg["faithfulness"]["pillars"], n_examples=min(len(sub), 32), n_stability_runs=1,
                          batch_size=1)
            fr = getattr(pipe, "faithfulness_report", None) or getattr(pipe, "report", None)
            info["faithfulness"] = json.loads(json.dumps(fr, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception as e:  # noqa: BLE001 - recorded, not fatal: E can still be built from the scores
            info["faithfulness_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        report["languages"][lang] = info
        print(f"[circuits] {lang}: {len(scores_by_lang[lang])} node scores, {info.get('peak_gpu_gb')} GB", flush=True)
        del pipe
        torch.cuda.empty_cache()

    # 3. overlap and component sets
    k = cfg.get("select", {}).get("top_k_nodes", 36)
    agg = aggregate(scores_by_lang)
    cap = cfg.get("select", {}).get("max_blocks_per_kind", 9)
    e_blocks, chosen = select_blocks(agg, cap)
    r_blocks = random_control(e_blocks)
    sel = {"top_k_nodes": k, "max_blocks_per_kind": cap, "nodes": chosen,
           "top_k_blocks": blocks(top_nodes(agg, k)), "E": {"blocks": e_blocks, "modules": module_names(e_blocks)},
           "R": {"blocks": r_blocks, "modules": module_names(r_blocks), "seed": 13},
           "path_used": "E1 (PEFT target_modules = projections of selected attention/MLP blocks)"}
    (run_dir / "selected_components.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    report["overlap_top_k"] = overlap_matrix(scores_by_lang, k)
    report["E_blocks"], report["R_blocks"] = e_blocks, r_blocks
    (run_dir / "circuits_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    print(json.dumps({"overlap": report["overlap_top_k"], "E": e_blocks, "R": r_blocks}, indent=1))
    return 0


# ------------------------------------------------------------------ job: SafeTune cross-check (spec 8.4, always run)

def main_crosscheck(args) -> int:  # pragma: no cover - GPU job (core venv: safetune)
    """onegap circuit-check --input <circuits run>/selected_components.json [--limit N]
    SafeTune Interpret (the steps of safety_circuit_info, method="weight") on the same clean/corrupt requests, passed
    as raw user text: SafeTune applies the chat template itself (user turn only, no system prompt A). Layers ranked by the mean |cosine| of their top-16 mlp.down_proj units with the refusal direction;
    the top-9 are compared with E's and R's MLP blocks."""
    import torch
    import yaml
    from safetune.instrumentation.interpret.safety_neurons import (
        SafetyNeuronConfig,
        identify_safety_neurons,
    )
    from safetune.interventions.steer.refusal_direction import (
        RefusalDirectionConfig,
        extract_refusal_direction,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from onegap.schema import read_jsonl

    ecfg = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path("configs/circuits.yaml").read_text(encoding="utf-8"))
    cap = cfg.get("select", {}).get("max_blocks_per_kind", 9)
    run_dir = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-circuit-check"))
    run_dir.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = list(read_jsonl("data/generated/circuit_prompts.jsonl"))
    if args.limit:
        rows = [r for lang in LANGS for r in [x for x in rows if x["lang"] == lang][: args.limit]]
    tok = AutoTokenizer.from_pretrained(ecfg["model"], revision=ecfg.get("model_revision"))
    model = AutoModelForCausalLM.from_pretrained(ecfg["model"], revision=ecfg.get("model_revision"),
                                                 dtype=torch.bfloat16, device_map="cuda").eval()
    out: dict = {"n_pairs": len(rows), "top_k_layers": cap, "per_language": {}}
    for lang in (*LANGS, "all"):
        sub = rows if lang == "all" else [r for r in rows if r["lang"] == lang]
        # raw user text: SafeTune applies the chat template itself (vector_extraction._fmt, user turn only)
        harmful, harmless = [r["clean"] for r in sub], [r["corrupt"] for r in sub]
        n = len(model.model.layers)
        _, picked, per_dir = extract_refusal_direction(
            model, tok, harmful, harmless, RefusalDirectionConfig(target_layers=list(range(n)), pick_layer=n // 2,
                                                                  normalize=True))
        rep = identify_safety_neurons(model, per_dir, SafetyNeuronConfig(method="weight", top_k_per_layer=16,
                                                                         target_layers=list(range(n)),
                                                                         target_module="mlp.down_proj"),
                                      tokenizer=tok, harmful_prompts=harmful, harmless_prompts=harmless)
        st = rank_layers(rep.per_layer, cap)
        res = crosscheck(st, sel["E"]["blocks"]["mlp"], sel["R"]["blocks"]["mlp"])
        res["direction_layer"] = picked
        res["layer_mean_abs"] = {int(L): round(sum(abs(s) for _, s in u) / len(u), 4)
                                 for L, u in rep.per_layer.items() if u}
        out["per_language"][lang] = res
        print(f"[circuit-check] {lang}: {json.dumps({k: v for k, v in res.items() if k != 'layer_mean_abs'})}",
              flush=True)
    (run_dir / "safetune_crosscheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0