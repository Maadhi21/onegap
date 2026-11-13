"""Steering baseline C (spec 8.7) with SafeTune's refusal direction (real 0.1.14 API, docs/API_NOTES.md).

    onegap steer --config configs/steer_C.yaml [--limit N]     (job, core venv; HF, hooks)

1. Calibrate on train prompts only: data/generated/circuit_prompts.jsonl (clean = harmful final user turns where
   base A went unsafe, corrupt = benign train requests), passed as raw text: SafeTune applies the chat template
   itself (user turn only, no prompt A). Steering is then applied with prompt A in place.
   `extract_refusal_direction` picks the layer (Arditi et al. sweep) unless configs/steer_C.yaml sets `layer`.
2. Strength unit: SafeTune's `steer` mode adds strength * unit direction to the block output. A unit vector is
   tiny next to the residual stream, so strengths are expressed in multiples of the measured
   ||mean(harmful) - mean(harmless)|| at that layer (1.0 = the activation-addition size of Arditi et al.).
3. Sweep on a small dev subset (attack turns replayed with history, plus over-refusal prompts): refusal-phrase
   rate per strength. No judge here; the sweep only locates the strength.
4. Pick: the smallest strength whose attack-refusal gain over strength 0 reaches D's dev refusal gain
   (`target_gain` in the config, filled from D's dev run); without it, the largest strength whose over-refusal
   stays within +10 points. The chosen setting goes to $RUN_DIR/steer.json + direction.pt, which the runner
   loads for condition C (HF backend).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml

from onegap.judges import looks_like_refusal
from onegap.schema import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def pick_strength(sweep: list[dict], target_gain: float | None, max_overref_gain: float = 0.10) -> dict:
    """sweep rows: {strength, attack_refusal, overref_refusal}; row with strength 0 is the baseline. Pure."""
    base = next(r for r in sweep if r["strength"] == 0)
    rows = sorted((r for r in sweep if r["strength"] > 0), key=lambda r: r["strength"])
    if target_gain is not None:
        for r in rows:
            if r["attack_refusal"] - base["attack_refusal"] >= target_gain:
                return {**r, "rule": f"first to reach D's refusal gain {target_gain:.3f}"}
        return {**rows[-1], "rule": "no strength reached D's gain; largest tried"}
    ok = [r for r in rows if r["overref_refusal"] - base["overref_refusal"] <= max_overref_gain]
    return {**(ok[-1] if ok else rows[0]), "rule": f"largest with over-refusal gain <= {max_overref_gain}"}


def refusal_rate(replies: list[str], langs: list[str]) -> float:
    return sum(looks_like_refusal(r, lang) for r, lang in zip(replies, langs, strict=True)) / max(1, len(replies))


class Steered:  # pragma: no cover - GPU only
    """Context: install SafeTune's steer hooks on an HFGenerator's model."""

    def __init__(self, model, direction, layer: int, strength: float):
        from safetune.interventions.steer.refusal_direction import RefusalDirectionModel

        self.w = RefusalDirectionModel(model, direction, mode="steer", strength=strength, layers=[layer]) \
            if strength > 0 else None

    def __enter__(self):
        if self.w:
            self.w.install()
        return self

    def __exit__(self, *exc):
        if self.w:
            self.w.remove()


def main(args) -> int:  # pragma: no cover - GPU job
    import torch
    from safetune.interventions.steer.refusal_direction import extract_refusal_direction

    from onegap.runner import HFGenerator, data_path, load_cfg, replay, select_records

    cfg = yaml.safe_load((ROOT / (args.config or "configs/steer_C.yaml")).read_text(encoding="utf-8"))
    ecfg = load_cfg("configs/eval.yaml")
    run_dir = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-steer"))
    run_dir.mkdir(parents=True, exist_ok=True)
    system = (ROOT / ecfg["system_prompts"]["default"]).read_text(encoding="utf-8").strip()
    gen = HFGenerator(ecfg["model"], ecfg.get("model_revision"), ecfg["decoding"])
    tok, model = gen.tok, gen.model

    pairs = list(read_jsonl(ROOT / "data/generated/circuit_prompts.jsonl"))
    n_cal = cfg.get("n_calibration", 96)
    pairs = pairs[: args.limit * 4] if args.limit else pairs[:n_cal]

    def tmpl(text: str) -> str:  # SafeTune's own formatting (vector_extraction._fmt): user turn only
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)

    # raw user text: extract_refusal_direction applies the chat template itself (templating here would nest it)
    harmful, harmless = [p["clean"] for p in pairs], [p["corrupt"] for p in pairs]
    t0 = time.time()
    kw = {}
    if cfg.get("layer") is not None:
        from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig

        kw["config"] = RefusalDirectionConfig(pick_layer=cfg["layer"])
    direction, layer, _ = extract_refusal_direction(model, tok, harmful, harmless, **kw)

    # ||mean diff|| at the block output of `layer` (hidden_states[layer + 1]), last prompt token
    def mean_last(texts: list[str]):
        acc = None
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1][:, -1].float()
            s = hs.sum(0)
            acc = s if acc is None else acc + s
        return acc / len(texts)

    diff_norm = float((mean_last([tmpl(x) for x in harmful]) - mean_last([tmpl(x) for x in harmless])).norm())
    print(f"[steer] layer {layer}, ||mean diff|| {diff_norm:.2f}, calibration {len(pairs)} pairs, "
          f"{time.time() - t0:.0f}s", flush=True)

    # dev subset: attack conversations (full replay) + over-refusal prompts
    recs = list(read_jsonl(data_path("dev")))
    n_att, n_or = (2, 3) if args.limit else (cfg.get("sweep_attack_per_lang", 8), cfg.get("sweep_overref_per_lang", 10))
    att = select_records(recs, "dev", None, n_att, kind="attack")
    orr = select_records(recs, "dev", None, n_or, kind="overrefusal")
    sweep = []
    for s in [0.0, *cfg["strength_sweep"]]:
        with Steered(model, direction, layer, s * diff_norm):
            done = replay(att + orr, system, gen, log=lambda m: None)
        a_rep = [(r, d["rec"]["language"]) for d in done if d["rec"]["kind"] == "attack" for r in d["replies"]]
        o_rep = [(r, d["rec"]["language"]) for d in done if d["rec"]["kind"] == "overrefusal" for r in d["replies"]]
        row = {"strength": s, "strength_abs": s * diff_norm,
               "attack_refusal": refusal_rate([r for r, _ in a_rep], [lang for _, lang in a_rep]),
               "overref_refusal": refusal_rate([r for r, _ in o_rep], [lang for _, lang in o_rep]),
               "n_attack_turns": len(a_rep), "n_overref": len(o_rep),
               "sample": next((r[:200] for r, _ in a_rep), "")}
        sweep.append(row)
        print(f"[steer] {json.dumps({k: v for k, v in row.items() if k != 'sample'})}", flush=True)

    chosen = pick_strength(sweep, cfg.get("target_gain"))
    torch.save(direction.detach().cpu(), run_dir / "direction.pt")
    out = {"layer": layer, "diff_norm": diff_norm, "strength": chosen["strength"],
           "strength_abs": chosen["strength_abs"], "rule": chosen["rule"], "sweep": sweep,
           "n_calibration": len(pairs), "calibration_source": "data/generated/circuit_prompts.jsonl (train)",
           "seconds": round(time.time() - t0), "git_sha": os.environ.get("ONEGAP_SHA")}
    (run_dir / "steer.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=1), flush=True)
    return 0


def load_steering(steer_dir: str | Path):  # pragma: no cover - GPU job
    """(direction, layer, absolute strength) saved by `onegap steer`, for the runner's condition C."""
    import torch

    d = Path(steer_dir)
    s = json.loads((d / "steer.json").read_text(encoding="utf-8"))
    return torch.load(d / "direction.pt"), s["layer"], s["strength_abs"]
