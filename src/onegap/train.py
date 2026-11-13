"""Repairs D, E and R (spec 8.7): LoRA DPO on the correction pairs with matched settings.

Invariant 4: D, E and R share data, pair count, epochs, steps, lr, beta and seed; only target modules differ.
The configs keep the shared block byte-identical (checked by tests/test_train.py and again here at run time).

    onegap train --config configs/train_{D,E,R}.yaml [--limit N] [--input runs/<circuits>/selected_components.json]

Targets:
  D  all-linear (every layer)
  E  path E1: PEFT target_modules = full names of the projections in the circuit's selected attention and MLP
     blocks (circuits.py -> selected_components.json["E"]["modules"])
  R  same counts of attention and MLP blocks, random over layers disjoint from E (selected_components["R"])

Outputs in $ONEGAP_RUN_DIR: adapter/ (PEFT), train_log.json (Trainer log history: loss, rewards/margins),
train_manifest.json (settings, targets, trainable parameters, timings, TRL version, config fields not
accepted by the installed TRL).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import yaml

from onegap.schema import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ("configs/train_D.yaml", "configs/train_E.yaml", "configs/train_R.yaml")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def load_train_cfg(path: str | Path) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def shared_fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg["shared"], sort_keys=True).encode()).hexdigest()[:16]


def target_modules(cfg: dict, components: dict | None) -> str | list[str]:
    mode = cfg["target"]["mode"]
    if mode == "all-linear":
        return "all-linear"
    if components is None:
        raise ValueError(f"target mode {mode!r} needs selected_components.json")
    key = {"circuit": "E", "random": "R"}[mode]
    mods = list(components[key]["modules"])
    if not mods:
        raise ValueError(f"empty module list for {key}")
    return mods


def module_counts(names: list[str]) -> dict[str, int]:
    """Blocks per type: attention blocks (any of q/k/v/o in a layer) and MLP blocks, plus projection count."""
    attn = {n.split(".self_attn.")[0] for n in names if ".self_attn." in n}
    mlp = {n.split(".mlp.")[0] for n in names if ".mlp." in n}
    return {"attn_blocks": len(attn), "mlp_blocks": len(mlp), "projections": len(names)}


def missing_targets(model_names: list[str], targets: list[str]) -> list[str]:
    """Targets with no matching module (PEFT matches a list entry by exact name or '.'+name suffix)."""
    return [t for t in targets if not any(n == t or n.endswith("." + t) for n in model_names)]


def split_kwargs(wanted: dict, accepted: set[str]) -> tuple[dict, dict]:
    """Keep the kwargs the installed class accepts; return (kept, dropped) so drops land in the manifest."""
    return ({k: v for k, v in wanted.items() if k in accepted}, {k: v for k, v in wanted.items() if k not in accepted})


def dpo_settings(shared: dict, out_dir: str) -> dict:
    eff = shared["effective_batch"]
    return {
        "output_dir": out_dir, "beta": shared["dpo"]["beta"], "learning_rate": shared["lr"],
        "num_train_epochs": shared["epochs"], "per_device_train_batch_size": 1, "gradient_accumulation_steps": eff,
        "max_length": shared["max_length"], "max_prompt_length": shared["max_prompt_length"],
        "bf16": shared["bf16"], "seed": shared["seed"], "data_seed": shared["seed"],
        "gradient_checkpointing": True, "logging_steps": 2, "save_strategy": "no", "report_to": [],
        # transformers 5: warmup_ratio is gone; warmup_steps takes a float in [0, 1) as a ratio (training_args.py)
        "lr_scheduler_type": "cosine", "warmup_steps": 0.1, "remove_unused_columns": False,
        "precompute_ref_log_probs": False,
    }


def keep_by_length(n_prompt: int, n_chosen: int, n_rejected: int, max_length: int, max_prompt: int) -> bool:
    """TRL 1.14 has no max_prompt_length and truncates prompt+completion with keep_start (cuts the reply end).
    Instead drop over-long pairs, the same ones for D, E and R (same data and tokenizer)."""
    return n_prompt <= max_prompt and n_prompt + max(n_chosen, n_rejected) <= max_length


def load_pairs(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in read_jsonl(ROOT / path)]
    return rows[:limit] if limit else rows


def main(args) -> int:  # pragma: no cover - GPU job
    import dataclasses
    import inspect

    import torch
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = load_train_cfg(args.config or "configs/train_D.yaml")
    fps = {c: shared_fingerprint(load_train_cfg(c)) for c in CONFIGS}
    if len(set(fps.values())) != 1:
        raise SystemExit(f"Invariant 4: shared settings differ across D/E/R: {fps}")
    shared = cfg["shared"]
    eval_cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    run_dir = Path(os.environ.get("ONEGAP_RUN_DIR", f"runs/local-train-{cfg['condition']}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    comp_path = Path(args.input) if args.input else ROOT / cfg["target"].get("circuit", cfg["target"].get("match", ""))
    components = json.loads(comp_path.read_text(encoding="utf-8")) if cfg["target"]["mode"] != "all-linear" else None
    targets = target_modules(cfg, components)
    pairs = load_pairs(shared["data"], args.limit)
    print(f"[train] condition {cfg['condition']}, {len(pairs)} pairs, targets "
          f"{targets if isinstance(targets, str) else module_counts(targets)}", flush=True)

    tok = AutoTokenizer.from_pretrained(eval_cfg["model"], revision=eval_cfg.get("model_revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    def ntok(msgs, gen: bool) -> int:
        text = tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)  # str in every version
        return len(tok(text, add_special_tokens=False)["input_ids"])

    keep = [keep_by_length(ntok(p["prompt"], True), len(tok(p["chosen"][0]["content"])["input_ids"]),
                           len(tok(p["rejected"][0]["content"])["input_ids"]),
                           shared["max_length"], shared["max_prompt_length"]) for p in pairs]
    n_long = keep.count(False)
    pairs = [p for p, k in zip(pairs, keep, strict=True) if k]
    print(f"[train] dropped {n_long} over-long pairs; training on {len(pairs)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(eval_cfg["model"], revision=eval_cfg.get("model_revision"),
                                                 dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    names = [n for n, _ in model.named_modules()]
    if not isinstance(targets, str):
        miss = missing_targets(names, targets)
        if miss:
            raise SystemExit(f"VERIFY failed: {len(miss)} target modules not in cohere2 model, e.g. {miss[:3]}")

    lora = LoraConfig(r=shared["lora"]["r"], lora_alpha=shared["lora"]["alpha"],
                      lora_dropout=shared["lora"]["dropout"], target_modules=targets, task_type="CAUSAL_LM")
    accepted = {f.name for f in dataclasses.fields(DPOConfig)}
    kept, dropped = split_kwargs(dpo_settings(shared, str(run_dir / "trainer_out")), accepted)
    if dropped:
        print(f"[train] VERIFY: DPOConfig in trl {trl.__version__} does not accept {sorted(dropped)}", flush=True)
    dcfg = DPOConfig(**kept)
    sig = inspect.signature(DPOTrainer.__init__).parameters
    tkw = {"model": model, "args": dcfg, "train_dataset": Dataset.from_list(pairs), "peft_config": lora}
    tkw["processing_class" if "processing_class" in sig else "tokenizer"] = tok
    trainer = DPOTrainer(**tkw)

    lora_mods = sorted({n.rsplit(".lora_A", 1)[0] for n, _ in trainer.model.named_modules() if ".lora_A" in n})
    lora_mods = [n.split("base_model.model.", 1)[-1] for n in lora_mods]
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[train] LoRA on {len(lora_mods)} modules {module_counts(lora_mods)}; trainable params {trainable:,}",
          flush=True)
    if not isinstance(targets, str) and len(lora_mods) != len(targets):
        raise SystemExit(f"LoRA landed on {len(lora_mods)} modules, expected {len(targets)}")

    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0
    trainer.model.save_pretrained(run_dir / "adapter")
    tok.save_pretrained(run_dir / "adapter")
    hist = trainer.state.log_history
    (run_dir / "train_log.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    margins = [h["rewards/margins"] for h in hist if "rewards/margins" in h]
    from onegap.edit_size import measure

    manifest = {
        "edit_size": measure(trainer.model),
        "condition": cfg["condition"], "config": str(args.config), "shared": shared,
        "shared_fingerprint": fps[f"configs/train_{cfg['condition']}.yaml"], "target_mode": cfg["target"]["mode"],
        "path_used": components.get("path_used") if components else "all-linear",
        "components_file": str(comp_path) if components else None,
        "lora_modules": len(lora_mods), "lora_module_counts": module_counts(lora_mods),
        "lora_layers": sorted({int(n.split("layers.")[1].split(".")[0]) for n in lora_mods if "layers." in n}),
        "trainable_params": trainable, "n_pairs": len(pairs), "n_dropped_long": n_long,
        "transformers_version": __import__("transformers").__version__, "steps": trainer.state.global_step,
        "train_loss": result.training_loss, "margins_first_last": [margins[:1], margins[-1:]],
        "seconds_train": round(secs), "trl_version": trl.__version__, "dpo_config_dropped": dropped,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "pairs_by_kind": dict(Counter(p["meta"]["kind"] for p in read_jsonl(ROOT / shared["data"])))
        if not args.limit else None, "git_sha": os.environ.get("ONEGAP_SHA"),
    }
    (run_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "shared"}, indent=1), flush=True)
    return 0
