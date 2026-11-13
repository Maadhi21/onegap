"""Edit size of repairs D, E and R (spec 8.7): parameters changed, layers touched, and ||Delta||_F of the merged
LoRA delta (Delta W = B A * alpha / r per adapted module).

`measure(peft_model)` runs at the end of each training job (train.py) and lands in train_manifest.json;
`onegap edit-size` collects the manifests of the runs listed in data/generated/train_runs.txt into
runs/report/edit_size.json, with each arm's share of the model's parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def measure(model) -> dict:  # pragma: no cover - GPU, called from train.py
    import torch

    per_layer: dict[int, float] = {}
    changed = adapter = 0
    sq = 0.0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not hasattr(mod, "lora_A") or "default" not in getattr(mod, "lora_A", {}):
                continue
            a, b = mod.lora_A["default"].weight.float(), mod.lora_B["default"].weight.float()
            delta = (b @ a) * mod.scaling["default"]
            f2 = float((delta ** 2).sum())
            sq += f2
            changed += int((delta != 0).sum())
            adapter += a.numel() + b.numel()
            if "layers." in name:
                layer = int(name.split("layers.")[1].split(".")[0])
                per_layer[layer] = per_layer.get(layer, 0.0) + f2
    total = sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n)
    return {"params_changed": changed, "adapter_params": adapter, "model_params": total,
            "changed_share": changed / total if total else None, "delta_fro": sq ** 0.5,
            "layers_touched": sorted(per_layer), "delta_fro_by_layer": {k: v ** 0.5 for k, v in sorted(per_layer.items())}}


def table(manifests: dict[str, dict]) -> dict:
    """condition -> row for the report; pure, unit-tested."""
    out = {}
    for cond, m in sorted(manifests.items()):
        e = m["edit_size"]
        out[cond] = {"params_changed": e["params_changed"], "changed_pct": round(100 * e["changed_share"], 3),
                     "adapter_params": e["adapter_params"], "layers_touched": len(e["layers_touched"]),
                     "delta_fro": round(e["delta_fro"], 4), "lora_modules": m["lora_modules"]}
    return out


def main(args) -> int:
    runs = [ROOT / p.strip() for p in (ROOT / "data/generated/train_runs.txt").read_text().splitlines() if p.strip()]
    mans = {}
    for r in runs:
        m = json.loads((r / "train_manifest.json").read_text(encoding="utf-8"))
        mans[m["condition"]] = m
    rep = table(mans)
    out = ROOT / "runs/report/edit_size.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0
