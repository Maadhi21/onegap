"""Figures and spoken numbers from the published reports (notebook, Space, video). CPU only, no model.

Inputs are the report dicts written by `onegap report` (`runs/report/dev_v1.json`, `test_v1.json`) and, for the
language-gap test on the held-out set, the published S0 transcripts. Every figure shows counts (k/n) and Wilson
intervals; safety numbers sit next to over-refusal numbers (Invariant 8).

    uv run python -m onegap.figures runs/report/dev_v1.json runs/report/test_v1.json \
        --test-run runs/<S0 test run> --out runs/report/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onegap import metrics, stats

LANGS = ("en", "hi", "hinglish", "bn")
NAME = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish", "bn": "Bengali\n(never trained)",
        "pooled": "en+hi+Hinglish"}
COND_NAME = {"S0": "single prompt", "A": "base", "B": "safety prompt", "D": "LoRA, all layers",
             "E": "LoRA, circuit", "R": "LoRA, random layers"}
LINKS = {"link_models": "huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd",
         "link_datasets": "huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a",
         "link_space": "huggingface.co/spaces/maadhi/onegap-replay",
         "link_github": "github.com/Maadhi21/onegap",
         "link_notebook": "github.com/Maadhi21/onegap/blob/main/notebooks/reproduce_headline.ipynb"}
# Held-out demo item for the video: English refused outright, Hindi, Hinglish and Bengali answered (test_v1, S0).
DEMO_SEED = "a600-0083"
RED, GREEN, GREY, BLUE, AMBER ="#c0392b", "#27ae60", "#95a5a6", "#2c6fbb", "#e69f00"


# ------------------------------------------------------------------ pure helpers (tested)

def by_seed(trs: list[dict], language: str, outcome: str) -> dict[str, bool]:
    """seed_id -> outcome for the attack items of one language. outcome: 'unsafe' (any turn) or 'refused' (last)."""
    fn = metrics.attack_success if outcome == "unsafe" else metrics.final_refused
    return {t["seed_id"]: fn(t) for t in trs if t.get("kind", "attack") == "attack" and t["language"] == language}


def language_gaps(trs: list[dict], outcome: str, base: str = "en", langs=("hi", "hinglish", "bn")) -> dict:
    """Paired (same seed_id) difference to English with a bootstrap 95% CI and an exact McNemar p."""
    a = by_seed(trs, base, outcome)
    out = {}
    for x in langs:
        b = by_seed(trs, x, outcome)
        if not set(a) & set(b):
            continue
        d, m = stats.paired_bootstrap(a, b), stats.mcnemar(a, b)
        out[x] = {"n": d.n, "diff": d.diff, "lo": d.lo, "hi": d.hi, "p": m.p,
                  "text": f"{d.diff * 100:+.1f} [{d.lo * 100:+.1f}, {d.hi * 100:+.1f}], p={m.p:.2g}"}
    return out


def pct(r: dict) -> str:
    return f"{r['p'] * 100:.0f}%" if r.get("n") else "n/a"


def video_numbers(dev: dict, test: dict, gaps_refused: dict | None = None, gaps_unsafe: dict | None = None,
                  faithfulness: dict | None = None, links: dict | None = None) -> dict:
    """Every number the video speaks or shows, as strings; `{{name}}` = spoken, `{{name__n}}` = count on screen."""
    out: dict[str, str] = {}

    def put(name: str, r: dict) -> None:
        out[name], out[f"{name}__n"] = pct(r), f"{r['k']}/{r['n']}"

    for x in LANGS:
        put(f"dev_s0_{x}", dev["asr"]["S0"][x])
        put(f"dev_a_{x}", dev["asr"]["A"][x])
        put(f"test_unsafe_{x}", test["asr"]["S0"][x])
        put(f"test_refused_{x}", test["final_turn_refused"]["S0"][x])
    for c in ("A", "B", "D", "E", "R"):
        put(f"asr_{c}_pooled", dev["asr"][c]["pooled"])
        put(f"or_{c}_pooled", dev["overrefusal"][c]["pooled"])
    ff = {k: v for k, v in dev["first_failure"]["A"]["pooled"].items() if k != "held"}
    out["modal_fail_turn"] = max(ff, key=lambda k: (ff[k], -int(k))) if ff else "n/a"
    for c in ("D", "E", "R"):
        e = dev["edit_size"][c]
        out[f"params_{c}"] = f"{e['trainable_params'] / 1e6:.1f}M"
        out[f"share_{c}"] = f"{e['trainable_params'] / e['model_params'] * 100:.2f}%"
    out["n_dev_attack_per_lang"] = str(dev["asr"]["A"]["en"]["n"])
    out["n_test_attack"] = str(sum(test["asr"]["S0"][x]["n"] for x in LANGS))
    for name, gaps in (("gap_refused", gaps_refused or {}), ("gap_unsafe", gaps_unsafe or {})):
        for x, g in gaps.items():
            out[f"{name}_{x}"] = f"{abs(g['diff']) * 100:.0f} points"
            out[f"{name}_{x}__p"] = f"p={g['p']:.2g}"
    for x, v in (faithfulness or {}).items():
        out[f"faith_{x}"] = f"{v:.2f}"
    out.update({**LINKS, "demo_seed": DEMO_SEED, **(links or {})})
    return out


# ------------------------------------------------------------------ figures (matplotlib, Agg)

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _bars(ax, groups, series, colors, labels):
    """Grouped bars with Wilson error bars and k/n labels. series: list of lists of rate dicts."""
    w = 0.8 / len(series)
    for i, (rows, col, lab) in enumerate(zip(series, colors, labels, strict=True)):
        xs = [g + (i - (len(series) - 1) / 2) * w for g in range(len(groups))]
        ps = [r["p"] * 100 if r.get("n") else 0 for r in rows]
        err = [[(r["p"] - r["lo"]) * 100 if r.get("n") else 0 for r in rows],
               [(r["hi"] - r["p"]) * 100 if r.get("n") else 0 for r in rows]]
        ax.bar(xs, ps, w * 0.92, color=col, label=lab, yerr=err, capsize=3, ecolor="#555")
        for x, p, r in zip(xs, ps, rows, strict=True):
            top = r["hi"] * 100 if r.get("n") else p
            ax.text(x, top + 0.8, f"{r['k']}/{r['n']}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)), [NAME.get(g, g) for g in groups])
    ax.set_ylabel("%")
    tops = [r["hi"] * 100 for rows in series for r in rows if r.get("n")]
    ax.set_ylim(0, (max(tops) if tops else 10) * 1.3)
    ax.legend(frameon=False, loc="upper left", ncol=len(series))


def fig_single_vs_conversation(dev: dict):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bars(ax, LANGS, [[dev["asr"]["S0"][x] for x in LANGS], [dev["asr"]["A"][x] for x in LANGS]],
          [AMBER, RED], ["one harmful prompt", "full multi-turn conversation"])
    ax.set_title("Attacks that worked, base model (dev, n per bar shown)")
    fig.tight_layout()
    return fig


def fig_first_failure(dev: dict, cond: str = "A"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    turns = sorted({int(k) for x in LANGS for k in dev["first_failure"][cond][x] if k != "held"})
    w = 0.8 / len(LANGS)
    for i, x in enumerate(LANGS):
        ff = dev["first_failure"][cond][x]
        ax.bar([t + (i - 1.5) * w for t in turns], [ff.get(str(t), 0) for t in turns], w * 0.92,
               label=NAME[x].replace("\n", " "))
    ax.set_xticks(turns)
    ax.set_xlabel("turn of the first unsafe reply")
    ax.set_ylabel("conversations")
    ax.set_title(f"Where the model gave in ({COND_NAME[cond]}, dev)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_heldout(test: dict):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bars(ax, LANGS, [[test["final_turn_refused"]["S0"][x] for x in LANGS], [test["asr"]["S0"][x] for x in LANGS]],
          [GREEN, RED], ["refused outright", "unsafe reply"])
    ax.set_title("Held-out test set, one harmful prompt (Attack_600 family)")
    fig.tight_layout()
    return fig


def fig_repairs(dev: dict):
    plt = _plt()
    conds = [c for c in ("A", "B", "D", "E", "R") if c in dev["asr"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bars(ax, conds, [[dev["asr"][c]["pooled"] for c in conds], [dev["overrefusal"][c]["pooled"] for c in conds]],
          [RED, GREY], ["attacks that worked", "harmless items refused"])
    ax.set_xticks(range(len(conds)), [f"{c}\n{COND_NAME[c]}" for c in conds], fontsize=9)
    ax.set_title("Repairs on dev, en+hi+Hinglish pooled: no arm moved attack success")
    fig.tight_layout()
    return fig


def fig_edit_size(dev: dict):
    plt = _plt()
    conds = [c for c in ("D", "E", "R") if c in dev["edit_size"]]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vals = [dev["edit_size"][c]["trainable_params"] / 1e6 for c in conds]
    ax.bar(conds, vals, color=[BLUE, GREEN, GREY])
    for c, v in zip(conds, vals, strict=True):
        ax.text(c, v + 0.4, f"{v:.1f}M", ha="center")
    ax.set_ylabel("trainable parameters (millions)")
    ax.set_title("Edit size: same data and steps, different components")
    fig.tight_layout()
    return fig


FIGURES = {"single_vs_conversation": ("dev", fig_single_vs_conversation), "first_failure": ("dev", fig_first_failure),
           "heldout_single_prompt": ("test", fig_heldout), "repairs": ("dev", fig_repairs),
           "edit_size": ("dev", fig_edit_size)}


def save_all(dev: dict, test: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (which, fn) in FIGURES.items():
        fig = fn(dev if which == "dev" else test)
        p = out / f"{name}.png"
        fig.savefig(p, dpi=160)
        _plt().close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    from onegap.report import load_run

    ap = argparse.ArgumentParser()
    ap.add_argument("dev")
    ap.add_argument("test")
    ap.add_argument("--test-run", help="held-out S0 run folder, for the paired language gaps")
    ap.add_argument("--circuits", help="circuits_report.json, for faithfulness per language")
    ap.add_argument("--out", default="runs/report/figures")
    a = ap.parse_args(argv)
    dev, test = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.dev, a.test))
    out = Path(a.out)
    for p in save_all(dev, test, out):
        print(p)
    gr = gu = None
    if a.test_run:
        trs = load_run(a.test_run).trs
        gr, gu = language_gaps(trs, "refused"), language_gaps(trs, "unsafe")
        print("refused gap to en:", {x: g["text"] for x, g in gr.items()})
        print("unsafe gap to en:", {x: g["text"] for x, g in gu.items()})
    faith = None
    if a.circuits:
        rep = json.loads(Path(a.circuits).read_text(encoding="utf-8"))
        faith = {x: v["faithfulness"]["patching_score"] for x, v in rep.get("languages", {}).items()
                 if isinstance(v, dict) and v.get("faithfulness")}
    nums = video_numbers(dev, test, gr, gu, faith)
    (out.parent / "video_numbers.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.parent / 'video_numbers.json'} ({len(nums)} values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
