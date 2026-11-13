"""Pure helpers for the replay Space: index the published transcripts and render one conversation as HTML.

Only published files are read (maadhi/onegap-transcripts-and-logs). Those are already redacted: unsafe replies,
attack-conversation replies and XGuard-Train user turns are withheld markers, so nothing here can show them.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

CONDITION_ORDER = ("S0", "A", "B", "D", "E", "R")
COND_NAME = {"S0": "single prompt", "A": "base", "B": "safety prompt", "D": "LoRA all layers",
             "E": "LoRA circuit", "R": "LoRA random layers"}
LANG_NAME = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish", "bn": "Bengali (never trained)"}
KIND_NAME = {"attack": "attack (multi-turn jailbreak)", "benign": "harmless conversation",
             "overrefusal": "harmless prompt that sounds risky"}
LABEL_STYLE = {"unsafe": ("#c0392b", "unsafe"), "refused": ("#27ae60", "refused"),
               "deflected": ("#7f8c8d", "answered / deflected")}
MARKERS = ("[withheld", "[REDACTED")  # onegap.redact markers; the text after "|" is a hash, not content


def load_runs(root: Path) -> dict[str, dict[str, dict]]:
    """split -> condition -> {conv_id: transcript}, from the run folders a published report lists."""
    out: dict[str, dict[str, dict]] = {}
    for name in ("dev_v1", "test_v1"):
        rep = json.loads((root / "report" / f"{name}.json").read_text(encoding="utf-8"))
        split = rep["split"]
        for cond, meta in rep["runs"].items():
            path = root / "runs" / meta["run_id"] / "transcripts.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            out.setdefault(split, {})[cond] = {t["conv_id"]: t for t in rows}
    return out


def conditions(data: dict, split: str) -> list[str]:
    return [c for c in CONDITION_ORDER if c in data.get(split, {})]


def conversation_ids(data: dict, split: str, language: str, kind: str) -> list[str]:
    """conv_ids of one language and kind, from the fullest condition (A on dev, S0 on test)."""
    conds = conditions(data, split)
    ref = "A" if "A" in conds else conds[0]
    return sorted(c for c, t in data[split][ref].items() if t["language"] == language and t.get("kind") == kind)


def pill(label: str | None) -> str:
    if label is None:
        return '<span style="color:#bbb">—</span>'
    color, text = LABEL_STYLE.get(label, ("#7f8c8d", label))
    return (f'<span style="background:{color};color:white;border-radius:9px;padding:1px 8px;'
            f'font-size:0.85em;white-space:nowrap">{html.escape(text)}</span>')


def _text(s: str | None, limit: int = 1200) -> str:
    if not s:
        return '<i style="color:#999">(empty)</i>'
    if s.startswith(MARKERS):
        return f'<i style="color:#999">{html.escape(s.split("|")[0].strip("[ "))}</i>'
    s = s if len(s) <= limit else s[:limit] + " …"
    return html.escape(s).replace("\n", "<br>")


def aligned_turn(tr: dict, i: int, n: int) -> dict | None:
    """Turn i (0-based) of an n-turn conversation; S0 asks only the final user turn, so its one turn is the last."""
    j = i - (n - len(tr["turns"]))
    return tr["turns"][j] if j >= 0 else None


def outcome(t: dict) -> str:
    labels = [x["label"] for x in t["turns"]]
    if "unsafe" in labels:
        return f"unsafe at turn {labels.index('unsafe') + 1}"
    return "refused at the end" if labels[-1] == "refused" else "held"


def render(data: dict, split: str, conv_id: str) -> str:
    """Grid of turns x conditions with judge labels, then the user turns and (where published) the replies."""
    conds = [c for c in conditions(data, split) if conv_id in data[split][c]]
    if not conds:
        return "<p>Conversation not found.</p>"
    ref = data[split]["A" if "A" in conds else conds[0]][conv_id]
    n = max(len(data[split][c][conv_id]["turns"]) for c in conds)
    head = "".join(f"<th style='padding:4px 8px'>{c}<br><small>{COND_NAME[c]}</small></th>" for c in conds)
    rows = []
    for i in range(n):
        cells = []
        for c in conds:
            turn = aligned_turn(data[split][c][conv_id], i, n)
            cells.append(f"<td style='text-align:center;padding:4px'>{pill(turn['label'] if turn else None)}</td>")
        rows.append(f"<tr><td style='padding:4px 8px'>turn {i + 1}</td>{''.join(cells)}</tr>")
    summary = "".join(f"<td style='text-align:center;padding:4px'><small>{outcome(data[split][c][conv_id])}</small></td>"
                      for c in conds)
    meta = (f"<p><b>{html.escape(LANG_NAME.get(ref['language'], ref['language']))}</b> · "
            f"{html.escape(KIND_NAME.get(ref.get('kind', ''), ref.get('kind', '')))} · category "
            f"<code>{html.escape(str(ref.get('category')))}</code> · split <code>{split}</code> · "
            f"<code>{html.escape(conv_id)}</code></p>")
    grid = (f"<table style='border-collapse:collapse'><tr><th></th>{head}</tr>{''.join(rows)}"
            f"<tr><td style='padding:4px 8px'><b>outcome</b></td>{summary}</tr></table>")
    detail = []
    for i, t in enumerate(ref["turns"]):
        replies = []
        for c in conds:
            turn = aligned_turn(data[split][c][conv_id], i, n)
            if turn is not None:
                replies.append(f"<div style='margin:4px 0 4px 16px'>{pill(turn['label'])} <b>{c}</b>: "
                               f"{_text(turn.get('assistant'))}</div>")
        detail.append(f"<div style='margin-top:10px'><b>User, turn {i + 1}:</b> {_text(t.get('user'))}"
                      f"{''.join(replies)}</div>")
    note = ("<p style='color:#777'><small>Replay of published transcripts, judged by Llama Guard 3 8B. Unsafe "
            "replies, replies in attack conversations and XGuard-Train user turns are withheld in the published "
            "data, so only their labels appear here.</small></p>")
    return meta + grid + note + "".join(detail)
