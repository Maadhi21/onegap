"""Build the public copy of run folders (Invariant 7). Nothing in this module uploads anything.

    onegap publish-stage runs/<run> [runs/<run> ...] --out publish/transcripts-and-logs
                         [--attack-replies withheld|redact-unsafe]

For every run folder it writes a public `transcripts.jsonl`, the run manifest and job status and, for runs that
used an adapter, the training manifest and log of that adapter. What changes on the way out:

- a reply labelled unsafe becomes the redaction marker with the SHA-256 of the original text (redact.py), in any
  kind of conversation, and its English view is dropped;
- in attack conversations, with `--attack-replies withheld` (the default until the human audit has checked the
  judge), every other reply becomes a "withheld" marker with its SHA-256 and its English view is dropped;
- user turns that are XGuard-Train text (seed ids `xg-…`; gated source, no licence stated) become a "withheld"
  marker with their SHA-256;
- everything else is copied as it is: labels, judge output, and harmless conversations in full.

The staged folder is then checked (`leaks`, plus redact.check_paths). Any finding makes the command exit with 1:
do not upload a folder that failed. `tests/test_redaction.py` scans the same folder when ONEGAP_PUBLISH_DIR is set.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from onegap.redact import check_paths, is_redacted, redaction_marker, sha256
from onegap.schema import read_jsonl, write_jsonl

WITHHELD = "[withheld: "
NO_LICENCE_PREFIXES = ("xg-",)  # XGuard-Train: publish ids and labels, never its text
MODES = ("withheld", "redact-unsafe")
COPY = ("run_manifest.json", "job_status.json")
TRAIN_COPY = ("train_manifest.json", "train_log.json", "job_status.json")


def withheld(text: str, why: str) -> str:
    return f"{WITHHELD}{why} | sha256={sha256(text)}]"


def is_withheld(text: str) -> bool:
    return text.startswith(WITHHELD)


def no_licence(seed_id: str) -> bool:
    return seed_id.startswith(NO_LICENCE_PREFIXES)


def public_transcript(tr: dict, attack_replies: str = "withheld") -> dict:
    """The publishable copy of one transcript line. The input is not changed."""
    if attack_replies not in MODES:
        raise ValueError(f"attack_replies must be one of {MODES}")
    out = dict(tr)
    hide_user = no_licence(tr["seed_id"])
    hide_reply = tr.get("kind", "attack") == "attack" and attack_replies == "withheld"
    turns = []
    for t in tr["turns"]:
        t = dict(t)
        if hide_user and not is_withheld(t["user"]):
            t["user"] = withheld(t["user"], "XGuard-Train text")
        hidden = is_redacted(t["assistant"]) or is_withheld(t["assistant"])
        if t.get("label") == "unsafe" and not hidden:
            judges = t.get("judge") or t.get("judges") or "unsafe"
            if isinstance(judges, dict):  # codes are lists: flatten for the marker
                judges = {k: "+".join(v) if isinstance(v, list) else v for k, v in judges.items()}
            t["assistant"] = redaction_marker(t["assistant"], judges, tr.get("category") or "none", t["turn"])
            hidden = True
        elif hide_reply and not hidden and not t.get("guard_blocked"):
            t["assistant"] = withheld(t["assistant"], "attack-conversation reply")
            hidden = True
        if hidden:
            t.pop("assistant_en", None)  # the English view of a hidden reply is the same content
        turns.append(t)
    out["turns"] = turns
    return out


def leaks(rows: list[dict], attack_replies: str = "withheld") -> list[str]:
    """Everything in a staged file that should not be there. Empty list = safe to publish."""
    bad = []
    for r in rows:
        rid = r.get("conv_id", "?")
        for t in r.get("turns", []):
            a, n = t.get("assistant", ""), t.get("turn")
            hidden = is_redacted(a) or is_withheld(a)
            if t.get("label") == "unsafe" and not is_redacted(a):
                bad.append(f"{rid} turn {n}: unsafe reply not redacted")
            if hidden and "assistant_en" in t:
                bad.append(f"{rid} turn {n}: English view of a hidden reply")
            if t.get("label") == "unsafe" and "assistant_en" in t:
                bad.append(f"{rid} turn {n}: English view of an unsafe reply")
            if no_licence(r.get("seed_id", "")) and not is_withheld(t.get("user", "")):
                bad.append(f"{rid} turn {n}: XGuard-Train user turn in the clear")
            if (attack_replies == "withheld" and r.get("kind", "attack") == "attack" and not hidden
                    and not t.get("guard_blocked")):
                bad.append(f"{rid} turn {n}: attack-conversation reply in the clear")
    return bad


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_run(run_dir: Path, out: Path, attack_replies: str = "withheld") -> dict:
    """Write the public copy of one run folder under out/runs/<run_id>/; returns counts for the publish manifest."""
    dst = out / "runs" / run_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    src_rows = list(read_jsonl(run_dir / "transcripts.jsonl"))
    rows = [public_transcript(t, attack_replies) for t in src_rows]
    write_jsonl(dst / "transcripts.jsonl", rows)
    for name in COPY:
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, dst / name)
    turns = [t for r in rows for t in r["turns"]]
    info = {"run_id": run_dir.name, "conversations": len(rows), "turns": len(turns),
            "replies_redacted_unsafe": sum(is_redacted(t["assistant"]) for t in turns),
            "replies_withheld": sum(is_withheld(t["assistant"]) for t in turns),
            "user_turns_withheld": sum(is_withheld(t["user"]) for t in turns),
            "replies_in_full": sum(not (is_redacted(t["assistant"]) or is_withheld(t["assistant"])) for t in turns),
            "labels_unchanged": [t.get("label") for r in rows for t in r["turns"]]
                                == [t.get("label") for r in src_rows for t in r["turns"]],
            "transcripts_sha256": _sha_file(dst / "transcripts.jsonl")}
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    info["condition"] = manifest.get("condition")
    rid = manifest.get("adapter_run")
    if rid and (run_dir.parent / rid).is_dir():
        tdst = out / "runs" / rid
        tdst.mkdir(parents=True, exist_ok=True)
        for name in TRAIN_COPY:
            if (run_dir.parent / rid / name).exists():
                shutil.copyfile(run_dir.parent / rid / name, tdst / name)
        info["train_run"] = rid
    return info


def check_stage(out: Path, attack_replies: str = "withheld") -> list[str]:
    bad = check_paths([out])
    for f in sorted(out.rglob("*.jsonl")):
        bad += [f"{f}: {x}" for x in leaks(list(read_jsonl(f)), attack_replies)]
    return sorted(set(bad))


def main(args) -> int:
    out, mode = Path(args.out), args.attack_replies
    out.mkdir(parents=True, exist_ok=True)
    runs = [stage_run(Path(p), out, mode) for p in args.paths]
    for extra in getattr(args, "add", None) or []:  # report files and the like: copied to out/report/
        (out / "report").mkdir(exist_ok=True)
        shutil.copyfile(extra, out / "report" / Path(extra).name)
    bad = check_stage(out, mode)
    (out / "publish_manifest.json").write_text(
        json.dumps({"attack_replies": mode, "runs": runs, "problems": bad}, indent=2), encoding="utf-8")
    for r in runs:
        print(f"[publish-stage] {r['condition']} {r['run_id']}: {r['conversations']} conversations, {r['turns']} turns; "
              f"redacted unsafe {r['replies_redacted_unsafe']}, withheld replies {r['replies_withheld']}, withheld "
              f"user turns {r['user_turns_withheld']}, replies in full {r['replies_in_full']}; labels unchanged "
              f"{r['labels_unchanged']}")
    print("\n".join(bad) if bad else f"[publish-stage] {out}: no leaks found. Nothing has been uploaded.")
    return 1 if bad or not all(r["labels_unchanged"] for r in runs) else 0
