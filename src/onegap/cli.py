"""Single entry point. Every Makefile target maps to one subcommand here."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

# subcommand -> (module, milestone, needs_gpu). Implemented commands are handled inline below.
PENDING = {
    "smoke": ("onegap.smoke", "M0", True),
    "data": ("onegap.data.sources", "M1", False),
    "translate": ("onegap.data.translate", "M1", True),
    "codemix": ("onegap.data.codemix", "M1", False),
    "benign": ("onegap.data.benign", "M1", False),
    "curate": ("onegap.data.curate", "M1", False),
    "score-hinglish": ("onegap.data.curate:main_score", "M1", False),
    "codemix-back": ("onegap.data.codemix:main_back", "M1", False),
    "codemix-regen": ("onegap.data.codemix:main_regen", "M1", False),
    "score-hinglish-back": ("onegap.data.curate:main_score_back", "M1", False),
    "run": ("onegap.runner", "M2", True),
    "vllm-smoke": ("onegap.vllm_backend:smoke", "M2", True),
    "judge": ("onegap.runner:main_judge", "M2", True),
    "prefdata": ("onegap.prefdata", "M5", True),
    "prefdata-refilter": ("onegap.prefdata:main_refilter", "M5", False),
    "circuits": ("onegap.circuits", "M6", True),
    "circuit-data": ("onegap.circuits:main_data", "M6", False),
    "circuit-check": ("onegap.circuits:main_crosscheck", "M6", True),
    "train": ("onegap.train", "M7", True),
    "steer": ("onegap.steer", "M7", True),
    "edit-size": ("onegap.edit_size", "M7", True),
    "replay": ("onegap.runner", "M8", True),
}


def cmd_check_env(_: argparse.Namespace) -> int:
    """Presence only; never prints secret values."""
    ok = True
    for var, required in (("HF_TOKEN", True), ("COHERE_API_KEY", True), ("HF_TOKEN_GATED", False),
                          ("ONEGAP_HF_WORK_REPO", True)):
        present = bool(os.environ.get(var))
        ok &= present or not required
        print(f"{var:20s} {'set' if present else ('MISSING' if required else 'not set (optional)')}")
    print(f"HF_HOME          {os.environ.get('HF_HOME', '(default)')}")
    try:
        import torch  # noqa: F401

        print(f"cuda             {torch.cuda.is_available()}")
    except ImportError:
        print("cuda             torch not installed (expected on the dev box)")
    return 0 if ok else 1


def cmd_freeze(_: argparse.Namespace) -> int:
    from onegap import splits

    out = splits.write_sums()
    print(f"wrote {out}; now: git add data/frozen && git commit && git tag test-freeze-v1")
    return 0


def cmd_verify_frozen(_: argparse.Namespace) -> int:
    from onegap import splits

    print(f"frozen test set OK; SHA256SUMS digest {splits.verify_sums()}")
    return 0


def cmd_check_splits(a: argparse.Namespace) -> int:
    from onegap.schema import read_jsonl, validate_record
    from onegap.splits import check_splits

    recs = [validate_record(r) for p in a.paths for r in read_jsonl(p)]
    errs = check_splits(recs)
    print("\n".join(errs) if errs else f"{len(recs)} records OK")
    return 1 if errs else 0


def cmd_redact_check(a: argparse.Namespace) -> int:
    from onegap.redact import check_paths

    bad = check_paths(a.paths)
    print("\n".join(bad) if bad else "redaction OK")
    return 1 if bad else 0


def cmd_metrics(a: argparse.Namespace) -> int:
    from onegap import metrics
    from onegap.schema import read_jsonl

    trs = [t for p in a.paths for t in read_jsonl(p)]
    table = metrics.asr_table(trs)
    out = {f"{c}/{lang}": r.fmt() for (c, lang), r in sorted(table.items())}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_publish_stage(a: argparse.Namespace) -> int:
    from onegap import publish

    return publish.main(a)


def cmd_publish_hub(a: argparse.Namespace) -> int:
    from onegap import publish_hub

    return publish_hub.main(a)


def cmd_publish_visibility(a: argparse.Namespace) -> int:
    from onegap import publish_hub

    return publish_hub.main_visibility(a)


def cmd_report(a: argparse.Namespace) -> int:
    from onegap import report

    return report.main(a)


def main(argv: list[str] | None = None) -> int:
    from onegap.env import load_dotenv

    for stream in (sys.stdout, sys.stderr):  # tables use "−" and "‖": a cp1252 console or pipe would crash on them
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()
    p = argparse.ArgumentParser(prog="onegap")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-env").set_defaults(fn=cmd_check_env)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    sub.add_parser("verify-frozen").set_defaults(fn=cmd_verify_frozen)
    for name, fn in (("check-splits", cmd_check_splits), ("redact-check", cmd_redact_check),
                     ("metrics", cmd_metrics)):
        sp = sub.add_parser(name)
        sp.add_argument("paths", nargs="+", type=Path)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("publish-hub", help="[M9, CPU job] build and upload the public repos; dry run unless --execute")
    sp.add_argument("--org", required=True)
    sp.add_argument("--eval-runs", nargs="+", required=True, help="evaluation run ids (folders under runs/)")
    sp.add_argument("--circuit-run")
    sp.add_argument("--crosscheck-run")
    sp.add_argument("--report", nargs="*", type=Path, help="report files copied into the transcripts dataset")
    sp.add_argument("--cards", type=Path, default=Path("docs/cards"))
    sp.add_argument("--attack-replies", choices=["withheld", "redact-unsafe"], default="withheld")
    sp.add_argument("--execute", action="store_true", help="create repos and upload (private unless --public)")
    sp.add_argument("--public", action="store_true")
    sp.add_argument("--message", default="OneGap: first public version", help="commit message on every repo")
    sp.set_defaults(fn=cmd_publish_hub)
    sp = sub.add_parser("publish-visibility", help="[M9] make the repos in a publish manifest public (or private)")
    sp.add_argument("manifest", type=Path)
    sp.add_argument("--public", action="store_true", help="without this flag the repos are set to private")
    sp.set_defaults(fn=cmd_publish_visibility)
    sp = sub.add_parser("report", help="[M4/M7/M8] tables and the D/E/R rule from run folders (no GPU)")
    sp.add_argument("paths", nargs="+", type=Path, help="run folders, one condition each")
    sp.add_argument("--name", default="report", help="output file stem")
    sp.add_argument("--base", default="A", help="condition the others are compared with")
    sp.add_argument("--out", type=Path, default=Path("runs/report"))
    sp.set_defaults(fn=cmd_report)
    sp = sub.add_parser("publish-stage", help="[M9] build the public copy of run folders; uploads nothing")
    sp.add_argument("paths", nargs="+", type=Path, help="run folders")
    sp.add_argument("--out", type=Path, required=True, help="staging folder, e.g. publish/transcripts-and-logs")
    sp.add_argument("--attack-replies", choices=["withheld", "redact-unsafe"], default="withheld",
                    help="withheld: labels and hashes only for attack conversations (until the human audit)")
    sp.add_argument("--add", nargs="*", type=Path, help="extra files copied to <out>/report/")
    sp.set_defaults(fn=cmd_publish_stage)
    for name, (mod, ms, gpu) in PENDING.items():
        sp = sub.add_parser(name, help=f"[{ms}{', GPU job' if gpu else ''}]")
        sp.add_argument("--config", type=Path)
        sp.add_argument("--split", default="dev")
        sp.add_argument("--condition")
        sp.add_argument("--language")
        sp.add_argument("--limit", type=int, help="smoke-run on the first N items")
        sp.add_argument("--input", type=Path, help="input file (judge: transcripts to re-judge)")
        sp.add_argument("--kind", choices=["attack", "benign", "overrefusal"], help="run: only this record kind")
        sp.add_argument("--adapter-run", help="run/replay: use the adapter trained in runs/<run_id> (fetched from the "
                                              "private work repo inside a job)")
        sp.set_defaults(fn=None, module=mod, milestone=ms)
    a = p.parse_args(argv)
    if a.fn is not None:
        return a.fn(a)
    mod_name, _, fn_name = a.module.partition(":")
    m = importlib.import_module(mod_name)
    fn = getattr(m, fn_name or "main", None)
    if fn is None:
        print(f"'{a.cmd}' is not implemented yet (milestone {a.milestone}, module {a.module})", file=sys.stderr)
        return 2
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
