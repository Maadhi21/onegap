"""Launch a Makefile target as an HF Job.

    uv run --extra local python scripts/launch_job.py smoke --flavor l4x1 --venv core -- --limit 4

1. `git archive HEAD` -> bundles/<sha>.tar.gz uploaded to the private working dataset repo.
2. Start an HF Job that runs scripts/job_entry.sh, which downloads the bundle, builds one venv,
   runs `make <target> ARGS=...`, and uploads runs/<run_id>/ back to the working repo.

VERIFY (docs/API_NOTES.md) before first real use: `huggingface_hub.run_job` signature, flavor names, price.
The working repo stays PRIVATE: it holds unredacted transcripts. Nothing here publishes anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Default image of huggingface_hub.run_uv_job; torch (with its CUDA runtime) installs into the job venv.
IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"


def sh(*cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from onegap.env import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("--flavor", default="l4x1")
    p.add_argument("--venv", choices=["core", "circuit", "audit", "vllm", "publish"], default="core")
    p.add_argument("--work-repo", default=os.environ.get("ONEGAP_HF_WORK_REPO"))
    p.add_argument("--timeout", default="2h")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--direct", action="store_true",
                   help="run `python -m onegap.cli <target> <args>` in the job instead of `make <target>` "
                        "(for commands that have no Makefile target, e.g. publish-hub)")
    p.add_argument("--inputs-from", help="reuse the --inputs already uploaded for this earlier run_id (no re-upload; "
                                         "the local files must be unchanged: their sha256 is checked against it)")
    p.add_argument("--inputs", nargs="*", default=[],
                   help="untracked local files (e.g. data/generated/*.jsonl) shipped via the private work repo")
    # Split at "--" ourselves: argparse.REMAINDER after a positional swallowed --dry-run (bug, 2026-11-10).
    argv = sys.argv[1:]
    job_args = argv[argv.index("--") + 1:] if "--" in argv else []
    a = p.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    a.args = job_args

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")
    if not a.work_repo:
        sys.exit("set ONEGAP_HF_WORK_REPO=<org>/onegap-work (private dataset repo)")
    if sh("git", "status", "--porcelain", "--untracked-files=no"):
        sys.exit("commit first: jobs run the committed tree only (reproducibility)")

    sha = sh("git", "rev-parse", "--short=12", "HEAD")
    run_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-{a.target}-{sha}"
    extra = " ".join(a.args)

    inputs = [Path(x) for x in a.inputs]
    for f in inputs:
        if not (ROOT / f).is_file():
            sys.exit(f"--inputs: {f} not found")
    upload = [] if a.inputs_from else inputs
    env = {"ONEGAP_INPUTS": " ".join(f.as_posix() for f in inputs), "ONEGAP_INPUTS_FROM": a.inputs_from or run_id,
           "ONEGAP_SHA": sha, "ONEGAP_RUN_ID": run_id, "ONEGAP_TARGET": a.target,
           "ONEGAP_VENV": a.venv, "ONEGAP_ARGS": extra, "ONEGAP_HF_WORK_REPO": a.work_repo,
           "ONEGAP_DIRECT": "1" if a.direct else "0", "HF_HOME": "/tmp/hf"}
    entry = (ROOT / "scripts" / "job_entry.sh").read_text(encoding="utf-8")
    print(f"run_id={run_id} flavor={a.flavor} venv={a.venv} target={a.target} args={extra!r} "
          f"work_repo={a.work_repo} dry_run={a.dry_run}")
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / f"{sha}.tar.gz"
        subprocess.check_call(["git", "archive", "--format=tar.gz", "-o", str(tar), "HEAD"], cwd=ROOT)
        print(f"bundle {tar.stat().st_size / 1024:.0f} KiB")
        if a.dry_run:  # no network calls at all below this point in a dry run
            return 0

        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=token)
        # One commit for bundle + inputs (the hub rate-limits commits), retried on network errors / 429.
        ops = [CommitOperationAdd(path_in_repo=f"bundles/{sha}.tar.gz", path_or_fileobj=str(tar))]
        ops += [CommitOperationAdd(path_in_repo=f"inputs/{run_id}/{f.as_posix()}", path_or_fileobj=str(ROOT / f))
                for f in upload]
        if a.inputs_from:  # same bytes as the earlier upload, or refuse (reproducibility)
            for f in inputs:
                path = f"inputs/{a.inputs_from}/{f.as_posix()}"
                remote = api.get_paths_info(a.work_repo, [path], repo_type="dataset", expand=True)
                data = (ROOT / f).read_bytes()
                lfs = getattr(remote[0], "lfs", None) if remote else None
                if lfs is not None:
                    same = lfs.sha256 == hashlib.sha256(data).hexdigest()
                else:  # regular git blob: compare the git blob id
                    git_id = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324
                    same = bool(remote) and getattr(remote[0], "blob_id", None) == git_id
                if not same:
                    sys.exit(f"--inputs-from {a.inputs_from}: {f} differs from the uploaded copy; "
                             "upload it again with --inputs")
            print(f"reusing {len(inputs)} input file(s) from inputs/{a.inputs_from}/ (hash match)")
        for attempt in range(5):
            try:
                api.create_commit(a.work_repo, ops, commit_message=f"launch {run_id}", repo_type="dataset")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"upload attempt {attempt + 1} failed ({type(e).__name__}); retrying in {30 * (attempt + 1)}s")
                time.sleep(30 * (attempt + 1))
        if upload:
            print(f"uploaded {len(inputs)} input file(s) to {a.work_repo}:inputs/{run_id}/")

    from huggingface_hub import run_job  # signature verified against huggingface_hub 2.4.0

    secrets = {"HF_TOKEN": token}
    # Llama Guard gates were accepted on a different HF account: its token is used only for those downloads.
    if os.environ.get("HF_TOKEN_GATED"):
        secrets["HF_TOKEN_GATED"] = os.environ["HF_TOKEN_GATED"]
    job = run_job(image=IMAGE, command=["bash", "-c", entry], env=env, secrets=secrets,
                  flavor=a.flavor, timeout=a.timeout, name=f"onegap-{a.target}-{a.venv}",
                  # label key "sha" is rejected by the Jobs API (400, 2026-11-10); sha lives in env
                  labels={"project": "onegap", "target": a.target, "venv": a.venv})
    print(f"job_id={job.id}\nurl={job.url}")
    print(f"status/logs: uv run python scripts/job_logs.py {job.id}")
    print(f"pull results: uv run python scripts/pull_run.py {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
