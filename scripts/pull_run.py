"""Pull runs/<run_id>/ from the private working repo.

    python scripts/pull_run.py <run_id> [<run_id> ...] [--light]

--light skips adapter weights, tokenizer files and the unlabelled transcript copy: it fetches only what
`onegap report` and the run log need (manifests, logs, job status, labelled transcripts, adapter_config.json).
Each run is retried three times; the exit code is the number of runs that could not be pulled.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onegap.env import load_dotenv  # noqa: E402

LIGHT = ("*.json", "*.txt", "transcripts.jsonl")
# hub patterns are fnmatch: "*" also crosses "/", so "*.json" would fetch the 21 MB adapter/tokenizer.json too
LIGHT_SKIP = ("*/adapter/tokenizer*", "*.safetensors", "*/transcripts.unlabelled.jsonl")


def patterns(run_id: str, light: bool) -> tuple[list[str], list[str]]:
    """(allow, ignore) patterns for snapshot_download."""
    if not light:
        return [f"runs/{run_id}/*"], []
    return [f"runs/{run_id}/{p}" for p in LIGHT], list(LIGHT_SKIP)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--light", action="store_true", help="manifests, logs and labelled transcripts only")
    a = p.parse_args()
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    repo = os.environ.get("ONEGAP_HF_WORK_REPO")
    if not repo:
        sys.exit("set ONEGAP_HF_WORK_REPO")
    from huggingface_hub import snapshot_download

    failed = 0
    for rid in a.run_ids:
        for attempt in range(3):
            try:
                allow, ignore = patterns(rid, a.light)
                snapshot_download(repo, repo_type="dataset", allow_patterns=allow, ignore_patterns=ignore,
                                  local_dir=ROOT)
                print(ROOT / "runs" / rid, flush=True)
                break
            except Exception as e:  # noqa: BLE001 - flaky link: report and retry
                print(f"{rid}: attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
        else:
            failed += 1
    return failed


if __name__ == "__main__":
    sys.exit(main())
