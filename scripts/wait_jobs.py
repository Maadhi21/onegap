"""Block until the given HF Jobs leave SCHEDULING/RUNNING; print a one-line summary per job.

    python scripts/wait_jobs.py <job_id> [<job_id> ...] [--poll 60]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from onegap.env import load_dotenv  # noqa: E402

load_dotenv()
from huggingface_hub import inspect_job  # noqa: E402

ACTIVE = {"SCHEDULING", "RUNNING", "STARTING", "PENDING", "BUILDING"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("job_ids", nargs="+")
    p.add_argument("--poll", type=int, default=60)
    a = p.parse_args()
    last = {}
    while True:
        states = {}
        for jid in a.job_ids:
            try:
                states[jid] = inspect_job(job_id=jid)
            except Exception as e:  # transient API error: keep waiting
                print(f"{jid}: inspect failed ({e})", flush=True)
        for jid, j in states.items():
            if last.get(jid) != j.status.stage:
                print(f"{time.strftime('%H:%M:%S')} {jid} -> {j.status.stage}", flush=True)
                last[jid] = j.status.stage
        if states and all(str(j.status.stage).upper() not in ACTIVE for j in states.values()):
            break
        time.sleep(a.poll)
    print("=== done ===")
    for jid, j in states.items():
        mins = None
        if j.started_at and j.finished_at:
            mins = round((j.finished_at - j.started_at).total_seconds() / 60, 1)
        print(f"{jid} stage={j.status.stage} msg={j.status.message} flavor={j.flavor} "
              f"run_minutes={mins} durations={j.durations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
