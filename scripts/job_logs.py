"""Status, durations and log tail of an HF Job:  python scripts/job_logs.py <job_id> [--tail N] [--follow]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from onegap.env import load_dotenv  # noqa: E402

load_dotenv()
from huggingface_hub import fetch_job_logs, inspect_job  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("job_id")
    p.add_argument("--tail", type=int, default=60)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    j = inspect_job(job_id=a.job_id)
    print(f"stage={j.status.stage} message={j.status.message} flavor={j.flavor}")
    print(f"created={j.created_at} started={j.started_at} finished={j.finished_at} durations={j.durations}")
    print("-" * 80)
    for line in fetch_job_logs(job_id=a.job_id, follow=a.follow, tail=a.tail):
        print(line, end="" if line.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
