"""List recent HF Jobs in the token's namespace:  python scripts/list_jobs.py [N]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from onegap.env import load_dotenv  # noqa: E402

load_dotenv()
from huggingface_hub import list_jobs  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
jobs = sorted(list_jobs(), key=lambda j: j.created_at, reverse=True)[:n]
print(f"{len(jobs)} job(s)")
for j in jobs:
    print(f"{j.id}  {j.status.stage:10s} {j.flavor:12s} created={j.created_at:%Y-%m-%d %H:%M} "
          f"labels={j.labels} durations={j.durations}")
