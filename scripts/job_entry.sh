#!/usr/bin/env bash
# Runs inside an HF Job. Inputs come from env set by scripts/launch_job.py.
set -euo pipefail
: "${ONEGAP_SHA:?}" "${ONEGAP_RUN_ID:?}" "${ONEGAP_TARGET:?}" "${ONEGAP_VENV:?}" "${ONEGAP_HF_WORK_REPO:?}"

START=$(date +%s)
pip install -q "huggingface_hub>=0.30" uv
mkdir -p /work && cd /work
python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["ONEGAP_HF_WORK_REPO"], f"bundles/{os.environ['ONEGAP_SHA']}.tar.gz", repo_type="dataset")
tarfile.open(p).extractall("/work")
# Untracked inputs (e.g. data/generated/*.jsonl) shipped via the private work repo, restored to the same paths.
for rel in os.environ.get("ONEGAP_INPUTS", "").split():
    src = hf_hub_download(os.environ["ONEGAP_HF_WORK_REPO"], f"inputs/{os.environ.get('ONEGAP_INPUTS_FROM') or os.environ['ONEGAP_RUN_ID']}/{rel}",
                          repo_type="dataset")
    dst = os.path.join("/work", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import shutil
    shutil.copy(src, dst)
    print(f"input restored: {rel}")
PY

apt-get update -qq && apt-get install -y -qq make git >/dev/null
uv venv -q --python 3.12 ".venv-${ONEGAP_VENV}"
source ".venv-${ONEGAP_VENV}/bin/activate"
uv pip install -q -r "requirements-${ONEGAP_VENV}.txt"
uv pip freeze > "/work/pip-freeze-${ONEGAP_VENV}.txt"

export ONEGAP_RUN_DIR="runs/${ONEGAP_RUN_ID}"
mkdir -p "$ONEGAP_RUN_DIR"
cp "/work/pip-freeze-${ONEGAP_VENV}.txt" "$ONEGAP_RUN_DIR/"
nvidia-smi > "$ONEGAP_RUN_DIR/nvidia-smi.txt" 2>&1 || true

set +e
if [ "${ONEGAP_DIRECT:-0}" = "1" ]; then
  # Commands without a Makefile target (publish-hub). ONEGAP_ARGS is split on spaces on purpose.
  # shellcheck disable=SC2086
  python -m onegap.cli "$ONEGAP_TARGET" ${ONEGAP_ARGS:-} 2>&1 | tee "$ONEGAP_RUN_DIR/log.txt"
else
  make "$ONEGAP_TARGET" PY=python ARGS="${ONEGAP_ARGS:-}" 2>&1 | tee "$ONEGAP_RUN_DIR/log.txt"
fi
STATUS=${PIPESTATUS[0]}
set -e
echo "{\"status\": $STATUS, \"seconds\": $(( $(date +%s) - START )), \"sha\": \"$ONEGAP_SHA\"}" > "$ONEGAP_RUN_DIR/job_status.json"

python - <<'PY'
import os, time
from huggingface_hub import HfApi
rid = os.environ["ONEGAP_RUN_ID"]
# The hub rate-limits commits (HTTP 429); results must not be lost, so retry for up to ~20 minutes.
for attempt in range(8):
    try:
        HfApi().upload_folder(folder_path=f"runs/{rid}", path_in_repo=f"runs/{rid}",
                              repo_id=os.environ["ONEGAP_HF_WORK_REPO"], repo_type="dataset")
        print("run uploaded")
        break
    except Exception as e:
        print(f"upload attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(60 * (attempt + 1) if attempt < 4 else 300)
else:
    raise SystemExit("run upload failed after retries")
PY
exit "$STATUS"
