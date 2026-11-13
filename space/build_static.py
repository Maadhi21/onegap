"""Build the static replay Space (free tier) from the PUBLISHED, redacted dataset; never from local run folders.

    uv run python space/build_static.py --out space/dist

Writes index.html, about.md, reports/*.md, figures/*.png and data/<split>-<language>.json, where each data file
maps kind -> [{"id", "label", "html"}] with the HTML pre-rendered by logic.render (tested in tests/test_space_logic.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from logic import KIND_NAME, LANG_NAME, conversation_ids, load_runs, outcome, render

HERE = Path(__file__).parent
DATASET = "maadhi/onegap-transcripts-and-logs"


def build(root: Path, out: Path) -> dict:
    data = load_runs(root)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    index: dict = {}
    for split in data:
        for lang in LANG_NAME:
            per_kind = {}
            for kind in KIND_NAME:
                items = []
                for cid in conversation_ids(data, split, lang, kind):
                    ref = data[split]["A" if "A" in data[split] else "S0"][cid]
                    items.append({"id": cid, "label": f"{cid} · {ref.get('category')} · base: {outcome(ref)}",
                                  "html": render(data, split, cid)})
                if items:
                    per_kind[kind] = items
            if per_kind:
                (out / "data" / f"{split}-{lang}.json").write_text(json.dumps(per_kind, ensure_ascii=False),
                                                                   encoding="utf-8")
                index.setdefault(split, {})[lang] = {k: len(v) for k, v in per_kind.items()}
    (out / "data" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    shutil.copytree(HERE / "figures", out / "figures")
    (out / "reports").mkdir()
    for name in ("dev_v1", "test_v1"):
        shutil.copy(root / "report" / f"{name}.md", out / "reports" / f"{name}.md")
    shutil.copy(HERE / "about.md", out / "about.md")
    shutil.copy(HERE / "index.html", out / "index.html")
    shutil.copy(HERE / "README.md", out / "README.md")
    return index


def main() -> int:
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dist"))
    a = ap.parse_args()
    root = Path(snapshot_download(DATASET, repo_type="dataset"))
    index = build(root, Path(a.out))
    print(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
