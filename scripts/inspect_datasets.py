"""VERIFY helper (spec 7.1): card metadata, licence, gating, configs/splits, features and a first row per dataset.

    python scripts/inspect_datasets.py [repo_id[:config] ...]   -> data/raw/_inspect/<repo>.json + stdout summary
Row text is truncated; nothing here is published.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onegap.env import load_dotenv  # noqa: E402

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

DEFAULT = [
    "marslabucla/XGuard-Train",
    "SafeMTData/SafeMTData",
    "walledai/XSTest",
    "natolambert/xstest-v2-copy",
    "Paul/XSTest",
    "CohereLabs/Global-MMLU",
    "openlanguagedata/flores_plus",
    "google/IFEval",
]


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def trunc(x, n=160):
    if isinstance(x, str):
        return x[:n] + ("…" if len(x) > n else "")
    if isinstance(x, list):
        return [trunc(v, n) for v in x[:6]] + (["…"] if len(x) > 6 else [])
    if isinstance(x, dict):
        return {k: trunc(v, n) for k, v in list(x.items())[:20]}
    return x


def inspect(repo: str) -> dict:
    api = HfApi()
    out: dict = {"repo": repo}
    try:
        info = api.dataset_info(repo)
    except Exception as e:
        return {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    card = info.card_data.to_dict() if info.card_data else {}
    out.update(gated=info.gated, private=info.private, license=card.get("license"),
               tags=[t for t in (info.tags or []) if t.startswith(("license:", "language:", "task_categories:"))],
               sha=info.sha)
    try:
        readme = Path(hf_hub_download(repo, "README.md", repo_type="dataset")).read_text(encoding="utf-8")
        out["readme_head"] = readme[:2500]
    except Exception as e:
        out["readme_head"] = f"<{type(e).__name__}>"
    q = urllib.parse.quote(repo, safe="")
    try:
        splits = get(f"https://datasets-server.huggingface.co/splits?dataset={q}")["splits"]
        out["splits"] = splits[:40]
        seen = set()
        out["first_rows"] = {}
        for s in splits:
            key = s["config"]
            if key in seen or len(seen) >= 6:
                continue
            seen.add(key)
            try:
                fr = get("https://datasets-server.huggingface.co/first-rows?dataset="
                         f"{q}&config={urllib.parse.quote(s['config'])}&split={urllib.parse.quote(s['split'])}")
                out["first_rows"][f"{s['config']}/{s['split']}"] = {
                    "features": [(f["name"], f["type"].get("_type", str(f["type"])[:80])) for f in fr["features"]],
                    "row0": trunc(fr["rows"][0]["row"]) if fr["rows"] else None,
                }
            except Exception as e:
                out["first_rows"][f"{s['config']}/{s['split']}"] = f"<{type(e).__name__}: {str(e)[:120]}>"
        try:
            size = get(f"https://datasets-server.huggingface.co/size?dataset={q}")
            out["num_rows_per_split"] = [(c["config"], c["split"], c["num_rows"])
                                         for c in size["size"].get("splits", [])][:40]
        except Exception as e:
            out["num_rows_per_split"] = f"<{type(e).__name__}>"
    except Exception as e:
        out["splits"] = f"<{type(e).__name__}: {str(e)[:160]}>"
    return out


def main() -> int:
    repos = sys.argv[1:] or DEFAULT
    d = ROOT / "data" / "raw" / "_inspect"
    d.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        o = inspect(repo)
        (d / (repo.replace("/", "__") + ".json")).write_text(json.dumps(o, indent=2, ensure_ascii=False),
                                                             encoding="utf-8")
        if "error" in o:
            print(f"## {repo}: {o['error']}\n")
            continue
        print(f"## {repo}  gated={o['gated']} license={o['license']} sha={o['sha'][:10]}")
        print(f"   rows: {o.get('num_rows_per_split')}")
        fr = o.get("first_rows")
        if isinstance(fr, dict):
            for k, v in list(fr.items())[:3]:
                feats = v["features"] if isinstance(v, dict) else v
                print(f"   {k}: {feats}")
        else:
            print(f"   splits: {o.get('splits')}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
