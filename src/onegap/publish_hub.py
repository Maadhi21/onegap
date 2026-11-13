"""Put the public copies on the Hugging Face Hub (M9). Meant to run as a CPU job: adapter weights and raw
transcripts live in the private work repo and are copied Hub to Hub, never through the dev box.

    onegap publish-hub --org <org> --eval-runs <run_id> [...] [--circuit-run <id>] [--crosscheck-run <id>]
                       [--report runs/report/dev_v1.json runs/report/dev_v1.md] [--execute] [--public]

Without --execute this is a dry run: it fetches the raw runs, builds the public copies (publish.py: unsafe replies
redacted, attack replies and XGuard-Train text withheld), checks them and prints what would be uploaded. With
--execute it creates the repos and uploads. Repos are PRIVATE unless --public is also given;
`onegap publish-visibility <publish_manifest.json> --public` flips them after someone has looked.

What goes where:
  <org>/onegap-transcripts-and-logs (dataset)  public transcripts, run and training manifests, report, card
  <org>/onegap-circuits (dataset)              node scores, faithfulness and overlap report, selected components,
                                               SafeTune cross-check, card. Never the prompt pairs (XGuard-Train text).
  <org>/onegap-tiny-aya-global-{circuit,plain,random}-lora (model)
                                               adapter_config.json, adapter_model.safetensors, card. No tokenizer or
                                               chat template: those belong to the gated base model.
Anything the checks flag stops the command before a single upload.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from onegap import publish

MODEL_REPO = {"E": "onegap-tiny-aya-global-circuit-lora", "D": "onegap-tiny-aya-global-plain-lora",
              "R": "onegap-tiny-aya-global-random-lora"}
TRANSCRIPTS_REPO, CIRCUITS_REPO = "onegap-transcripts-and-logs", "onegap-circuits"
BASE_MODEL = "CohereLabs/tiny-aya-global"
EVAL_FILES = ("transcripts.jsonl", "run_manifest.json", "job_status.json")
TRAIN_FILES = ("train_manifest.json", "train_log.json", "job_status.json", "adapter/adapter_config.json",
               "adapter/adapter_model.safetensors")
CIRCUIT_FILES = ("scores_en.json", "scores_hi.json", "scores_hinglish.json", "circuits_report.json",
                 "selected_components.json")
CROSSCHECK_FILES = ("safetune_crosscheck.json",)
MAX_STRING = 160  # circuit files hold numbers and names; a longer string could be a prompt
COLLECTIONS = {"model": ("OneGap models", "LoRA adapters D, E and R for Tiny Aya Global (Lexsi hackathon 2026, Track 2)"),
               "dataset": ("OneGap data", "Transcript labels, logs and refusal circuits (Lexsi hackathon 2026, Track 2)")}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def long_strings(obj, limit: int = MAX_STRING, path: str = "") -> list[str]:
    """Paths of string values longer than `limit` anywhere in a JSON value."""
    if isinstance(obj, str):
        return [f"{path} ({len(obj)} chars)"] if len(obj) > limit else []
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in long_strings(v, limit, f"{path}/{k}")]
    if isinstance(obj, list):
        return [x for i, v in enumerate(obj) for x in long_strings(v, limit, f"{path}[{i}]")]
    return []


def fetch(run_ids: list[str], files: tuple[str, ...], work_repo: str, root: Path) -> None:
    """Download the listed files of each run from the private work repo into root/runs/<run_id>/."""
    from huggingface_hub import snapshot_download

    need = [f"runs/{r}/{f}" for r in run_ids for f in files if not (root / "runs" / r / f).exists()]
    if need:
        snapshot_download(work_repo, repo_type="dataset", allow_patterns=need, local_dir=root)


def train_runs(eval_dirs: list[Path]) -> dict[str, str]:
    """condition -> training run id, read from the evaluation manifests (so the published adapter is the one that
    was evaluated)."""
    out = {}
    for d in eval_dirs:
        m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        if m.get("adapter_run") and m["condition"] in MODEL_REPO:
            out[m["condition"]] = m["adapter_run"]
    return out


def build_transcripts(eval_dirs: list[Path], out: Path, report_files: list[Path], card: Path,
                      attack_replies: str = "withheld") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    runs = [publish.stage_run(d, out, attack_replies) for d in eval_dirs]
    if report_files:
        (out / "report").mkdir(exist_ok=True)
        for f in report_files:
            shutil.copyfile(f, out / "report" / f.name)
    shutil.copyfile(card, out / "README.md")
    problems = publish.check_stage(out, attack_replies)
    problems += [f"{r['run_id']}: labels changed while staging" for r in runs if not r["labels_unchanged"]]
    problems += [str(p) for p in out.rglob("*") if p.suffix in (".safetensors", ".bin", ".pt", ".csv")]
    return {"runs": runs, "problems": problems}


def build_circuits(circuit_dir: Path, crosscheck_dir: Path | None, out: Path, card: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    problems, copied = [], []
    for src, names in ((circuit_dir, CIRCUIT_FILES), (crosscheck_dir, CROSSCHECK_FILES)):
        for name in names if src else ():
            f = src / name
            if not f.exists():
                problems.append(f"missing {f}")
                continue
            bad = long_strings(json.loads(f.read_text(encoding="utf-8")))
            if bad:
                problems.append(f"{name}: long strings, could be prompt text: {bad[:3]}")
                continue
            shutil.copyfile(f, out / name)
            copied.append(name)
    shutil.copyfile(card, out / "README.md")
    return {"files": copied, "problems": problems}


def build_model(train_dir: Path, out: Path, card: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    cfg_file, weights = train_dir / "adapter" / "adapter_config.json", train_dir / "adapter" / "adapter_model.safetensors"
    if not cfg_file.exists() or not weights.exists():
        return {"problems": [f"adapter files missing in {train_dir}"]}
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    if cfg.get("base_model_name_or_path") != BASE_MODEL:
        problems.append(f"{train_dir.name}: base model is {cfg.get('base_model_name_or_path')!r}, expected {BASE_MODEL}")
    shutil.copyfile(cfg_file, out / "adapter_config.json")
    shutil.copyfile(weights, out / "adapter_model.safetensors")
    shutil.copyfile(card, out / "README.md")
    extra = sorted(p.name for p in out.iterdir() if p.name not in ("adapter_config.json", "adapter_model.safetensors",
                                                                   "README.md"))
    if extra:
        problems.append(f"unexpected files in the model folder: {extra}")
    return {"train_run": train_dir.name, "weights_sha256": sha_file(out / "adapter_model.safetensors"),
            "weights_bytes": weights.stat().st_size, "r": cfg.get("r"), "target_modules": len(cfg.get("target_modules") or []),
            "problems": problems}


def upload(api, targets: list[dict], private: bool) -> list[dict]:
    """Create each repo (if needed) and upload its folder. Returns what was done, with the file list read back."""
    done = []
    for t in targets:
        api.create_repo(t["repo_id"], repo_type=t["repo_type"], private=private, exist_ok=True)
        api.upload_folder(repo_id=t["repo_id"], repo_type=t["repo_type"], folder_path=str(t["folder"]),
                          commit_message=t.get("message", "OneGap: first public version"))
        done.append({"repo_id": t["repo_id"], "repo_type": t["repo_type"], "private": private,
                     "files": sorted(api.list_repo_files(t["repo_id"], repo_type=t["repo_type"]))})
    return done


def collections(api, org: str, targets: list[dict], private: bool) -> list[dict]:
    """Best effort: a collection can also be made by hand in a minute, so a failure here is reported, not fatal."""
    out = []
    for kind, (title, description) in COLLECTIONS.items():
        items = [t["repo_id"] for t in targets if t["repo_type"] == kind]
        try:
            col = api.create_collection(title, namespace=org, description=description, private=private, exists_ok=True)
            for item in items:
                api.add_collection_item(col.slug, item_id=item, item_type=kind, exists_ok=True)
            out.append({"type": kind, "slug": col.slug, "url": f"https://huggingface.co/collections/{col.slug}",
                        "items": items, "private": private})
        except Exception as e:  # noqa: BLE001
            out.append({"type": kind, "error": f"{type(e).__name__}: {str(e)[:200]}", "items": items})
    return out


def main(args) -> int:
    root = Path(".")
    stage = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-publish")) / "stage"
    work_repo = os.environ.get("ONEGAP_HF_WORK_REPO")
    cards = Path(args.cards)
    if work_repo:
        fetch(args.eval_runs, EVAL_FILES, work_repo, root)
    eval_dirs = [root / "runs" / r for r in args.eval_runs]
    trains = train_runs(eval_dirs)
    if work_repo:
        fetch(sorted(trains.values()), TRAIN_FILES, work_repo, root)
        if args.circuit_run:
            fetch([args.circuit_run], CIRCUIT_FILES, work_repo, root)
        if args.crosscheck_run:
            fetch([args.crosscheck_run], CROSSCHECK_FILES, work_repo, root)

    manifest: dict = {"org": args.org, "execute": bool(args.execute), "public": bool(args.public), "problems": []}
    targets = []
    info = build_transcripts(eval_dirs, stage / TRANSCRIPTS_REPO, [Path(p) for p in args.report or []],
                             cards / "dataset_transcripts.md", args.attack_replies)
    manifest["transcripts"] = info
    targets.append({"repo_id": f"{args.org}/{TRANSCRIPTS_REPO}", "repo_type": "dataset", "folder": stage / TRANSCRIPTS_REPO})
    if args.circuit_run:
        info = build_circuits(root / "runs" / args.circuit_run,
                              root / "runs" / args.crosscheck_run if args.crosscheck_run else None,
                              stage / CIRCUITS_REPO, cards / "dataset_circuits.md")
        manifest["circuits"] = info
        targets.append({"repo_id": f"{args.org}/{CIRCUITS_REPO}", "repo_type": "dataset", "folder": stage / CIRCUITS_REPO})
    manifest["models"] = {}
    for cond, rid in sorted(trains.items()):
        info = build_model(root / "runs" / rid, stage / MODEL_REPO[cond], cards / f"model_{cond}.md")
        manifest["models"][cond] = info
        targets.append({"repo_id": f"{args.org}/{MODEL_REPO[cond]}", "repo_type": "model", "folder": stage / MODEL_REPO[cond]})
    for section in (manifest["transcripts"], manifest.get("circuits", {}), *manifest["models"].values()):
        manifest["problems"] += section.get("problems", [])
    manifest["plan"] = [{"repo_id": t["repo_id"], "repo_type": t["repo_type"],
                         "files": sorted(str(p.relative_to(t["folder"]).as_posix()) for p in Path(t["folder"]).rglob("*")
                                         if p.is_file())} for t in targets]
    out_file = stage.parent / "publish_manifest.json"
    if manifest["problems"]:
        out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("[publish-hub] STOPPED, nothing uploaded:\n  " + "\n  ".join(manifest["problems"]), flush=True)
        return 1
    for p in manifest["plan"]:
        print(f"[publish-hub] {p['repo_type']} {p['repo_id']}: {len(p['files'])} files: {p['files'][:12]}", flush=True)
    if args.execute:
        from huggingface_hub import HfApi

        api = HfApi()
        private = not args.public
        manifest["uploaded"] = upload(api, targets, private)
        manifest["collections"] = collections(api, args.org, targets, private)
        print(f"[publish-hub] uploaded {len(targets)} repos, private={private}; collections: "
              f"{[c.get('slug') or c.get('error') for c in manifest['collections']]}", flush=True)
    else:
        print("[publish-hub] dry run: nothing created, nothing uploaded. Add --execute to upload (private unless --public).")
    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in MODEL_REPO.values():  # the run folder goes back to the work repo: do not copy the weights again
        shutil.rmtree(stage / name, ignore_errors=True)
    return 0


def main_visibility(args) -> int:
    """Flip every repo and collection named in a publish manifest to public (or back to private)."""
    from huggingface_hub import HfApi

    api = HfApi()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    private = not args.public
    for r in manifest.get("uploaded", []):
        api.update_repo_settings(r["repo_id"], private=private, repo_type=r["repo_type"])
        print(f"[publish-visibility] {r['repo_type']} {r['repo_id']} private={private}", flush=True)
    for c in manifest.get("collections", []):
        if c.get("slug"):
            api.update_collection_metadata(c["slug"], private=private)
            print(f"[publish-visibility] collection {c['slug']} private={private}", flush=True)
    return 0
