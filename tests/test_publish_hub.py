"""publish_hub.py: what the upload job would put on the Hub, tested with local folders and a fake API."""

import json

import pytest

from onegap import publish_hub as H
from onegap.cli import main as cli_main
from onegap.schema import read_jsonl, write_jsonl


def turn(i, label, text="reply"):
    return {"turn": i, "user": f"user {i}", "assistant": f"{text} {i}", "assistant_en": f"english {i}", "label": label,
            "judge": {"unsafe_raw": label == "unsafe", "codes": []}}


def make_world(tmp_path, base_model=H.BASE_MODEL):
    """runs/ with one evaluated adapter (E), its training run, a circuits run and a cross-check run; plus cards."""
    runs = tmp_path / "runs"
    ev, tr, ck, cc = runs / "eval-E", runs / "train-E", runs / "circ", runs / "cross"
    write_jsonl(ev / "transcripts.jsonl", [
        {"conv_id": "xg-1-hi", "seed_id": "xg-1", "language": "hi", "kind": "attack", "condition": "E",
         "category": "fraud_scam", "turns": [turn(1, "deflected"), turn(2, "unsafe")]},
        {"conv_id": "gen-benign-1-hi", "seed_id": "gen-benign-1", "language": "hi", "kind": "benign", "condition": "E",
         "category": None, "turns": [turn(1, "deflected")]}])
    (ev / "run_manifest.json").write_text(json.dumps({"condition": "E", "adapter_run": "train-E"}))
    (tr / "adapter").mkdir(parents=True)
    (tr / "adapter" / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base_model, "r": 16,
                                                                     "target_modules": ["a", "b"]}))
    (tr / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (tr / "adapter" / "tokenizer.json").write_text("{}")
    (tr / "train_manifest.json").write_text("{}")
    ck.mkdir()
    for name in H.CIRCUIT_FILES:
        (ck / name).write_text(json.dumps({"scores": [0.1, 0.2], "note": "ok"}))
    cc.mkdir()
    (cc / "safetune_crosscheck.json").write_text("{}")
    (ck / "ck_pairs_en.csv").write_text("harmful prompt text that must stay private")
    cards = tmp_path / "cards"
    cards.mkdir()
    for n in ("model_E.md", "dataset_transcripts.md", "dataset_circuits.md"):
        (cards / n).write_text(f"# {n}")
    return runs, cards


class FakeApi:
    def __init__(self):
        self.calls, self.files = [], {}

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.calls.append(("create_repo", repo_id, repo_type, private))

    def upload_folder(self, repo_id, repo_type, folder_path, commit_message=None):
        from pathlib import Path

        self.files[repo_id] = sorted(p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*") if p.is_file())
        self.calls.append(("upload_folder", repo_id))

    def list_repo_files(self, repo_id, repo_type=None):
        return self.files[repo_id]

    def create_collection(self, title, namespace=None, description=None, private=False, exists_ok=False):
        if "data" in title:
            raise RuntimeError("collections API down")
        return type("C", (), {"slug": f"{namespace}/{title.lower().replace(' ', '-')}-abc"})()

    def add_collection_item(self, slug, item_id, item_type, exists_ok=False):
        self.calls.append(("add_item", slug, item_id, item_type))


def test_model_folder_holds_only_adapter_and_card(tmp_path):
    runs, cards = make_world(tmp_path)
    info = H.build_model(runs / "train-E", tmp_path / "m", cards / "model_E.md")
    assert info["problems"] == [] and info["target_modules"] == 2 and len(info["weights_sha256"]) == 64
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
    runs2, cards2 = make_world(tmp_path / "other", base_model="/work/local/path")
    assert H.build_model(runs2 / "train-E", tmp_path / "m2", cards2 / "model_E.md")["problems"]


def test_circuits_never_take_prompt_pairs_and_reject_long_strings(tmp_path):
    runs, cards = make_world(tmp_path)
    info = H.build_circuits(runs / "circ", runs / "cross", tmp_path / "c", cards / "dataset_circuits.md")
    assert info["problems"] == [] and "ck_pairs_en.csv" not in [p.name for p in (tmp_path / "c").iterdir()]
    (runs / "circ" / "scores_en.json").write_text(json.dumps({"prompt": "x" * 400}))
    info = H.build_circuits(runs / "circ", runs / "cross", tmp_path / "c2", cards / "dataset_circuits.md")
    assert any("scores_en.json" in p for p in info["problems"]) and not (tmp_path / "c2" / "scores_en.json").exists()
    assert H.long_strings({"a": [{"b": "y" * 161}]}) == ["/a[0]/b (161 chars)"]


def test_transcripts_are_staged_checked_and_carry_no_weights(tmp_path):
    runs, cards = make_world(tmp_path)
    rep = tmp_path / "dev_v1.json"
    rep.write_text("{}")
    info = H.build_transcripts([runs / "eval-E"], tmp_path / "t", [rep], cards / "dataset_transcripts.md")
    assert info["problems"] == []
    rows = list(read_jsonl(tmp_path / "t" / "runs" / "eval-E" / "transcripts.jsonl"))
    assert "reply 2" not in json.dumps(rows) and "user 1" not in json.dumps(rows[0]) and rows[1]["turns"][0]["user"] == "user 1"
    assert (tmp_path / "t" / "runs" / "train-E" / "train_manifest.json").exists()
    assert not list((tmp_path / "t").rglob("*.safetensors")) and (tmp_path / "t" / "report" / "dev_v1.json").exists()
    assert H.train_runs([runs / "eval-E"]) == {"E": "train-E"}


def test_upload_and_collections_with_a_fake_api(tmp_path):
    runs, cards = make_world(tmp_path)
    H.build_model(runs / "train-E", tmp_path / "m", cards / "model_E.md")
    api = FakeApi()
    targets = [{"repo_id": "org/x-lora", "repo_type": "model", "folder": tmp_path / "m"}]
    done = H.upload(api, targets, private=True)
    assert api.calls[0] == ("create_repo", "org/x-lora", "model", True)
    assert done[0]["files"] == ["README.md", "adapter_config.json", "adapter_model.safetensors"] and done[0]["private"]
    cols = H.collections(api, "org", targets + [{"repo_id": "org/d", "repo_type": "dataset", "folder": tmp_path}], True)
    assert cols[0]["slug"] == "org/onegap-models-abc" and cols[0]["items"] == ["org/x-lora"]
    assert "error" in cols[1]  # reported, not raised


def test_dry_run_from_the_cli_uploads_nothing(tmp_path, monkeypatch, capsys):
    runs, cards = make_world(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONEGAP_HF_WORK_REPO", raising=False)
    monkeypatch.setenv("ONEGAP_RUN_DIR", str(tmp_path / "runs" / "pub"))
    rc = cli_main(["publish-hub", "--org", "org", "--eval-runs", "eval-E", "--circuit-run", "circ", "--crosscheck-run",
                   "cross", "--cards", str(cards)])
    out = capsys.readouterr().out
    assert rc == 0 and "dry run: nothing created" in out
    man = json.loads((tmp_path / "runs" / "pub" / "publish_manifest.json").read_text())
    assert [p["repo_id"] for p in man["plan"]] == ["org/onegap-transcripts-and-logs", "org/onegap-circuits",
                                                   "org/onegap-tiny-aya-global-circuit-lora"]
    assert "uploaded" not in man and man["problems"] == []
    assert not (tmp_path / "runs" / "pub" / "stage" / "onegap-tiny-aya-global-circuit-lora").exists()
    # a leak stops everything with exit code 1
    write_jsonl(runs / "eval-E" / "transcripts.jsonl", [{"conv_id": "xg-1-hi", "seed_id": "xg-1", "language": "hi",
                "kind": "attack", "condition": "E", "category": "x", "turns": [turn(1, "unsafe")]}])
    (runs / "circ" / "scores_hi.json").write_text(json.dumps({"p": "z" * 999}))
    assert cli_main(["publish-hub", "--org", "org", "--eval-runs", "eval-E", "--circuit-run", "circ", "--cards", str(cards)]) == 1
    assert "STOPPED, nothing uploaded" in capsys.readouterr().out


def test_visibility_needs_a_manifest(tmp_path):
    with pytest.raises(SystemExit):
        cli_main(["publish-visibility"])
