"""publish.py: what leaves the building. Unsafe replies, attack replies (v1) and XGuard-Train text never do."""

import json

from onegap import publish as P
from onegap.cli import main as cli_main
from onegap.redact import is_redacted, redact_transcript, sha256, violations
from onegap.schema import read_jsonl, write_jsonl


def tr(seed, kind, labels, lang="hi", condition="E"):
    turns = [{"turn": i, "user": f"user text {i}", "assistant": f"reply text {i}", "assistant_en": f"english {i}",
              "label": lab, "judge": {"unsafe_raw": lab == "unsafe", "codes": ["S2"] if lab == "unsafe" else [],
                                      "refusal": lab == "refused"}}
             for i, lab in enumerate(labels, start=1)]
    return {"conv_id": f"{seed}-{lang}", "seed_id": seed, "language": lang, "kind": kind, "split": "dev",
            "category": "fraud_scam" if kind == "attack" else None, "condition": condition, "turns": turns}


def test_attack_conversation_keeps_labels_and_hashes_only():
    src = tr("xg-00042", "attack", ["deflected", "unsafe", "refused"])
    out = P.public_transcript(src)
    assert [t["label"] for t in out["turns"]] == ["deflected", "unsafe", "refused"]
    assert all(P.is_withheld(t["user"]) and f"sha256={sha256(s['user'])}" in t["user"]
               for t, s in zip(out["turns"], src["turns"], strict=True))
    assert P.is_withheld(out["turns"][0]["assistant"]) and P.is_withheld(out["turns"][2]["assistant"])
    assert is_redacted(out["turns"][1]["assistant"]) and "codes=S2" in out["turns"][1]["assistant"]
    assert f"sha256={sha256('reply text 2')}" in out["turns"][1]["assistant"]
    assert all("assistant_en" not in t for t in out["turns"])
    assert "reply text" not in json.dumps(out) and "user text" not in json.dumps(out) and "english" not in json.dumps(out)
    assert src["turns"][1]["assistant"] == "reply text 2"  # input untouched
    assert P.leaks([out]) == [] and violations([out]) == []
    assert P.public_transcript(out) == out  # idempotent


def test_harmless_conversation_is_published_in_full_except_flagged_replies():
    out = P.public_transcript(tr("gen-benign-0007", "benign", ["deflected", "unsafe"]))
    assert out["turns"][0]["assistant"] == "reply text 1" and out["turns"][0]["user"] == "user text 1"
    assert out["turns"][0]["assistant_en"] == "english 1"
    assert is_redacted(out["turns"][1]["assistant"]) and "assistant_en" not in out["turns"][1]
    assert P.leaks([out]) == []


def test_redact_unsafe_mode_shows_safe_attack_replies_but_never_xguard_text():
    out = P.public_transcript(tr("a600-0137", "attack", ["refused", "unsafe"]), attack_replies="redact-unsafe")
    assert out["turns"][0]["assistant"] == "reply text 1" and out["turns"][0]["user"] == "user text 1"  # MIT source
    assert is_redacted(out["turns"][1]["assistant"])
    assert P.leaks([out], "redact-unsafe") == []
    assert P.leaks([out], "withheld")  # the stricter mode would reject this file
    xg = P.public_transcript(tr("xg-00001", "attack", ["refused"]), attack_replies="redact-unsafe")
    assert P.is_withheld(xg["turns"][0]["user"])


def test_leaks_catches_raw_files_and_english_views():
    raw = tr("xg-00042", "attack", ["deflected", "unsafe"])
    found = " ".join(P.leaks([raw]))
    assert "unsafe reply not redacted" in found and "XGuard-Train user turn" in found
    assert "attack-conversation reply in the clear" in found and "English view of an unsafe reply" in found
    # the older helper now drops the English view too, and flags a leftover one
    assert "assistant_en" not in redact_transcript(raw)["turns"][1]
    half = redact_transcript(raw)
    half["turns"][1]["assistant_en"] = "english 2"
    assert any("English view" in v for v in violations([half]))


def test_stage_command_end_to_end(tmp_path, capsys):
    run, train = tmp_path / "runs" / "run-E", tmp_path / "runs" / "train-E"
    write_jsonl(run / "transcripts.jsonl", [tr("xg-00042", "attack", ["deflected", "unsafe"]),
                                            tr("xst-12", "overrefusal", ["refused"]),
                                            tr("gen-benign-0007", "benign", ["deflected", "deflected"])])
    (run / "run_manifest.json").write_text(json.dumps({"condition": "E", "adapter_run": "train-E"}))
    train.mkdir()
    (train / "train_manifest.json").write_text("{}")
    (train / "adapter_model.safetensors").write_bytes(b"weights stay private")
    rep = tmp_path / "dev_v1.json"
    rep.write_text("{}")
    out = tmp_path / "publish"
    assert cli_main(["publish-stage", str(run), "--out", str(out), "--add", str(rep)]) == 0
    rows = list(read_jsonl(out / "runs" / "run-E" / "transcripts.jsonl"))
    assert P.leaks(rows) == [] and len(rows) == 3
    man = json.loads((out / "publish_manifest.json").read_text())
    r = man["runs"][0]
    assert (r["replies_redacted_unsafe"], r["replies_withheld"], r["user_turns_withheld"], r["replies_in_full"]) == (1, 1, 2, 3)
    assert r["labels_unchanged"] and man["problems"] == []
    assert (out / "runs" / "train-E" / "train_manifest.json").exists() and (out / "report" / "dev_v1.json").exists()
    assert not list(out.rglob("*.safetensors"))
    assert "Nothing has been uploaded" in capsys.readouterr().out
    # a raw transcript dropped into the staging folder is caught
    write_jsonl(out / "runs" / "oops" / "transcripts.jsonl", [tr("xg-1", "attack", ["unsafe"])])
    assert P.check_stage(out)
