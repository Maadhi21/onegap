import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))
import logic as L  # noqa: E402


def tr(cid, labels, kind="attack", lang="hinglish", user="[withheld: XGuard-Train text | sha256=ab]",
       reply="[withheld: attack-conversation reply | sha256=cd]"):
    return {"conv_id": cid, "seed_id": cid, "language": lang, "kind": kind, "category": "fraud_scam",
            "turns": [{"turn": i, "user": user, "assistant": reply, "label": lab}
                      for i, lab in enumerate(labels, start=1)]}


def data():
    return {"dev": {"S0": {"c1": tr("c1", ["unsafe"], reply="[REDACTED unsafe reply | sha256=ef]")},
                    "A": {"c1": tr("c1", ["deflected", "deflected", "unsafe"]),
                          "b1": tr("b1", ["deflected"], kind="benign", user="UPI fraud kaise pehchane?",
                                   reply="Pehle <b>sender</b> check karo.")},
                    "E": {"c1": tr("c1", ["deflected", "refused", "refused"]),
                          "b1": tr("b1", ["deflected"], kind="benign", user="x", reply="y")}}}


def test_aligned_turn_puts_the_single_prompt_on_the_last_row():
    d = data()["dev"]
    assert L.aligned_turn(d["S0"]["c1"], 0, 3) is None and L.aligned_turn(d["S0"]["c1"], 2, 3)["label"] == "unsafe"


def test_conversation_ids_filter_language_and_kind():
    assert L.conversation_ids(data(), "dev", "hinglish", "attack") == ["c1"]
    assert L.conversation_ids(data(), "dev", "hinglish", "benign") == ["b1"]


def test_render_shows_labels_hides_withheld_text_and_escapes_html():
    h = L.render(data(), "dev", "c1")
    assert "unsafe at turn 3" in h and "refused at the end" in h and h.count("unsafe</span>") >= 2
    assert "sha256" not in h and "withheld: XGuard-Train text" in h and "REDACTED unsafe reply" in h
    b = L.render(data(), "dev", "b1")
    assert "&lt;b&gt;sender&lt;/b&gt;" in b and "<b>sender</b>" not in b


def test_load_runs_reads_the_runs_a_report_lists(tmp_path):
    for name, split, run in (("dev_v1", "dev", "r1"), ("test_v1", "test", "r2")):
        (tmp_path / "report").mkdir(exist_ok=True)
        (tmp_path / "report" / f"{name}.json").write_text(json.dumps({"split": split, "runs": {"A": {"run_id": run}}}))
        (tmp_path / "runs" / run).mkdir(parents=True)
        (tmp_path / "runs" / run / "transcripts.jsonl").write_text(json.dumps(tr("c1", ["refused"])) + "\n")
    d = L.load_runs(tmp_path)
    assert set(d) == {"dev", "test"} and d["test"]["A"]["c1"]["turns"][0]["label"] == "refused"
