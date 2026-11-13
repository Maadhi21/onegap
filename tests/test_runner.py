from pathlib import Path

import pytest

from onegap import runner as R
from onegap.guard import as_parts


def rec(sid, lang="en", turns=("u1", "u2", "u3"), split="dev", kind="attack"):
    return {"conv_id": f"{sid}-{lang}", "seed_id": sid, "split": split, "language": lang, "kind": kind,
            "family": "xguard", "category": "fraud_scam", "user_turns": list(turns), "provenance": {}}


def echo(convs):
    return [f"reply to {c[-1]['content']} after {len(c)} msgs" for c in convs]


def test_build_messages_alternates_and_ends_on_user():
    m = R.build_messages("S", ["u1", "u2", "u3"], ["a1"])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert m[-1]["content"] == "u2"


def test_replay_carries_history_and_handles_uneven_lengths():
    out = R.replay([rec("a"), rec("b", turns=("x",))], "S", echo, log=lambda _: None)
    assert out[0]["replies"] == ["reply to u1 after 2 msgs", "reply to u2 after 4 msgs", "reply to u3 after 6 msgs"]
    assert out[1]["replies"] == ["reply to x after 2 msgs"]


def test_guard_blocks_without_calling_model():
    calls = []

    def gen(convs):
        calls.append(len(convs))
        return echo(convs)

    out = R.replay([rec("a", turns=("ok", "BAD"))], "S", gen, screen=lambda cs: ["BAD" in c[-1]["content"] for c in cs],
                   refusal_text="no", log=lambda _: None)
    assert out[0]["replies"][1] == "no" and out[0]["blocked"] == [False, True] and calls == [1]
    tr = R.to_transcript(out[0]["rec"], out[0]["replies"], out[0]["blocked"], "G")
    assert tr["turns"][1]["guard_blocked"] is True and "guard_blocked" not in tr["turns"][0]


def test_select_s0_and_limit_per_language():
    recs = [rec("a"), rec("b"), rec("a", "hi"), rec("c", kind="overrefusal"), rec("t", split="test")]
    s0 = R.select_records(recs, "dev", None, None, final_turn_only=True)
    assert all(r["user_turns"] == ["u3"] and r["kind"] == "attack" for r in s0) and len(s0) == 3
    assert [r["conv_id"] for r in R.select_records(recs, "dev", None, 1)] == ["a-en", "a-hi"]
    assert len(R.select_records(recs, "dev", "en", None)) == 3


def test_apply_labels_unsafe_in_english_view_wins():
    done = R.replay([rec("a", "hinglish", turns=("u1", "u2"))], "S", lambda cs: ["Sorry, I can't help"] * len(cs),
                    log=lambda _: None)
    trs = [R.to_transcript(d["rec"], d["replies"], d["blocked"], "A") for d in done]
    safe = {"unsafe": False, "codes": []}
    R.apply_labels(trs, [safe, safe], {(0, 1): {"unsafe": True, "codes": ["S2"]}})
    assert [t["label"] for t in trs[0]["turns"]] == ["refused", "unsafe"]
    assert trs[0]["turns"][1]["judge"]["unsafe_en"] is True


def test_english_view_skips_english_and_guard_blocked_turns():
    out = R.replay([rec("a", "hi", turns=("ok", "BAD")), rec("a", "en")], "S", echo,
                   screen=lambda cs: ["BAD" in c[-1]["content"] for c in cs], refusal_text="no", log=lambda _: None)
    trs = [R.to_transcript(d["rec"], d["replies"], d["blocked"], "G") for d in out]
    assert R.english_view_items(trs, ["hi", "hinglish", "bn"]) == [(0, 0)]
    assert set(R.ENGLISH_VIEW_TRANSLATOR) == {"hi", "hinglish", "bn"}


def test_english_view_conversation_is_fully_english():
    tr = {"turns": [{"turn": 1, "user": "bn1", "assistant": "bnA1", "assistant_en": "A1"},
                    {"turn": 2, "user": "bn2", "assistant": "I can't help", "guard_blocked": True},
                    {"turn": 3, "user": "bn3", "assistant": "bnA3", "assistant_en": "A3"}]}
    c = R.english_view_conversation(tr, 2, ["e1", "e2", "e3"])
    assert [m["content"] for m in c] == ["e1", "A1", "e2", "I can't help", "e3", "A3"]
    # S0 keeps only the final turn; the English source still has all turns
    s0 = {"turns": [{"turn": 1, "user": "bn3", "assistant": "x", "assistant_en": "X"}]}
    assert [m["content"] for m in R.english_view_conversation(s0, 0, ["e1", "e2", "e3"])] == ["e3", "X"]
    assert R.english_view_conversation(s0, 0, None)[0]["content"] == "bn3"


def test_test_split_requires_frozen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        R.data_path("test")
    assert R.data_path("dev").name == "conversations.jsonl"


def test_guard_parts_format():
    assert as_parts([{"role": "user", "content": "hi"}]) == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_select_records_kind_filter():
    from onegap.runner import select_records

    recs = [{"conv_id": f"c{i}", "split": "train", "language": "en", "kind": k, "user_turns": ["a", "b"]}
            for i, k in enumerate(["attack", "benign", "benign"])]
    assert [r["conv_id"] for r in select_records(recs, "train", None, None, kind="benign")] == ["c1", "c2"]
    assert len(select_records(recs, "train", None, None)) == 3


def test_resolve_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert R.resolve_adapter("runs/train_D/adapter", None) == "runs/train_D/adapter"
    assert R.resolve_adapter(None, None) is None
    with pytest.raises(ValueError):
        R.resolve_adapter(None, "rid")
    d = tmp_path / "runs" / "rid" / "adapter"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert R.resolve_adapter("runs/train_D/adapter", "rid") == str(Path("runs") / "rid" / "adapter")
    assert R.dir_sha256(d) and R.dir_sha256(None) is None
