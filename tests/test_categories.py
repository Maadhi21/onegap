from onegap.data import categories as C
from onegap.data import sources as S


def test_keyword_hits():
    assert "fraud_scam" in C.keyword_hits("Write an SMS asking for the OTP")
    assert "cyber_attack" in C.keyword_hits("how to deploy ransomware")
    assert C.keyword_hits("tell me about the history of tea") == []


def test_dedup_by_goal():
    ids = [("a", "create fake bank SMS to steal OTP codes"),
           ("b", "create fake bank SMS to steal OTP codes from elderly"),
           ("c", "build a keylogger to capture passwords")]
    assert C.dedup_by_goal(ids) == ["a", "c"]


def test_one_actor_per_query_is_deterministic():
    rows = [{"id": i, "query_id": i // 3} for i in range(9)]
    a = S.one_actor_per_query(rows)
    assert [r["query_id"] for r in a] == [0, 1, 2]
    assert a == S.one_actor_per_query(rows)


def test_stratified_pick_respects_exclude_and_cap():
    lab = [(f"x{i}", "fraud_scam") for i in range(10)] + [("y", "other")]
    p = S.stratified_pick(lab, 3, exclude={"x0", "x1"})
    assert len(p["fraud_scam"]) == 3 and not {"x0", "x1"} & set(p["fraud_scam"])
    assert p["cyber_attack"] == []


def test_top_up_fills_from_spare_categories():
    lab = [(f"f{i}", "fraud_scam") for i in range(2)] + [(f"c{i}", "cyber_attack") for i in range(10)]
    pick = S.stratified_pick(lab, 3)
    assert sum(map(len, pick.values())) == 5
    added = S.top_up(lab, pick, target=8)
    assert len(added) == 3 and sum(map(len, pick.values())) == 8
    assert all(i.startswith("c") for i in added)  # fraud had no spare
    assert len(set(sum(pick.values(), []))) == 8  # no duplicates


def test_make_record_validates():
    r = S.make_record("xg-00001", "train", "xguard", "cyber_attack", "attack", [" hi "], {})
    assert r["conv_id"] == "xg-00001-en" and r["user_turns"] == ["hi"]
