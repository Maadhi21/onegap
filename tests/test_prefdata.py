from onegap import prefdata as P


def tr(sid, labels, lang="en", kind="attack", replies=None):
    replies = replies or [f"reply {i} " * 20 for i in range(1, len(labels) + 1)]
    return {"conv_id": f"{sid}-{lang}", "seed_id": sid, "language": lang, "kind": kind, "category": "fraud_scam",
            "turns": [{"turn": i, "user": f"u{i}", "assistant": r, "label": lab}
                      for i, (lab, r) in enumerate(zip(labels, replies, strict=True), start=1)]}


def test_plan_harmful_first_unsafe_and_cap():
    trs = [tr(f"f{i}", ["deflected", "unsafe", "unsafe"]) for i in range(7)]
    trs += [tr(f"n{i}", ["deflected"] * 3) for i in range(10)]
    trs += [tr("r", ["deflected", "refused"])]  # held by refusing: no pair
    plan = P.plan_harmful(trs)
    first = [c for c in plan if c["kind"] == "harmful"]
    final = [c for c in plan if c["kind"] == "harmful_final"]
    assert len(first) == 7 and all(c["turn"] == 2 for c in first)
    assert len(final) == 3 and all(c["turn"] == 3 for c in final)  # 3/10 = 30%
    assert all(c["tr"]["seed_id"] != "r" for c in plan)


def test_prompt_messages_includes_own_history_up_to_turn():
    m = P.prompt_messages("S", tr("a", ["deflected", "unsafe", "deflected"]), 2)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"] and m[-1]["content"] == "u2"


def test_plan_benign_prefers_real_overrefusals():
    trs = [tr("b1", ["deflected", "deflected"], kind="benign"), tr("b2", ["refused", "deflected"], kind="benign")]
    plan = P.plan_benign(trs, 1)
    assert plan[0]["kind"] == "benign_refused" and plan[0]["turn"] == 1


def test_gates():
    c = {"tr": tr("a", ["unsafe"], replies=["word " * 100]), "turn": 1, "kind": "harmful"}
    good = "I can't help with that, because it would enable fraud. " + "Here is how to report scams safely. " * 12
    assert P.gate(c, good) is None
    assert P.gate(c, "I can't help with that.") == "length"
    assert P.gate(c, "Sure, here are the details " * 20) == "no_refusal"
    h = {"tr": tr("a", ["unsafe"], lang="hi", replies=["शब्द " * 50]), "turn": 1, "kind": "harmful"}
    assert P.gate(h, "I cannot help " * 25) == "script"
    assert P.script_ok("main aapki madad nahi kar sakta", "hinglish")
    assert not P.script_ok("मैं मदद नहीं कर सकता", "hinglish")
    assert not P.script_ok("Sorry, I can't help with that request because it is harmful.", "hinglish")


def test_make_pair_sides():
    c = {"tr": tr("a", ["deflected"], kind="benign"), "turn": 1, "kind": "benign_ok"}
    p = P.make_pair(c, "S", "I'm sorry, I can't discuss this topic. " * 5)
    assert p["chosen"][0]["content"].startswith("reply 1") and p["rejected"][0]["content"].startswith("I'm sorry")
    assert p["meta"]["teacher_side"] == "rejected" and p["meta"]["rejected_unsafe"] is False
    assert p["meta"]["trimmed"] is None


def test_match_lengths_trims_the_longer_reply_at_sentence_ends():
    base = "One two three four five. " * 20  # 100 words, 20 sentences
    teacher = "I can't help with that request today, sorry. " * 6  # 48 words
    b, t, side = P.match_lengths(base, teacher)
    assert side == "base" and t == teacher and P.length_ok(b, t) and b.endswith("five.")
    assert P.match_lengths("w " * 100, teacher) is None  # one 100-word sentence: cannot cut
    assert P.trim_to_words("A b. C d e. F.", 3) == "A b."
    assert P.trim_to_words("एक दो। तीन चार।", 2) == "एक दो।"


def test_make_pair_records_trimmed_side():
    c = {"tr": tr("a", ["unsafe"], replies=["Step one is this. " * 30]), "turn": 1, "kind": "harmful"}
    p = P.make_pair(c, "S", "I can't help with that, it causes harm. " * 6)
    assert p["meta"]["trimmed"] == "rejected" and P.length_ok(p["chosen"][0]["content"], p["rejected"][0]["content"])


def test_benign_ok_accepts_soft_deflection_but_not_an_answer():
    c = {"tr": tr("a", ["deflected"], kind="benign", replies=["Use WPA3 and update firmware. " * 10]),
         "turn": 1, "kind": "benign_ok"}
    soft = ("Discussing specific router settings could potentially lead to misuse or confusion. "
            "Please consult your manufacturer's official documentation instead. ") * 3
    assert P.gate(c, soft) is None
    assert P.gate(c, "Sure: enable WPA3, change the default password, update firmware. " * 5) == "refusal_pair_unclear"
    assert P.looks_like_deflection("Yaar, ye topic thoda sensitive hai, galat jaankaari ho sakti hai.", "hinglish")
    assert P.looks_like_deflection("ऐसे विषयों पर चर्चा करना संवेदनशील हो सकता है।", "hi")


def test_near_duplicates():
    assert P.near_duplicates(["a b c d e f", "a b c d e f", "x y z w v u"]) == {1}


def test_base_gate_script_and_real_refusal():
    assert P.base_gate("benign_ok", "हाँ, ये संकेत हैं जो बताते हैं कि बच्चे ने कुछ छुआ है", "hinglish") == "base_script"
    assert P.base_gate("benign_ok", "Haan, kuch signs hai jo batate hai ki bachche ne kuch chhua hai", "hinglish") is None
    assert P.base_gate("benign_ok", "Yes, here are the signs to watch for.", "hinglish") is None  # Latin is fine
    assert P.base_gate("harmful", "हाँ, ये रहे तरीके", "hinglish") is None  # rejected side: script not checked
    assert P.base_gate("benign_refused", "हाँ, अपने नेटवर्क से जुड़े उपकरणों का लॉग रखना अच्छा विचार है।", "hi") \
        == "base_not_refusal"
    assert P.base_gate("benign_refused", "मैं एक AI सहायक हूँ और चिकित्सा सलाह नहीं दे सकता।", "hi") is None


def test_pair_gate_uses_base_side_and_exclusions():
    def pair(kind, side, chosen, rejected, lang="en"):
        return {"chosen": [{"content": chosen}], "rejected": [{"content": rejected}],
                "meta": {"conv_id": "c-en", "turn": 2, "kind": kind, "teacher_side": side, "language": lang}}
    ok = pair("benign_ok", "rejected", "Here is how to report it.", "I can't discuss this, it could lead to misuse.")
    assert P.pair_gate(ok, {}) is None
    assert P.pair_gate(ok, {("c-en", 2): "bad"}) == "human_excluded"
    bad = pair("benign_ok", "rejected", "यह रहा तरीका", "Sorry yaar, main is pe baat nahi kar sakta", lang="hinglish")
    assert P.pair_gate(bad, {}) == "base_script"


def test_load_exclusions(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("# header\nconv-a 3 medical dose  # note\n\n", encoding="utf-8")
    assert P.load_exclusions(f) == {("conv-a", 3): "medical dose"}


def test_hindi_refusal_detector_ignores_helpful_text():
    from onegap.judges import looks_like_refusal
    assert not looks_like_refusal("हाँ, आप इसका प्रबंधन कर सकती हैं और डॉक्टर से बात कर सकती हैं।", "hi")
    assert not looks_like_refusal("कोई भी वैध कंपनी गारंटी नहीं दे सकता, इसलिए सावधान रहें।", "hi")
    assert looks_like_refusal("मैं आपको इस बारे में विस्तृत जानकारी नहीं दे सकता, क्योंकि यह अवैध है।", "hi")
    assert looks_like_refusal("मैं एक AI सहायक हूँ और चिकित्सा सलाह नहीं दे सकता।", "hi")
    assert not looks_like_refusal("Aap isko ignore nahi kar sakte, report karna zaroori hai.", "hinglish")
    assert looks_like_refusal("Yaar, main iske detailed steps provide nahi kar sakta, kyunki ye illegal hai.", "hinglish")
