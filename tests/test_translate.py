from onegap.data.translate import plan_languages, script_share, split_sentences


def test_bengali_never_planned_for_train():
    assert plan_languages({"split": "train"}) == ["hi"]
    assert plan_languages({"split": "dev"}) == ["hi", "bn"]
    assert plan_languages({"split": "test"}) == ["hi", "bn"]


def test_split_sentences_chunks_long_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = split_sentences(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == text


def test_split_keeps_short_text():
    assert split_sentences("Hello there. How are you?") == ["Hello there. How are you?"]


def test_script_share():
    assert script_share("यह हिंदी है", "hi") == 1.0
    assert script_share("this is English", "hi") == 0.0
    # vowel signs (ि, ै) are not isalpha(), so only base letters count: 3 Devanagari of 9 letters
    assert 0.2 < script_share("UPI पिन kya है", "hi") < 0.5
