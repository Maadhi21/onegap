import os

from onegap.env import load_dotenv


def test_load_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# c\nA_KEY="x y"\nexport B_KEY=2\nUser.Name=ignored\nEMPTY=\nPRESET=new\n', encoding="utf-8")
    monkeypatch.setenv("PRESET", "old")
    for k in ("A_KEY", "B_KEY", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    loaded = load_dotenv(f)
    assert sorted(loaded) == ["A_KEY", "B_KEY"]
    assert os.environ["A_KEY"] == "x y" and os.environ["PRESET"] == "old"
    assert "EMPTY" not in os.environ
