"""Cohere chat wrapper for a TRIAL key (~1,000 calls/month, 20/min).

- Every request is cached on disk by hash of (model, messages, schema, temperature): re-runs cost 0 calls.
- Live calls are throttled to <= RATE_PER_MIN and appended to data/generated/cohere_ledger.jsonl.
- A hard monthly budget guard refuses to exceed BUDGET calls (counted from the ledger for the month).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "generated" / "cohere_cache"
LEDGER = ROOT / "data" / "generated" / "cohere_ledger.jsonl"
RATE_PER_MIN = 18
BUDGET = int(os.environ.get("ONEGAP_COHERE_BUDGET", "900"))  # keep ~100 calls of headroom under 1,000
DEFAULT_MODEL = "command-a-03-2025"

_last_calls: list[float] = []


class BudgetExceeded(RuntimeError):
    pass


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def calls_this_month() -> int:
    if not LEDGER.exists():
        return 0
    month = dt.date.today().strftime("%Y-%m")
    n = 0
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("ts", "").startswith(month):
            n += 1
    return n


def _throttle() -> None:
    now = time.time()
    while _last_calls and now - _last_calls[0] > 60:
        _last_calls.pop(0)
    if len(_last_calls) >= RATE_PER_MIN:
        time.sleep(60 - (now - _last_calls[0]) + 0.5)
    _last_calls.append(time.time())


def chat_json(messages: list[dict], schema: dict | None = None, *, purpose: str, model: str = DEFAULT_MODEL,
              temperature: float = 0.0, max_retries: int = 2) -> dict | list:
    """Return parsed JSON from a Cohere chat call (cached). `purpose` is recorded in the ledger."""
    payload = {"model": model, "messages": messages, "schema": schema, "temperature": temperature}
    key = _key(payload)
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["parsed"]

    used = calls_this_month()
    if used >= BUDGET:
        raise BudgetExceeded(f"Cohere budget reached: {used}/{BUDGET} calls this month")

    import cohere

    co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    fmt = {"type": "json_object", "schema": schema} if schema else {"type": "json_object"}
    err: Exception | None = None
    for attempt in range(max_retries):
        _throttle()
        err = None
        try:
            r = co.chat(model=model, messages=messages, temperature=temperature, response_format=fmt)
            text = r.message.content[0].text
            parsed = json.loads(text)
            break
        except json.JSONDecodeError as e:  # malformed JSON: retry (each retry is a billed call)
            err = e
        except Exception as e:  # rate limit / transient
            err = e
            time.sleep(10 * (attempt + 1))
        finally:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "purpose": purpose,
                                    "model": model, "key": key[:16], "attempt": attempt,
                                    "ok": err is None,
                                    "err": type(err).__name__ if err else None}) + "\n")
    else:
        raise RuntimeError(f"Cohere call failed after {max_retries} attempts: {err}")

    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "parsed": parsed}, ensure_ascii=False), encoding="utf-8")
    return parsed
