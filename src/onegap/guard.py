"""Condition G: Llama Guard 3 1B screens each user turn in context; flagged turns get a fixed refusal.

Invariant 6: the guard is the 1B model; the judge (judges.py) is the 8B model. They are never the same model.
"""

from __future__ import annotations

from onegap.judges import LlamaGuardJudge

GUARD_MODEL = "meta-llama/Llama-Guard-3-1B"


def as_parts(conversation: list[dict]) -> list[dict]:
    """Llama Guard 3 1B's chat template takes content as a list of typed parts, not a plain string."""
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in conversation]


class TurnGuard(LlamaGuardJudge):  # pragma: no cover - GPU only
    """Callable: conversations ending in the new user turn -> True where the guard says unsafe."""

    def __init__(self, model_name: str = GUARD_MODEL):
        if "8B" in model_name:
            raise ValueError("the guard must not be the judge model (Invariant 6)")
        super().__init__(model_name)

    def render(self, conv: list[dict]) -> str:
        try:
            return self.tok.apply_chat_template(as_parts(conv), tokenize=False)
        except Exception:  # template variant that takes plain strings
            return self.tok.apply_chat_template(conv, tokenize=False)

    def __call__(self, conversations: list[list[dict]]) -> list[bool]:
        return [v.unsafe for v in self.classify(conversations)]

    def close(self) -> None:
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
