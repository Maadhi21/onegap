"""OneGap: multilingual multi-turn safety audit and circuit-restricted repair for Tiny Aya Global."""

__version__ = "0.1.0"

LANGUAGES = ("en", "hi", "hinglish", "bn")
TRAIN_LANGUAGES = ("en", "hi", "hinglish")
HELD_OUT_LANGUAGES = ("bn",)
CATEGORIES = ("fraud_scam", "cyber_attack", "weapons_violence", "dangerous_drugs", "hate_harassment")
CONDITIONS = ("A", "S0", "B", "G", "C", "D", "E", "R", "F")
OUTCOMES = ("refused", "unsafe", "deflected")
