import pytest


def make_tr(seed, labels, condition="A", language="en", kind="attack", category="fraud_scam"):
    return {
        "conv_id": f"{seed}-{language}", "seed_id": seed, "condition": condition, "language": language,
        "category": category, "kind": kind,
        "turns": [{"turn": i, "user": f"u{i}", "assistant": f"reply {i}", "label": lab}
                  for i, lab in enumerate(labels, start=1)],
    }


@pytest.fixture
def tr():
    return make_tr
