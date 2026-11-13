import pytest

from onegap import train as T


def test_shared_settings_identical_across_D_E_R():  # Invariant 4
    fps = {c: T.shared_fingerprint(T.load_train_cfg(c)) for c in T.CONFIGS}
    assert len(set(fps.values())) == 1, fps


def test_only_targets_differ():
    cfgs = [T.load_train_cfg(c) for c in T.CONFIGS]
    assert [c["target"]["mode"] for c in cfgs] == ["all-linear", "circuit", "random"]
    for c in cfgs:
        assert set(c) == {"condition", "shared", "target"}


def test_target_modules_and_counts():
    comp = {"E": {"modules": ["model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.k_proj",
                              "model.layers.9.mlp.up_proj"]},
            "R": {"modules": ["model.layers.20.self_attn.q_proj"]}}
    d, e, r = (T.load_train_cfg(c) for c in T.CONFIGS)
    assert T.target_modules(d, None) == "all-linear"
    assert T.target_modules(e, comp) == comp["E"]["modules"]
    assert T.target_modules(r, comp) == comp["R"]["modules"]
    with pytest.raises(ValueError):
        T.target_modules(e, None)
    assert T.module_counts(comp["E"]["modules"]) == {"attn_blocks": 1, "mlp_blocks": 1, "projections": 3}


def test_missing_targets_uses_suffix_match_not_prefix():
    names = ["model.layers.11.self_attn.q_proj", "base_model.model.model.layers.2.mlp.up_proj"]
    assert T.missing_targets(names, ["model.layers.1.self_attn.q_proj"]) == ["model.layers.1.self_attn.q_proj"]
    assert T.missing_targets(names, ["model.layers.2.mlp.up_proj", "model.layers.11.self_attn.q_proj"]) == []


def test_split_kwargs_and_settings():
    s = T.dpo_settings(T.load_train_cfg(T.CONFIGS[0])["shared"], "out")
    assert s["gradient_accumulation_steps"] * s["per_device_train_batch_size"] == 16
    assert s["beta"] == 0.1 and s["learning_rate"] == 5e-6 and s["num_train_epochs"] == 2 and s["seed"] == 13
    kept, dropped = T.split_kwargs({"beta": 0.1, "max_prompt_length": 9}, {"beta"})
    assert kept == {"beta": 0.1} and dropped == {"max_prompt_length": 9}


def test_edit_size_table():
    from onegap.edit_size import table

    m = {"E": {"lora_modules": 7, "edit_size": {"params_changed": 1000, "changed_share": 0.0005, "adapter_params": 50,
                                                "layers_touched": [3, 9], "delta_fro": 0.123456}}}
    row = table(m)["E"]
    assert row == {"params_changed": 1000, "changed_pct": 0.05, "adapter_params": 50, "layers_touched": 2,
                   "delta_fro": 0.1235, "lora_modules": 7}


def test_steer_pick_strength():
    from onegap.steer import pick_strength

    sweep = [{"strength": 0, "attack_refusal": 0.1, "overref_refusal": 0.0},
             {"strength": 1, "attack_refusal": 0.2, "overref_refusal": 0.05},
             {"strength": 2, "attack_refusal": 0.5, "overref_refusal": 0.3}]
    assert pick_strength(sweep, 0.3)["strength"] == 2
    assert pick_strength(sweep, 0.9)["strength"] == 2
    assert pick_strength(sweep, None)["strength"] == 1


def test_keep_by_length_drops_long_prompts_and_long_totals():
    from onegap.train import keep_by_length

    assert keep_by_length(2000, 300, 500, 3072, 2688)
    assert not keep_by_length(2700, 10, 10, 3072, 2688)   # prompt over max_prompt_length
    assert not keep_by_length(2600, 300, 500, 3072, 2688)  # prompt + longer reply over max_length
