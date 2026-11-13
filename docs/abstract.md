# OneGap: where Tiny Aya Global's safety gives way, turn by turn, in English, Hindi and Hinglish

**Track 2:** multi-turn audits; multilingual and code-mixed safety; locating where safety lives; circuit-level interventions.

A public helpline assistant answers in the user's language. Someone after a scam script does not ask outright: they build the request over a few harmless-looking turns, in Hindi or in Hinglish, the romanized mix people type on WhatsApp. We audited `CohereLabs/tiny-aya-global` for this, tried three repairs, and report what held and what did not. This first version uses the development split only (40 behaviours per language).

**What we found**

- **The gap depends on the language.** In English, a harmful request asked outright gets an unsafe reply 15% of the time (6/40) and a refusal 25% of the time. Built up over a conversation, it succeeds 30% of the time (12/40). In Hindi and Hinglish a single prompt already works 37.5% and 30% of the time, and the build-up adds nothing. A second run on another inference stack agreed.
- **Conversations break in the middle.** 24 of 34 failed conversations first fail at turn 3 or 4.
- **Nothing we tried moved it yet.** A stronger system prompt left attack success at 27.5%. We also built 1,191 preference pairs from the model's own failures and trained three LoRA adapters with DPO under identical settings: every layer (D), only the blocks a CircuitKIT refusal circuit points to (E), and as many random blocks (R). Attack success stayed at 25.8%, 30.0% and 25.8% against 28.3% for the base model. The training logs show why: the recipe barely moved the model.
- **The circuit is weak.** Patching faithfulness is 0.69 in English, 0.36 in Hindi and 0.67 in Hinglish, below the 0.75 we had set, and SafeTune's independent ranking overlaps it no more than chance.
- **The measuring stick wobbles.** Two runs of one condition disagreed on about 4% of turn labels: batched greedy decoding is not reproducible on this stack, so single runs on a few dozen prompts cannot resolve small differences between defences.

**How.** Attacks come from XGuard-Train (train, dev) and SafeMTData `Attack_600` (held-out test, a different attack family). Hindi and Bengali come from NLLB and Hinglish from a Cohere model, behind a back-translation gate; Bengali is never trained on. Every assistant turn is generated live and judged by Llama Guard 3 8B. A single-prompt control isolates what the build-up adds. We report Wilson intervals, a bootstrap over behaviours and McNemar tests. The rule for comparing D, E and R was committed before any result, and we say where it fell short.

**What is public.** Code with tests and the full run log, failures included; the three adapters; per-turn labels and manifests for every run; the circuits. Unsafe model text is never published: markers with hashes replace it.

**Limits and next.** Everything ran on a $20 compute credit. No human has audited the judge or the translations. Next: the held-out test run of the audit, then a revised repair recipe and a guard-model baseline if we are shortlisted.
