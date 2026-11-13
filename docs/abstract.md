# OneGap: where Tiny Aya Global's safety gives way, turn by turn, in English, Hindi and Hinglish

**Track 2:** multi-turn audits; multilingual and code-mixed safety; locating where safety lives; circuit-level interventions.

A helpline assistant answers in the user's language. Someone after a scam script builds the request over harmless-looking turns, in Hindi or in Hinglish, the romanized mix people type on WhatsApp. We audited `CohereLabs/tiny-aya-global` on a development split (40 behaviours per language) and a frozen held-out set from a different attack family (83 to 100 per language), tried three repairs, and report what held and what did not.

**What we found**

- **The gap depends on the language.** In English a harmful request asked outright gets an unsafe reply 15% of the time (6/40); built up over a conversation, 30% (12/40). In Hindi and Hinglish a single prompt already works 37.5% and 30% of the time, and the build-up adds nothing.
- **It holds on held-out attacks.** On 365 frozen `Attack_600` requests the model refuses outright 14% of the time in English and 1 to 3% in Hindi, Hinglish and Bengali (paired gaps of 11 to 12 points, p ≤ 0.007). Bengali, never trained on, gives unsafe replies 3.6 times as often as English (p = 0.04). Asked alone, these requests rarely succeed (3 to 11%); the multi-turn run on this set was cut off before it finished.
- **Conversations break in the middle.** 24 of 34 failed conversations first fail at turn 3 or 4.
- **Nothing we tried moved it yet (dev).** A stronger system prompt left attack success at 27.5%. Three LoRA adapters trained with DPO on 1,191 pairs built from the model's own failures, on every layer (D), only the blocks a CircuitKIT refusal circuit points to (E), or as many random blocks (R), left it at 25.8%, 30.0% and 25.8% against 28.3% for the base model. The recipe barely moved the model.
- **The circuit is weak.** Faithfulness is 0.69 (English), 0.36 (Hindi) and 0.67 (Hinglish), below our 0.75 bar; SafeTune's ranking overlaps it no more than chance.
- **The measuring stick wobbles.** Two runs of one condition disagreed on about 4% of turn labels, so single small runs cannot resolve small differences.

**How.** Attacks come from XGuard-Train (train, dev) and SafeMTData `Attack_600` (held-out test, a different attack family). Hindi and Bengali come from NLLB, Hinglish from a Cohere model, behind a back-translation gate; Bengali is never trained on. Every assistant turn is generated live and judged by Llama Guard 3 8B. We report Wilson intervals, paired bootstraps and McNemar tests; the D/E/R rule was committed before any result.

**What is public.** Code, tests and the full run log, failures included; the three adapters; per-turn labels and manifests for every run; the circuits; a replay Space and a CPU notebook that rebuilds every table. Unsafe text is hashed.

**Limits and next.** Everything ran on a $20 compute credit. No human has audited the judge or the translations. Next, if shortlisted: the multi-turn run and the defences on the held-out set, a revised repair recipe, and a guard-model baseline.
