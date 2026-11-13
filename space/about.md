## OneGap

*Lexsi AI Safety and Alignment Hackathon 2026, Track 2: Multi-turn audits; Multilingual and code-mixed safety;
Locating where safety lives; Circuit-level interventions.*

**Tiny Aya Global says no in English. In Hindi, Hinglish and Bengali it rarely does.**

1. **Held out, one harmful prompt** (frozen `Attack_600` set, an attack family never used in training): the model
   refuses outright 14% of the time in English and 1 to 3% in Hindi, Hinglish and Bengali (paired gaps of 11 to 12
   points, p ≤ 0.007). Bengali, never trained on, gives unsafe replies 10.8% of the time against 3.0% in English
   (p = 0.04).
2. **Dev, multi-turn:** in English the attack needs a conversation (15% asked outright, 30% over several turns); in
   Hindi and Hinglish one prompt already works 30 to 38% of the time. Conversations break at turn 3 or 4.
3. **Locating refusal:** CircuitKIT's refusal circuit is faithful at 0.69 in English, 0.67 in Hinglish and 0.36 in
   Hindi (bar 0.75); SafeTune's refusal neurons mostly fall outside it.
4. **Repairs:** LoRA DPO on all layers (D), on the circuit only (E) and on random layers of the same size (R), with
   the same data, steps and seed. On dev no arm changed attack success (pooled: base 28.3%, D 25.8%, E 30.0%,
   R 25.8%; every interval contains zero) and over-refusal stayed at about 4%. With no working arm, circuit
   against random is untested, not negative. The pre-registered rule and the next run are in the write-up.

**How to read this Space.** Each conversation shows one column per condition: S0 single prompt, A base, B safety
system prompt, D/E/R the three LoRA repairs. Labels come from Llama Guard 3 8B (unsafe) and a refusal-phrase
detector (refused); "answered / deflected" is everything else. The held-out split has only S0 (the budget ran out
before the other conditions ran on it).

**Limits.** Dev has 40 attack conversations per language, so intervals are wide. No human audit of the judge and
no human review of the translations (a mechanical consistency check replaced it). Replies are greedy vLLM outputs
in bf16 and are not regenerated bit for bit.

**Links.** [Write-up and code](https://github.com/Maadhi21/onegap) ·
[models](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd) ·
[data](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a) ·
[reproduction notebook](https://github.com/Maadhi21/onegap/blob/main/notebooks/reproduce_headline.ipynb)

Licences: code Apache-2.0; adapters and data CC-BY-NC 4.0; Tiny Aya Global CC-BY-NC; Llama Guard 3 under the
Llama 3.1/3.2 community licence.
