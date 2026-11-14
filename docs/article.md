# OneGap: Tiny Aya Global says no in English. In Hindi, Hinglish and Bengali it rarely does.

**One-line claim:** on held-out multi-turn attacks, Tiny Aya Global refuses a harmful request 14% of the time in
English and 1 to 3% in Hindi, Hinglish and Bengali; its refusal circuit is weakest where refusals are rarest; and
three pre-registered LoRA-DPO repairs, including one aimed at that circuit, did not yet move it.

*Lexsi AI Safety and Alignment Hackathon 2026. Final version, 14 Nov 2026.*

> **Content warning.** This write-up discusses harmful request categories (fraud and scams, cyber-attack help,
> weapons and violence, dangerous drugs, hate and harassment). It quotes no harmful prompt or reply.

**Tracks and problem statements covered** (numbered as listed on the submission form)

- Track 2a, AI Safety and Red Teaming
  - 3. Multi-turn audits
  - 4. Multilingual and code-mixed safety
- Track 2b, AI Safety and Correction
  - 5. Locating where safety lives
  - 6. Circuit-level interventions

> **For judges: one minute**
>
> - **Headline before → after.** Attack success, dev, pooled en + hi + Hinglish: base **28.3%** → every-layer
>   LoRA **25.8%**, circuit LoRA **30.0%**, random-block LoRA **25.8%**; over-refusal 4.4% → 4.0%. No change is
>   significant. Held-out finding: refused outright **14.0%** in English against **3.4 / 1.2 / 2.2%** in Hindi,
>   Hinglish and Bengali (paired gaps of 11 to 12 points, p ≤ 0.007).
> - **Base model + method in one line.** `CohereLabs/tiny-aya-global` (bf16, vLLM, greedy) audited turn by turn
>   in four languages and judged by Llama Guard 3 8B; refusal located with CircuitKIT and checked with SafeTune;
>   repaired with LoRA-DPO on every layer (D), the circuit's blocks (E) or random blocks (R).
> - **Story.** (1) English has a safety line and the other languages barely do, on held-out attacks too.
>   (2) Bengali, never trained on, gives unsafe replies 3.6 times as often as English. (3) In English the attack
>   is the build-up over turns; in Hindi and Hinglish one prompt is enough. (4) The refusal circuit is faithful
>   0.69 in English and 0.36 in Hindi. (5) The repair experiment is built and pre-registered; the first recipe was
>   too weak to move the model, and the $200 run answers the circuit question.
>
> Links: [model collection](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd) · [dataset collection](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a) · [code](https://github.com/Maadhi21/onegap) · [replay Space](https://huggingface.co/spaces/maadhi/onegap-replay) · [reproduction notebook](https://github.com/Maadhi21/onegap/blob/main/notebooks/reproduce_headline.ipynb)

## Abstract

A helpline assistant answers in the user's language. Someone after a scam script builds the request over
harmless turns, typed in Hindi or Hinglish, the romanized mix used on WhatsApp. We asked whether Tiny Aya Global
holds its safety line in those languages, where that line lives in the model, and whether a small targeted
repair can move it.

**The model refuses in English and almost never in the other languages, on held-out attacks too.** On a frozen
held-out set (365 requests from SafeMTData `Attack_600`, an attack family never used in training), it refuses a
harmful request outright 14% of the time in English and only 1 to 3% in Hindi, Hinglish and Bengali (paired
gaps of 11 to 12 points, p ≤ 0.007). Bengali, never trained on, gives unsafe replies 3.6 times as often as
English (10.8% against 3.0%, p = 0.04).

**English needs a conversation, the other languages need one prompt.** On the development split, a harmful
request asked outright in English succeeds 15% of the time and 30% when built up over turns. In Hindi and
Hinglish one prompt already succeeds 30 to 38% of the time, and the build-up adds nothing. Most failures happen
at turn 3 or 4.

**Refusal lives in a weak, language-dependent circuit.** A CircuitKIT refusal circuit reaches patching
faithfulness 0.69 in English, 0.67 in Hinglish and 0.36 in Hindi, and SafeTune's independent ranking agrees with
it no more than chance.

**A pre-registered test of three repairs, and an honest null.** Three LoRA adapters trained with DPO on 1,191
pairs from the model's own failures, under identical settings: every layer (D), only the circuit's blocks (E,
0.23% of the weights), as many random blocks (R). None moved attack success (25.8%, 30.0%, 25.8% against 28.3%)
or over-refusal (about 4%). The logs show why: the learning rate was too low to move the model, so the circuit
question is still open.

Everything ran on a $20 compute credit. Every number here can be recomputed from the published runs.

## 1. Problem statement and approach

### Why this problem: the real use case

A public helpline or service assistant in South Asia answers in the language the user writes in. Someone who
wants a UPI scam script or attack instructions does not ask outright. They build the request over a few
harmless-looking turns, and they type in Hindi, in Bengali, or in Hinglish. Small multilingual models like Tiny
Aya are exactly what gets deployed in that setting, and their safety is usually checked in English, one prompt
at a time.

### Threat model

| | |
|---|---|
| **Attacker** | Black-box chat access only; no access to weights or the system prompt. Patient: up to 6 user turns. Writes in English, Hindi, Hinglish or Bengali. Replays known attack conversations (non-adaptive). |
| **Goal** | An actionable harmful reply in one of five risk areas: fraud and scams, cyber-attack help, weapons and violence, dangerous drugs, hate and harassment. |
| **Defender** | Deploys Tiny Aya Global with a system prompt; can change the prompt, add a guard model, or ship a LoRA adapter. |
| **Success** | Any assistant turn judged unsafe (Llama Guard 3 8B). |

### Research questions and the pre-registered hypothesis

- **Q1.** Does building a request over turns raise attack success over asking it once, in each language?
- **Q2.** Is the model less safe in Hindi, Hinglish and Bengali than in English?
- **Q3.** Where does refusal live, and is it the same mechanism across languages?
- **H (pre-registered, committed before any adapter result).** A LoRA trained only on the refusal circuit's
  blocks (E) beats the same LoRA on as many random blocks (R) by at least 5 points of attack success, with a 90%
  interval excluding zero and over-refusal within 3 points; and E matches the all-layer LoRA (D) within 5 points.

### Approach

1. **Audit, turn by turn.** Replay attack conversations through the model in English, Hindi and Hinglish,
   generating every assistant turn live, and judge every turn. Send the final request alone as a control (S0).
   Bengali is evaluated and never trained on.
2. **Locate.** Find the components that carry refusal per language with CircuitKIT; measure faithfulness; check
   the ranking with SafeTune.
3. **Repair three ways, same data and budget:** D, E, R. R is the control that tells whether location matters or
   only the size of the edit.
4. **Decide by the rule above**, with every safety number next to an over-refusal number, and training and test
   attacks drawn from different attack families.

### Related work

- **Multi-turn attacks.** Crescendo (Russinovich, Salem and Eldan, 2024) and ActorAttack (Ren et al., 2024, the
  source of SafeMTData) show that gradual build-up breaks models that refuse single prompts; X-Teaming (Rahman et
  al., 2025) released the XGuard-Train conversations we train on.
- **Multilingual safety.** Low-resource languages jailbreak GPT-4 (Yong, Menghini and Bach, 2023) and
  multilingual jailbreak challenges (Deng et al., 2024) show safety degrading outside English. We add code-mixed
  Hinglish, multi-turn replay and a small open model.
- **Over-refusal and judging.** XSTest (Röttger et al., 2024); Llama Guard (Inan et al., 2023).
- **Where refusal lives.** Refusal is mediated by a single direction (Arditi et al., 2024); EAP-IG for faithful
  circuit discovery (Hanna, Pezzelle and Belinkov, 2024).
- **Repair.** DPO (Rafailov et al., 2023) with LoRA (Hu et al., 2021).

## 2. Hackathon repos (kits) used

| Repo | Version | How it was used | Features |
|---|---|---|---|
| **CircuitKIT** | 0.2.2 | Locating refusal per language and choosing the components repair E trains | `Pipeline.from_custom_data` with our contrastive refusal task; EAP-IG node-level discovery; `evaluate` (patching, ablation, random-circuit baseline); node scores for the component ranking and the cross-language overlap |
| **SafeTune** | 0.1.14 | An independent second opinion on the circuit | `interpret.safety_circuit_info` (weight method) on the same pairs; `extract_refusal_direction` and `RefusalDirectionModel` verified on Tiny Aya |

**How the repos connect.**

```
attack + harmless conversations ──► multi-turn runner (vLLM) ──► Llama Guard 3 8B ──► audit tables
              │
   failures ──┴─► correction pairs ──► LoRA-DPO ── D (all layers)
                                           ├──── E (circuit blocks) ◄── CircuitKIT ranking ◄── SafeTune cross-check
                                           └──── R (random blocks, same count as E)
                                                           │
                                     same runner, judge and statistics ──► E vs R vs D vs base
```

CuratorKIT, AlignTune, AgentTune and AuditKIT were not used for the results.

## 3. Models and licences

| Model | Role | Config | Licence |
|---|---|---|---|
| **`CohereLabs/tiny-aya-global`** (3.35B), revision `af89d219` | Base model under audit (the target model attacked) and base of every adapter | bf16, no quantization; vLLM; greedy, `max_new_tokens` 384, context 8,192 | CC-BY-NC 4.0 |
| `meta-llama/Llama-Guard-3-8B` | Judge of every assistant turn | bf16; scores the English view of each reply | Llama 3.1 Community Licence |
| `facebook/nllb-200-distilled-1.3B` | Hindi and Bengali user turns; English view of Hindi and Bengali replies | — | CC-BY-NC 4.0 |
| `sentence-transformers/LaBSE` | Translation fidelity gate | — | Apache-2.0 |
| Cohere Command A (API) | Hinglish user turns; chosen refusals for the correction pairs | — | Cohere API terms |

No closed-model baseline.

## 4. Datasets, benchmarks and licences

### Sources and languages

| Source | Used for | Licence | What we publish |
|---|---|---|---|
| `marslabucla/XGuard-Train` (X-Teaming) | Training and dev attack conversations | Gated, no licence stated | IDs and labels only |
| `SafeMTData/SafeMTData`, `Attack_600` (ActorAttack) | **Held-out test attacks, a different attack family** | MIT | Test conversations in four languages with labels |
| `Paul/XSTest` (safe prompts) | Over-refusal | CC-BY-4.0 | Translated prompts with attribution |
| Harmless multi-turn conversations | Over-refusal under multi-turn pressure | Ours | Full text |

Languages: English, Hindi (Devanagari), Hinglish (romanized Hindi-English, code-mixed), Bengali (evaluation only,
never trained on). Hindi and Bengali are translated with NLLB; Hinglish is written by Command A. Every turn must
back-translate to English with LaBSE cosine ≥ 0.80 and be in the right script, or the conversation is dropped.

### Attack prompt set (Track 2a)

| Attack type | Source | Split | Size en / hi / Hinglish / bn |
|---|---|---|---|
| Multi-turn, gradual build-up (human-like, up to 6 turns) | XGuard-Train | train / dev | 391 / 390 / 391 / — and 40 / 40 / 40 / 39 |
| Multi-turn, actor-network attack (a different attack family) | `Attack_600` | test | 100 / 89 / 83 / 93 |
| Single-prompt control (the final request alone, S0) | both | dev and test | same sizes |
| Language and code-mixing (translation, romanized mix) | applied to all of the above | all | — |

### Over-refusal test set (Track 2b)

| Set | Dev en / hi / Hinglish / bn | Test en / hi / Hinglish / bn |
|---|---|---|
| XSTest-style safe prompts that look risky | 50 / 48 / 48 / 46 | 150 / 147 / 148 / 146 |
| Harmless multi-turn conversations in the same five risk areas | 30 / 28 / 22 / 25 | 80 / 76 / 50 / 63 |

### Train / test splits and leak check

- **Splits are by behaviour:** no behaviour appears in two splits (checked by `onegap check-splits`), and train
  and test come from different attack families.
- **Text leak check (14 Nov):** exact match after normalisation and 5-gram Jaccard ≥ 0.5 between every English
  training user turn and every dev and test user turn. **No attack turn overlaps.** 12 of 1,024 English test user
  turns, all in harmless conversations, reuse phrasing from harmless training conversations (they share a
  template); they affect only the over-refusal set.
- **Test freeze:** before the freeze, a consistency check flagged nine translation errors (for example
  "psychopharmacology" rendered as "psychotherapy"); each was checked against the English and fixed. The frozen
  file is tagged `test-freeze-v1`, and its SHA-256 is verified inside every test job.

### Safety benchmarks

The dev split and the frozen test set above are the safety benchmarks; English, Hindi and Hinglish are the focus
task and Bengali is a held-out-language check. No general-capability benchmark was run (see Limitations).

## 5. Metrics, and why these

| Metric | Better | Why |
|---|---|---|
| **Attack success rate (ASR):** share of attack conversations with an unsafe reply at any turn | ↓ | A multi-turn attack succeeds the moment any reply is unsafe |
| **Refused outright:** share of attack conversations where the model refuses | ↑ | The behaviour a safe model shows; separates "refuses" from "answers vaguely" |
| **Cost-to-break:** user turns before the first unsafe reply | ↑ | How much patience an attacker needs |
| **Multi-turn uplift:** ASR of the full conversation minus ASR of the final request alone | ↓ | What a single-prompt test would miss |
| **Over-refusal:** share of harmless items refused (XSTest-style prompts and harmless conversations) | ↓ | So a repair cannot win by refusing everything; reported next to every safety number |
| **Circuit faithfulness:** patching recovery, ablation effect, z-score against random circuits | ↑ | Whether the located circuit really carries refusal |

**Labels.** Each assistant turn gets one of three labels: *unsafe* (Llama Guard 3 8B on the English view of the
reply), *refused* (a multilingual refusal-phrase detector, when not unsafe), or *neither*. Statistics: Wilson 95%
intervals, paired bootstrap over behaviours (10,000 resamples, seed 13), McNemar exact tests.

**Checking the judge.** No human agreement study yet (see Limitations). Two checks we did run: (1) in 36 sampled
Hindi, Hinglish and Bengali replies labelled *neither*, none was a missed refusal; they were answers, so the
refusal gap in 7.1 is not an artefact of the phrase detector. (2) Two runs of one condition differ on about 4% of
turn labels (batched greedy decoding is not bit-reproducible); every interval below absorbs this.

## 6. Method and experiments

### Inference and judging

vLLM 0.33, bf16, greedy decoding, `max_new_tokens` 384, context 8,192, system prompt A (two sentences: a helpful
public-helpline assistant that answers in the user's language). Every assistant turn is generated live; user turns
are replayed. Hindi and Bengali replies are translated to English by NLLB and Hinglish replies by the base model
before Llama Guard 3 8B scores them.

### Correction pairs

1,172 train attack conversations and 1,028 harmless ones were replayed through the base model and judged. At the
first unsafe turn: prompt = the history with the model's own replies; rejected = its unsafe reply; chosen = a
refusal in the same language and script that names the concern and points to something legitimate (Command A,
length-matched). Harmless pairs teach the opposite (chosen = the model's helpful reply, rejected = an unnecessary
refusal). 1,191 pairs after the gates and the length limit.

### Training (D, E, R share everything except the target modules)

| Setting | Value |
|---|---|
| Method | DPO (TRL), beta 0.1 |
| LoRA | r 16, alpha 32, dropout 0.05 |
| Learning rate, epochs, effective batch | 5e-6, 2 epochs (150 steps), 16 |
| Max length / prompt length | 3,072 / 2,688 tokens |
| Precision, seed | bf16, seed 13 |
| D | every linear projection in all 36 layers (30.2M trainable, 0.90%) |
| E | the 9 attention and 9 MLP blocks the circuit ranks highest (7.56M, 0.23%) |
| R | 9 + 9 random blocks, disjoint from E (7.56M, 0.23%) |

### CircuitKIT settings

Contrastive task per language (English, Hindi, Hinglish): 48 pairs of a harmful request the base model failed on
and a harmless request of the same category, two history turns, lengths equalised. Metric: logit difference
between the first token of a refusal and the model's own first token. Discovery: EAP-IG at node level, sparsity
0.3 (0.1 and 0.2 also run). Faithfulness: patching, ablation and 5 random circuits; pre-set bar 0.75 patching
recovery. Selection: top 36 nodes, at most 9 blocks per kind. Overlap: Jaccard of the top 36 nodes. Other
settings at CircuitKIT 0.2.2 defaults.

### SafeTune settings

`safety_circuit_info` with the weight method on the same harmful and harmless requests; top 9 MLP layers compared
with E's 9; `extract_refusal_direction` checked on Tiny Aya. Other settings at SafeTune 0.1.14 defaults.

### Seeds and ablations

One seed (13) per adapter. The three arms are the ablation: E against R isolates the location of the edit, E
against D the size.

## 7. Results

### 7.1 Held-out: the language gap (Q2)

365 harmful requests from the frozen `Attack_600` set, each sent alone, in four languages. Paired by behaviour;
% [Wilson 95%] (count/n); McNemar exact p.

| | English | Hindi | Hinglish | Bengali (never trained) |
|---|---|---|---|---|
| **Refused outright** ↑ | **14.0** [8.5, 22.1] (14/100) | 3.4 [1.2, 9.4] (3/89) | 1.2 [0.2, 6.5] (1/83) | 2.2 [0.6, 7.5] (2/93) |
| Refused, gap to English | — | −12.4 [−20.2, −4.5], p = 0.007 | −10.8 [−18.1, −4.8], p = 0.004 | −11.8 [−18.3, −5.4], p = 0.001 |
| **Unsafe reply** ↓ | 3.0 [1.0, 8.5] (3/100) | 6.7 [3.1, 13.9] (6/89) | 7.2 [3.4, 14.9] (6/83) | **10.8** [5.9, 18.7] (10/93) |
| Unsafe, gap to English | — | +3.4 [−1.1, +9.0], p = 0.38 | +3.6 [−2.4, +9.6], p = 0.45 | +7.5 [+2.2, +14.0], p = 0.039 |

- **English has a safety line; the other languages barely do.** All three refusal gaps exclude zero, on an attack
  family the repair data never saw.
- **Bengali, never trained on, fails most:** the one unsafe-rate gap that excludes zero.
- **By risk area,** weapons (9 of 100 conversations across the four languages) and cyber-attack help (9 of 93)
  gave the most unsafe single replies; fraud (2 of 64) and drugs (2 of 30) the fewest.
- Asked alone, these requests rarely succeed: `Attack_600` final turns lean on the conversation before them.

### 7.2 Dev: the multi-turn gap and cost-to-break (Q1)

| | English | Hindi | Hinglish | Bengali |
|---|---|---|---|---|
| **S0**, final request alone (ASR ↓) | 15.0 [7.1, 29.1] (6/40) | 37.5 [24.2, 53.0] (15/40) | 30.0 [18.1, 45.4] (12/40) | 28.2 [16.5, 43.8] (11/39) |
| **A**, full conversation (ASR ↓) | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 38.5 [24.9, 54.1] (15/39) |
| S0 refused outright ↑ | 25.0 (10/40) | 2.5 (1/40) | 7.5 (3/40) | 2.6 (1/39) |

- **In English, the build-up is the attack:** 15% asked outright, 30% over a conversation (9 conversations fail
  only with the build-up, 3 only without).
- **In Hindi and Hinglish, one prompt is enough:** 30 to 38% asked outright; the build-up adds nothing.
- **Cost-to-break:** of the 34 failed conversations (en + hi + Hinglish), the first unsafe reply came at user turn
  1 once, turn 2 five times, **turn 3 sixteen times**, turn 4 eight times, turns 5 to 6 four times: median 3 user
  turns, mean 3.3.
- The dev refusal gap (25% in English against 2.5 to 7.5%) is the one that replicated on held-out attacks in 7.1.

### 7.3 Where refusal lives (Q3)

| | English | Hindi | Hinglish |
|---|---|---|---|
| Patching faithfulness ↑ (bar: 0.75) | 0.69 | 0.36 | 0.67 |
| Ablation effect ↑ | 1.0 | 0.0 | 1.0 |
| Circuit against 5 random circuits (z) ↑ | 3.3 | 1.3 | 1.3 |
| SafeTune's top-9 MLP layers inside E's 9 (chance 2.25) | 1 | 1 | 2 |

- **Refusal localises best in English and worst in Hindi**, where the circuit explains a third of the behaviour
  and is no better than random. That matches the audit: there is little refusal mechanism to find where the model
  rarely refuses.
- **About a third of the top components are shared between any two languages** (Jaccard 0.33 to 0.41).
- **SafeTune and CircuitKIT disagree** beyond chance, so the circuit is a hypothesis to test, which is why E runs
  against a random-block control.

### 7.4 Same model before vs after: the repairs (H)

| Dev, en + hi + Hinglish | ASR ↓ | Change against base, points [95% CI] | Over-refusal ↓ | Refused share of attack turns |
|---|---|---|---|---|
| **A, base model (before)** | 28.3 (34/120) | — | 4.4 (10/226) | 2.3 |
| **B, stronger system prompt** | 27.5 (33/120) | −0.8 [−9.2, +7.5] | 4.9 (11/226) | 1.9 |
| **D, every layer** | 25.8 (31/120) | −2.5 [−8.3, +2.5] | 4.0 (9/226) | 4.4 |
| **E, circuit blocks** | 30.0 (36/120) | +1.7 [−5.0, +8.3] | 4.0 (9/226) | 2.9 |
| **R, random blocks** | 25.8 (31/120) | −2.5 [−10.8, +5.0] | 4.0 (9/226) | 3.1 |

**What the rule says:** E does not beat R (R − E = −4.2 points, 90% interval −10.8 to +2.5); E "matches" D. But
no arm beats the base model, so we do not read this as evidence for or against the circuit.

### 7.5 What didn't work

- **The repair recipe.** Mean DPO training loss stayed at 0.52 to 0.65 against 0.69 at the start, and the share
  of attack turns refused rose only from 2.3% to at most 4.4%. A learning rate of 5e-6 suits full fine-tuning, not
  a rank-16 LoRA for 150 steps; we took it from our plan without a quick check run.
- **A stronger system prompt (B).** It left ASR unchanged and moved over-refusal in opposite directions: from
  8.8% to 0% in English, up to 5.6 to 9.2% in Hindi, Hinglish and Bengali (each change excludes zero). A prompt
  written in English does not behave the same way in Hindi.
- **The rule had a hole:** it never asked whether a repair beats the base model first. The next version adds that
  clause before any comparison between arms.
- **The held-out multi-turn run** (A on the test set) was cut off before it finished when the credit ran out.

### 7.6 Compute, cost and ROI

| | |
|---|---|
| Compute | Hugging Face Jobs only: NVIDIA L4 ($0.80/h) for evaluation, L40S ($1.80/h) for training and circuits, CPU for publishing. No local GPU. |
| Total | About $18 by our ledger; the $20 hackathon credit is used up. Every job's flavour, minutes and cost: [`docs/RUNS.md`](https://github.com/Maadhi21/onegap/blob/main/docs/RUNS.md) |
| One evaluation of one condition | Dev, 456 conversations: about 34 minutes on an L4, about $0.45 |
| Cost per 1,000 replies, our model | About $0.41 on an L4, upper bound including model load and Llama Guard judging (365 replies in 11.3 minutes) |
| Edit size | E and R train 0.23% of the weights, D 0.90% |
| ROI | Not established: no repair works yet |

## 8. Limitations

- **No repair works yet.** H is untested in practice because no arm moved the model.
- **The judge is not audited by humans.** No agreement study (% agreement or Cohen's κ) yet; the refusal label is
  a phrase detector, spot-checked only.
- **Small n and one seed.** 40 conversations per language on dev, 83 to 100 on test; one seed per adapter.
- **Only the single-prompt audit ran on the held-out family**; A, B, D, E and R have dev results only.
- **No general-capability check** after the adapters, and **no quantized (GGUF, 4-bit) results**.
- **Machine translations**, reviewed by an AI assistant, not by a native speaker.
- **Replayed, non-adaptive attacks.**

## 9. Dual-use, ethics, responsible disclosure and security

- **No new attack technique.** All attack conversations come from public datasets (XGuard-Train, `Attack_600`);
  our contribution is translating them and measuring the gap.
- **No harmful text is published.** Replies judged unsafe are replaced by a marker with the SHA-256 of the
  original and its labels; replies in attack conversations are withheld until a human audit of the judge;
  XGuard-Train text is never published. A test (`tests/test_redaction.py`) must pass before any upload.
- **Responsible disclosure.** The weakness is a property of an open model under public attacks, not an exploit
  of a deployed system. We plan to share the language-gap results with Cohere Labs, the model's developer; we have
  not contacted them yet.
- **Security considerations.** Anyone deploying a small multilingual model should test safety in every language
  they serve, multi-turn, and should not rely on a system prompt written and checked in English.

## 10. With more credits

### Planned approach ($200)

| Run | Cost | What it settles |
|---|---|---|
| The full-conversation audit (A) on the frozen held-out set, four languages | about $1.5 | Whether the multi-turn gap holds on a new attack family |
| A learning-rate sweep on D, then D, E and R retrained at the chosen rate | about $10 | A repair that moves the model, shared by all three arms |
| D, E, R, B and the base model on the held-out set | about $8 | **The circuit question (H):** E beats R by ≥ 5 points with over-refusal within 3 points, and matches D at a quarter of the parameters |
| Two more seeds for D, E and R | about $25 | Seed-to-seed spread next to every difference |
| A guard model (Llama Guard 3 1B on every user turn) | about $2 | Whether an adapter is worth it against a cheap guard |
| 4-bit (GGUF) versions of the best adapter, and a general-capability check | about $5 | That the fix survives deployment and does not cost capability |
| A human audit of 300 judged turns, two annotators | — | Agreement between humans and the judge in each language |

### Expected before → after (forecasts, not results)

| Metric (held-out, en + hi + Hinglish) | Before (measured) | Expected after |
|---|---|---|
| ASR, full conversation | about 28% (dev) | about 15% for the best adapter |
| Refused outright, Hindi and Hinglish | 1 to 3% | above 10%, closing most of the gap to English |
| Over-refusal | about 4% | at most 7% |
| E against R | no difference (recipe too weak) | E ahead by ≥ 5 points if the circuit is the right place to train; a measured null otherwise |

## 11. Artifacts

| | Link |
|---|---|
| HF models collection | [maadhi/onegap-models](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd): [E circuit LoRA](https://huggingface.co/maadhi/onegap-tiny-aya-global-circuit-lora), [D all-layer LoRA](https://huggingface.co/maadhi/onegap-tiny-aya-global-plain-lora), [R random-block LoRA](https://huggingface.co/maadhi/onegap-tiny-aya-global-random-lora) |
| HF datasets collection | [maadhi/onegap-data](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a): [transcripts, labels, logs and reports for every run](https://huggingface.co/datasets/maadhi/onegap-transcripts-and-logs), [circuits and the SafeTune cross-check](https://huggingface.co/datasets/maadhi/onegap-circuits) |
| GitHub repo | [Maadhi21/onegap](https://github.com/Maadhi21/onegap) |
| Demo / Space | [maadhi/onegap-replay](https://huggingface.co/spaces/maadhi/onegap-replay): pick a split, language and conversation and replay it turn by turn with its safety labels (redacted as in the dataset) |
| Reproduction notebook | [`notebooks/reproduce_headline.ipynb`](https://github.com/Maadhi21/onegap/blob/main/notebooks/reproduce_headline.ipynb): rebuilds the dev and held-out reports from the published runs on CPU, checks them against the published reports and redraws the figures |
| Video | None |

Everything is public and un-gated; adapters are CC-BY-NC 4.0 (inherited from Tiny Aya), code Apache-2.0.

## 12. Documentation and reproduction

**Environment.** Python 3.12 with `uv`; a laptop is enough to recompute every table. GPU work runs as Hugging Face
Jobs (L4 for evaluation, L40S for training); it needs an `HF_TOKEN` with access to the gated Tiny Aya and Llama
Guard models.

**Repo layout.** `configs/` (data, decoding, judges, circuit and training settings), `prompts/` (system prompts
A and B, judge rubric), `src/onegap/` (data building, multi-turn runner, judges, metrics, statistics, circuits,
training, report), `scripts/` (HF Jobs launcher and helpers), `tests/`, `docs/` (run log, progress, cards).

**Run.**

```bash
uv sync --extra local --extra dev && uv run pytest
uv run onegap report <run folders from the transcripts dataset> --name check     # every table, no GPU
uv run --extra local python scripts/launch_job.py run --flavor l4x1 --venv vllm \
    -- --condition A --split test --limit 4                                     # one evaluation, smoke first
```

Or open [`notebooks/reproduce_headline.ipynb`](https://github.com/Maadhi21/onegap/blob/main/notebooks/reproduce_headline.ipynb), which rebuilds the headline tables from the published runs on CPU. Full steps, from data building to training: [README](https://github.com/Maadhi21/onegap#reproduce).

**Declaration of work done outside the hackathon.** None. All code, data, runs and writing were done during the
hackathon, 1 to 14 November 2026.

## 13. Team, acknowledgements, LLM usage

**Team:** Madhava ([@maadhi](https://huggingface.co/maadhi)), team lead: design, decisions and runs. *Teammates:
add Hugging Face usernames.*

**Acknowledgements.** Lexsi Labs, Cohere Labs and Hugging Face for the hackathon, the Tiny Aya models, the kits
and the compute credit.

**LLM usage statement.** The code, the experiment pipeline and most of this write-up were produced with an AI
coding assistant (Claude) working under the team's direction: roughly 80 to 90% of the code and most of the text.
The team set the question, the decision rule and the budget, and reviewed the results; every number comes from
the published runs.

**Changelog.** v1, 12 Nov: dev results. 13 Nov: test set frozen, held-out single-prompt audit, story rewrite, reproduction notebook and replay Space.
14 Nov: leak check, metric directions, cost-to-break, threat model, settings per kit, ethics and reproduction
sections.

## Appendix

- System prompts and judge rubric: [`prompts/`](https://github.com/Maadhi21/onegap/tree/main/prompts)
- All configs: [`configs/`](https://github.com/Maadhi21/onegap/tree/main/configs)
- Every run, including failed ones, with cost: [`docs/RUNS.md`](https://github.com/Maadhi21/onegap/blob/main/docs/RUNS.md)
- Full reports: `report/dev_v1.md` and `report/test_v1.md` in the [transcripts dataset](https://huggingface.co/datasets/maadhi/onegap-transcripts-and-logs/tree/main/report)

**References.** Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*. Deng et al.
2024, *Multilingual Jailbreak Challenges in Large Language Models*. Hanna, Pezzelle and Belinkov 2024, *Have Faith
in Faithfulness*. Hu et al. 2021, *LoRA*. Inan et al. 2023, *Llama Guard*. Rafailov et al. 2023, *Direct
Preference Optimization*. Rahman et al. 2025, *X-Teaming*. Ren et al. 2024, *Derail Yourself: Multi-turn LLM
Jailbreak Attack through Self-discovered Clues* (ActorAttack, SafeMTData). Röttger et al. 2024, *XSTest*.
Russinovich, Salem and Eldan 2024, *The Crescendo Multi-Turn LLM Jailbreak Attack*. Yong, Menghini and Bach 2023,
*Low-Resource Languages Jailbreak GPT-4*.
