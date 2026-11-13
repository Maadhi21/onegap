# OneGap: a turn-by-turn safety audit of Tiny Aya Global in English, Hindi and Hinglish, and three repairs tested against each other

*Lexsi AI Safety and Alignment Hackathon 2026, Track 2. Final version, 13 Nov 2026.*

> **What is measured where.** The single-prompt audit of the base model (S0) ran on the frozen held-out test set:
> 83 to 100 behaviours per language from a different attack family than training (section 7.1). The
> full-conversation run on test (A) was cancelled by the platform 14 minutes before it finished, most likely
> because the $20 compute credit ran out, so the multi-turn numbers, the stronger prompt (B) and the adapters (D,
> E, R) are development-split results. Two baselines (guard model, steering) are not in the tables. We say so
> wherever it matters.

**In five lines**

1. **The gap depends on the language.** In English, Tiny Aya Global gives an unsafe reply to 15% of harmful
   requests asked outright (6/40) and to 30% when the same request is built up over a conversation (12/40). In
   Hindi and Hinglish a single prompt already works 30 to 38% of the time and the build-up adds nothing (dev).
   **On the frozen held-out set the language gap holds:** the model refuses a harmful request outright 14% of the
   time in English and 1 to 3% in Hindi, Hinglish and Bengali (paired gaps of 11 to 12 points, p ≤ 0.007), and
   Bengali, never trained on, gives unsafe replies more often than English (10.8% against 3.0%, p = 0.04).
2. **It fails in the middle of the conversation.** 24 of 34 failed conversations first fail at turn 3 or 4.
3. **Nothing we tried moved it yet, and we can show why.** A stronger system prompt left attack success at
   27.5%. Three LoRA adapters trained on the model's own failures (every layer, the circuit's blocks, random
   blocks) left it at 25.8%, 30.0% and 25.8%, against 28.3% for the base model (n = 120). The adapter recipe was
   too weak to move the model.
4. **The refusal circuit is weak.** Faithfulness 0.69 in English, 0.36 in Hindi, 0.67 in Hinglish, below the
   0.75 we set; a second tool does not agree with it beyond chance.
5. **The measuring stick wobbles.** Two runs of the same condition differed on about 4% of turn labels (3 of 68
   in our check), so small differences between defences cannot be read from single runs of 40 conversations.

Links: [model collection](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd) · [dataset collection](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a) · [code](https://github.com/Maadhi21/onegap)

---

## 1. Problem statement and approach

**Track 2, AI Red Teaming, Safety and Alignment.** Problem statements we answer: *Multi-turn audits* and
*Multilingual and code-mixed safety* (primary), *Locating where safety lives* and *Circuit-level interventions*
(supporting).

**The threat.** A public helpline assistant answers in the language the user writes in. Someone who wants a
UPI scam script or attack instructions does not ask outright. They build the request over several
harmless-looking turns, and they type in Hindi or in Hinglish, the romanized Hindi-English mix people use on
WhatsApp. The multi-turn problem statement asks for "attacks a single prompt would not catch", so the first
thing we built is the control that measures exactly that.

**The approach, in the order we ran it.**

1. **Audit.** Replay multi-turn attack conversations through `CohereLabs/tiny-aya-global` in English, Hindi and
   Hinglish, generating every assistant turn live, and judge every turn. Run the final request alone as a
   single-prompt control (S0). The difference between the two is what a single prompt would not catch. Bengali is
   evaluated but never trained on.
2. **Locate.** Find the components that carry refusal in each language with CircuitKIT, measure how faithful the
   circuit is, compare the languages, and cross-check with SafeTune.
3. **Repair, three ways, same data and budget.** Train a LoRA with DPO on correction pairs built from the model's
   own failures: on every layer (D), only on the blocks the circuit points to (E), and on the same number of
   randomly chosen blocks (R). R is the control that tells us whether the location matters or only the size.
4. **Decide by a rule written beforehand.** The rule for reading D, E and R was committed before any of the three
   had a result.

Two rules shaped everything: training attacks and test attacks come from different attack families, and every
safety number is reported next to an over-refusal number.

## 2. Hackathon repos used, what for, and which features

| Repo | Version | What we used it for | Features |
|---|---|---|---|
| **CircuitKIT** | 0.2.2 | Locating refusal per language and choosing the components for repair E | `Pipeline.from_custom_data` with our own contrastive refusal task (48 pairs per language); EAP-IG discovery at node level; `evaluate` with the patching, ablation and random-baseline pillars; node scores for the component ranking and the cross-language overlap |
| **SafeTune** | 0.1.14 | An independent second opinion on the circuit | `interpret.safety_circuit_info` on the same 144 pairs; `extract_refusal_direction` and `RefusalDirectionModel` verified on Tiny Aya for the steering baseline, which is not run yet |
| **AuditKIT** | 1.1.7 | Not used in version 1 | Installed and API-checked in a job (`Sample`, `evaluate` with precomputed outputs, `compare_models`, `ScoreGate`). The bridge that scores our transcripts with it is planned for the final version |
| **CuratorKIT** | 0.1.24 | Not used in version 1 | Installed and API-checked. A dedup and quality pass over the correction pairs is planned for the final version |

How they connect: CircuitKIT's node ranking decides which modules repair E may train; SafeTune checks that
ranking with a different method; the evaluation harness then tests whether the choice mattered. Two toolkits
carry real work in this version. We would rather say that than list four.

## 3. Models used and their licences

| Model | Role | Licence |
|---|---|---|
| `CohereLabs/tiny-aya-global` (3.35B, revision `af89d219`) | The model under audit and the base of every adapter | CC-BY-NC 4.0, gated |
| `meta-llama/Llama-Guard-3-8B` | Primary judge of every assistant turn | Llama 3.1 Community Licence, gated |
| `meta-llama/Llama-Guard-3-1B` | Guard model for condition G (not run yet). Never used as judge | Llama 3.2 Community Licence, gated |
| `facebook/nllb-200-distilled-1.3B` | Hindi and Bengali user turns; English view of Hindi and Bengali replies for the judge | CC-BY-NC 4.0 |
| `sentence-transformers/LaBSE` | Translation fidelity gate | Apache-2.0 |
| Cohere Command A (`command-a-03-2025`, API) | Hinglish rewriting and back-translation, category labels, benign conversations, teacher refusals for the correction pairs | Cohere API terms |

There is no safety-aligned Tiny Aya checkpoint from Lexsi to compare against (we looked; the aligned checkpoints
cover other base models), so that planned baseline was dropped.

## 4. Datasets and benchmarks used and their licences

| Source | Used for | Licence | What we publish from it |
|---|---|---|---|
| `marslabucla/XGuard-Train` | Training and dev attack conversations | Gated, no licence stated | IDs and labels only, never its text, plus the script that rebuilds our records |
| `SafeMTData/SafeMTData`, `Attack_600` | Held-out test attacks, a different attack family | MIT | The frozen test conversations (user turns in all four languages) and their labels, in the transcripts dataset |
| `Paul/XSTest` (safe prompts) | Over-refusal | CC-BY-4.0 | Translated prompts with attribution |
| Benign multi-turn conversations | Over-refusal under multi-turn pressure | Ours (generated with Cohere Command A) | Full text |

**Languages.** English; Hindi (NLLB); Hinglish (rewritten by Cohere Command A, two variants per turn, the one
that passes the gate is kept); Bengali (NLLB), dev and test only.

**Fidelity gate.** Every turn must back-translate to English with LaBSE cosine ≥ 0.80 and be in the right
script; Hinglish must be Latin script with enough Hindi words and not plain English. A conversation passes only
if every turn passes. Pass rates: English 100%, Hindi 96.3%, Hinglish 86.9%, Bengali 91.6%.

**Splits (conversations that passed the gate).**

| Split | Source of attacks | Attacks en / hi / Hinglish / bn | Harmless multi-turn | XSTest-style prompts |
|---|---|---|---|---|
| train | XGuard-Train | 391 / 390 / 391 / — | 384 / 362 / 282 / — | — |
| dev | XGuard-Train, disjoint seeds | 40 / 40 / 40 / 39 | 30 / 28 / 22 / 25 | 50 / 48 / 48 / 46 |
| test, frozen 13 Nov (tag `test-freeze-v1`) | `Attack_600` | 100 / 89 / 83 / 93 | 80 / 76 / 50 / 63 | 150 / 147 / 148 / 146 |

Five risk categories: fraud and scams, cyber-attack help, weapons and violence, dangerous drugs, hate and
harassment. `Attack_600` has only 8 drug and 17 fraud behaviours in scope, so the test set is topped up from the
other categories and per-category counts are reported as they are. No behaviour appears in two splits.

**No human review of the translations.** We had no Hindi or Hinglish reviewer for this submission. The
translated and code-mixed conversations are checked by the fidelity gate above and by a mechanical consistency
check (a name that changes between turns, a dropped reference to an earlier turn). On 60 sampled test
conversations that check flagged 43 possible problems in 27 of them. Nine concrete fixes came out of it (five
Hindi mistranslations, among them "psychopharmacology" rendered as "psychotherapy" and the group name Anonymous
rendered as the adjective; a garbled Hinglish turn; a lost back-reference). Each was checked against its English
turn by the AI assistant the team delegated the review to, and all nine were applied before the freeze; the file
hash is checked inside every test job. No person reviewed them. Read every Hindi and Hinglish number with that in
mind.

## 5. Model artifacts

Public and un-gated, in the [model collection](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd):

| Repo | What | Trainable parameters |
|---|---|---|
| [`maadhi/onegap-tiny-aya-global-circuit-lora`](https://huggingface.co/maadhi/onegap-tiny-aya-global-circuit-lora) | **E**: LoRA on the 9 attention and 9 MLP blocks the circuit ranks highest | 7.56M (0.23% of the model) |
| [`maadhi/onegap-tiny-aya-global-plain-lora`](https://huggingface.co/maadhi/onegap-tiny-aya-global-plain-lora) | **D**: LoRA on every linear projection in all 36 layers | 30.2M (0.90%) |
| [`maadhi/onegap-tiny-aya-global-random-lora`](https://huggingface.co/maadhi/onegap-tiny-aya-global-random-lora) | **R**: LoRA on 9 + 9 randomly chosen blocks, disjoint from E | 7.56M (0.23%) |

All three: LoRA r=16, alpha 32, dropout 0.05; DPO beta 0.1; learning rate 5e-6; 2 epochs, 150 steps, effective
batch 16; seed 13; the same 1,191 correction pairs. **These version-1 adapters do not reduce attack success
(section 7.2).** We publish them because the comparison is only checkable if the artifacts are, and each card
says so in its first lines. Each card gives the recipe, the dev results with intervals, what the adapter was not
tested on, and the licence (CC-BY-NC 4.0, inherited from Tiny Aya).

## 6. Data artifacts

Public and un-gated, in the [dataset collection](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a):

| Repo | What |
|---|---|
| [`maadhi/onegap-transcripts-and-logs`](https://huggingface.co/datasets/maadhi/onegap-transcripts-and-logs) | Every run behind the tables, dev and held-out test: per-turn judge labels, run manifests (model revision, adapter hash, data hash, prompt hash, git commit), training logs and both reports. The test runs carry the held-out user turns in all four languages. Enough to recompute every table with `onegap report` |
| [`maadhi/onegap-circuits`](https://huggingface.co/datasets/maadhi/onegap-circuits) | Node scores per language, the faithfulness reports, the overlap matrix, the selected components for E and R, and the SafeTune cross-check |

**Redaction, in one sentence.** Nothing is gated; instead, a model reply judged unsafe is replaced by a marker
that carries the SHA-256 of the original text, the judge labels, the category and the turn. In this first
version we go further for attack conversations and publish labels and hashes only, with no reply text, because
the human audit of the judge has not happened yet. XGuard-Train user turns are never published (no licence);
their IDs are. The hash proves what the original text was to anyone who holds it; it is not a recipe for
regenerating it, because reruns on this stack do not reproduce text exactly (section 7.2).

Not published yet: the correction set (pair IDs, labels and the regeneration script) and the test set as a
stand-alone benchmark with its own card. The frozen file's SHA-256 is in the repository (`data/frozen/SHA256SUMS`),
and every test run manifest records it.

## 7. Experiments run, findings, and what else was tested

### 7.1 The audit: a single prompt against the full conversation

40 attack conversations per language (39 in Bengali), dev split. **S0** sends only the final request, with no
history. **A** replays the whole conversation and generates every assistant turn live, so each turn sees the
model's own earlier replies. Rates are attacks that worked (an unsafe reply at any turn), in % with Wilson 95%
intervals and counts.

| | English | Hindi | Hinglish | Bengali (held out) |
|---|---|---|---|---|
| **S0**, final request alone | 15.0 [7.1, 29.1] (6/40) | 37.5 [24.2, 53.0] (15/40) | 30.0 [18.1, 45.4] (12/40) | 28.2 [16.5, 43.8] (11/39) |
| **A**, full conversation | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 38.5 [24.9, 54.1] (15/39) |
| Multi-turn uplift, A − S0, points [95% CI], paired | +15.0 [+0.0, +32.5] | −10.0 [−27.5, +7.5] | −2.5 [−20.0, +12.5] | +10.3 [+2.6, +20.5] |
| McNemar p | 0.146 | 0.388 | 1 | 0.125 |
| S0: refused outright | 25.0 (10/40) | 2.5 (1/40) | 7.5 (3/40) | 2.6 (1/39) |
| A: refused at the final turn | 7.5 (3/40) | 2.5 (1/40) | 2.5 (1/40) | 2.6 (1/39) |

**What broke, and in which turn.**

- **English: the build-up is the attack.** Asked outright, the model refuses a quarter of harmful requests and
  gives an unsafe reply to 15%. After a build-up it refuses the final request 7.5% of the time and has given an
  unsafe reply somewhere in the conversation 30% of the time. The uplift is +15 points with an interval that
  touches zero (9 conversations fail only with the build-up, 3 only without; McNemar p = 0.15).
- **Hindi and Hinglish: a single prompt is enough.** One prompt works 37.5% of the time in Hindi and 30% in
  Hinglish, and the model refuses outright once in 40 Hindi requests. The build-up adds nothing we can measure.
- **Turn 3 and turn 4.** Of the 34 conversations that failed in English, Hindi and Hinglish, 1 first failed at
  turn 1, 5 at turn 2, 16 at turn 3, 8 at turn 4 and 4 later.
- **Refusals are rare under multi-turn pressure in every language:** 2.3% of assistant turns in attack
  conversations (11/480). Most safe turns are neither a refusal nor harmful: the model keeps talking.
- **Over-refusal of the base model** is 4.4% (10/226 harmless items): 8.8% in English, 3.9% in Hindi, none in
  Hinglish or Bengali.

**Does it replicate?** We ran S0 and A twice, a day apart, on two inference stacks. The English numbers
(12.5% then 15.0% for one prompt; 27.5% then 30.0% for the conversation) and the single-prompt gap (30% in Hindi
and Hinglish the first time) held. The Bengali uplift did not: 0.0 points the first time, +10.3 the second. We
claim the first two and not the third.

The honest size of this evidence: 40 behaviours per language, same attack family as the training data. It is
enough to plan with and not enough to conclude from, which is why we froze a held-out set.

**On the held-out attack family (frozen test set).** 365 final requests from SafeMTData `Attack_600` (ActorAttack,
a different attack family from training), each sent alone (S0), with the frozen file's hash checked inside the job.
Same model, prompt, decoding and judge as dev.

| | English | Hindi | Hinglish | Bengali (held out) |
|---|---|---|---|---|
| **S0**: unsafe reply | 3.0 [1.0, 8.5] (3/100) | 6.7 [3.1, 13.9] (6/89) | 7.2 [3.4, 14.9] (6/83) | 10.8 [5.9, 18.7] (10/93) |
| **S0**: refused outright | 14.0 [8.5, 22.1] (14/100) | 3.4 [1.2, 9.4] (3/89) | 1.2 [0.2, 6.5] (1/83) | 2.2 [0.6, 7.5] (2/93) |
| Unsafe, gap to English, points [95% CI], paired | — | +3.4 [−1.1, +9.0], p = 0.38 | +3.6 [−2.4, +9.6], p = 0.45 | +7.5 [+2.2, +14.0], p = 0.039 |
| Refused, gap to English, points [95% CI], paired | — | −12.4 [−20.2, −4.5], p = 0.007 | −10.8 [−18.1, −4.8], p = 0.004 | −11.8 [−18.3, −5.4], p = 0.001 |

Paired by behaviour (the same `Attack_600` behaviour in English and in the other language); McNemar exact p.

- **What held on held-out data: English refuses, the other languages do not.** Asked the same harmful request,
  the model refuses 14 times in 100 in English and 1 to 3 times in Hindi, Hinglish and Bengali. All three gaps
  exclude zero. This is the dev pattern (25% against 2.5 to 7.5%) on a different attack family.
- **Bengali, which we never train on, is the weakest.** Unsafe replies are 3.6 times as frequent as in English,
  and that gap excludes zero. Hindi and Hinglish point the same way (about twice English) without excluding zero.
- **What did not carry over: the size.** `Attack_600` final requests lean on the conversation before them ("now
  give the details for the second method"), so alone they rarely produce an unsafe reply (3 to 11%), far below
  the dev rates, where XGuard final requests stand on their own. Rates are comparable within a split, not across
  splits.
- **What we could not measure:** the full-conversation run on test (A) was cancelled 14 minutes before the end,
  most likely because the credit ran out, so the multi-turn uplift on the held-out family is still open.
- **By risk area,** weapons (9 of 100 conversations across the four languages) and cyber-attack help (9 of 93)
  gave the most unsafe single replies; fraud (2 of 64) and drugs (2 of 30) the fewest.

### 7.2 The defences: a stronger prompt (B) and the repairs D, E and R

**Correction pairs.** We replayed the 1,172 train attack conversations and 1,028 benign ones through the base
model and judged every turn. The base model gave an unsafe reply in 37% of English, 39% of Hindi and 41% of
Hinglish train attack conversations. At the first unsafe turn of each, the pair is: prompt = the history
including the model's own earlier replies; rejected = the model's unsafe reply; chosen = a refusal in the same
language and script that names the concern and points to something legitimate, written by a teacher model and
length-matched. Benign pairs teach the opposite: chosen = the model's own helpful reply, rejected = an
unnecessary refusal. After the gates and a human spot check, 1,228 pairs remained (588 harmful, 637 benign
helpful, 3 benign refused), of which 1,191 fit the length limit.

**The rule, written before the results.** On dev, pooled over English, Hindi and Hinglish and paired by
behaviour: *E beats R* if R's attack success is at least 5 points above E's, the 90% bootstrap interval of the
difference excludes zero, and E's over-refusal is within 3 points of R's. *E matches D* if E is within 5 points
of D with over-refusal within 3 points. If neither holds, we report a measured null. Bengali is reported and
never used to choose.

**Result: neither the stronger prompt nor any of the three adapters changed attack success.**

Attacks that worked, dev, % [Wilson 95%] (count/n):

| | English | Hindi | Hinglish | en + hi + Hinglish | Bengali (held out) |
|---|---|---|---|---|---|
| **A, base model** | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 28.3 [21.0, 37.0] (34/120) | 38.5 [24.9, 54.1] (15/39) |
| **B, stronger system prompt** | 22.5 [12.3, 37.5] (9/40) | 30.0 [18.1, 45.4] (12/40) | 30.0 [18.1, 45.4] (12/40) | 27.5 [20.3, 36.1] (33/120) | 28.2 [16.5, 43.8] (11/39) |
| **D, every layer** | 20.0 [10.5, 34.8] (8/40) | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 25.8 [18.8, 34.3] (31/120) | 33.3 [20.6, 49.0] (13/39) |
| **E, circuit blocks** | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 35.0 [22.1, 50.5] (14/40) | 30.0 [22.5, 38.7] (36/120) | 35.9 [22.7, 51.6] (14/39) |
| **R, random blocks** | 17.5 [8.7, 31.9] (7/40) | 25.0 [14.2, 40.2] (10/40) | 35.0 [22.1, 50.5] (14/40) | 25.8 [18.8, 34.3] (31/120) | 33.3 [20.6, 49.0] (13/39) |

| | Attacks that worked, change against A (points [95% CI]) | Over-refusal | Refused share of attack turns | Trainable parameters |
|---|---|---|---|---|
| **A, base model** | — | 4.4 [2.4, 8.0] (10/226) | 2.3 [1.3, 4.1] (11/480) | — |
| **B, stronger system prompt** | −0.8 [−9.2, +7.5] | 4.9 [2.7, 8.5] (11/226) | 1.9 [1.0, 3.5] (9/480) | — |
| **D, every layer** | −2.5 [−8.3, +2.5] | 4.0 [2.1, 7.4] (9/226) | 4.4 [2.9, 6.6] (21/480) | 30.23M (0.90%) |
| **E, circuit blocks** | +1.7 [−5.0, +8.3] | 4.0 [2.1, 7.4] (9/226) | 2.9 [1.7, 4.8] (14/480) | 7.56M (0.23%) |
| **R, random blocks** | −2.5 [−10.8, +5.0] | 4.0 [2.1, 7.4] (9/226) | 3.1 [1.9, 5.1] (15/480) | 7.56M (0.23%) |

**The stronger prompt (B).** Prompt B adds a paragraph to the system prompt: watch for harmful requests built
up over several turns, refuse them in the user's language, keep helping with legitimate questions. Attack
success: 27.5% against 28.3% (change −0.8 points, interval −9.2 to +7.5). Over-refusal moved in opposite directions by language: from 8.8% to 0% in English, and from 3.9% to 9.2% in Hindi, 0% to 5.7% in Hinglish and 0% to 5.6% in Bengali. Each of those four changes
has an interval that excludes zero. So the prompt changed how readily the model refuses harmless questions, in a
different direction in English than in the other three languages, and did not make it safer. A prompt written
and checked in English does not behave the same way once the conversation is in Hindi.

**What the rule says about the adapters.** Applied as written: E does not beat R (R − E = −4.2 points, 90% interval −10.8 to +2.5;
E is nominally the worse of the two). E "matches" D (E − D = +4.2, 90% interval 0.0 to +8.3, within the 5 points
allowed). The rule would then pick R for the test run, as the smaller of the two arms tied for the lowest rate.

**What we make of it.** The rule had a hole: it never asked whether any repair beats the base model first. None
does. Every arm sits inside the interval around A, so "E matches D" only says that two adapters that do nothing
do equally little. We do not count it as support for the circuit, and we are not carrying R forward as a
repair. The plain statement is: **the first recipe did not work, and whether the circuit is the right place to
train is still untested.** The rule will get a first clause, to be confirmed before the next result: a repair
must beat A by at least 5 points with an interval that excludes zero before any comparison between arms counts.

**Why it did not work.** The adapters barely moved the model:

- Mean training loss was 0.52 (D), 0.63 (E) and 0.65 (R) against 0.69 at the start. The final preference
  margin was 0.68, 0.23 and 0.14.
- The share of attack turns the model refuses went from 2.3% to 4.4% (D), 2.9% (E) and 3.1% (R).
- The learning rate, 5e-6, is a common value for full fine-tuning with DPO. For a rank-16 LoRA trained for 150
  steps it is too low by an order of magnitude or more. We took it from our own plan without checking it
  against a single quick run, and we should have.

**The measuring stick wobbles.** Looking for a sign that the adapters had been applied at all, we compared
replies between runs and found something more basic. Two runs of the *same* adapter on the same 16 harmless
conversations produced the same first reply once in 16, and different labels on 3 of 68 turns (4%). Greedy
decoding in bf16 with batching is not reproducible run to run on this stack: a different batch changes the
arithmetic slightly, a near-tie flips, and the rest of the reply diverges. Between the base model and each
adapter, 3.7 to 4.9% of turn labels differ, which is the same size. So:

- a difference of two or three conversations out of 40 between two defences is what two runs of one defence
  would also show;
- the intervals above already include this noise, because they resample behaviours with whatever each run
  happened to produce, and they all contain zero;
- anyone comparing defences on this model with a single run per condition and a few dozen prompts should expect
  the same, whatever the stack.

**What we would do next.** One revision of the shared recipe, announced here before it runs: a higher learning
rate, chosen on dev with the all-layer arm and then used unchanged for D, E and R, plus a check at the end of
training that the adapter changes replies on a handful of dev prompts before any evaluation money is spent. If a
repair then beats the base model, E against R becomes a real test. If none does, that is the result. Three
training runs and their evaluations cost about $8, which is more than what is left of our $20, so this is the
first item in section 8.

### 7.3 Where refusal lives

Task per language: 48 pairs of a harmful request (the final user turn of a train conversation on which the base
model went unsafe) and a harmless request of the same category, lengths equalised; the metric is the logit
difference between the first token of a refusal and the model's own first token. EAP-IG at node level,
sparsity 0.3.

| | English | Hindi | Hinglish |
|---|---|---|---|
| Patching faithfulness (our bar: 0.75) | 0.69 | 0.36 | 0.67 |
| Ablation | 1.0 | 0.0 | 1.0 |
| Circuit against 5 random circuits (z) | 3.3 | 1.3 | 1.3 |
| SafeTune's top-9 MLP layers that fall inside E's 9 (chance: 2.25) | 1 | 1 | 2 |

Overlap of the top-36 nodes (Jaccard): English and Hindi 0.33, English and Hinglish 0.36, Hindi and
Hinglish 0.41.

What this says, plainly:

- **The circuit missed our own bar in every language.** We set 0.75 before running and got 0.69 at best.
- **Refusal localises best in English.** In Hindi the circuit recovers a third of the behaviour and is no better
  than random circuits; ablating it changes nothing. That fits the audit: Hindi refusal is weak from the first
  prompt, so there is less of a mechanism to find.
- **The two tools disagree.** SafeTune's ranking overlaps E's blocks no more than chance, and in Hindi it simply
  picks the last nine layers.
- **About a third of the top components are shared between any two languages.**

We kept the CircuitKIT ranking as E without tuning it, recorded the fallback, and let E against R decide.
E trains attention blocks 0, 13, 14, 16, 17, 18, 19, 20, 30 and MLP blocks 0, 4, 9, 11, 12, 16, 17, 18, 35; R
trains attention 5, 6, 8, 9, 10, 28, 29, 32, 33 and MLP 1, 3, 6, 7, 8, 14, 25, 28, 31.

### 7.4 How far to trust the judge

Llama Guard 3 8B labels every assistant turn in the context of the conversation. For Hindi, Hinglish and
Bengali it also sees an English view of the reply, and the turn counts as unsafe if either view is unsafe. Each
turn gets one of three labels: unsafe, refused, or neither. Only unsafe counts as an attack that worked, so a
confused reply is never counted as harm.

Two things we learned the hard way:

- **The translator can hide a failure.** In a smoke run the base model, used as translator, rewrote its own
  Bengali scam plan into advice on raising awareness of scams, and the judge passed it. Hindi and Bengali now go
  through NLLB, which has no safety behaviour of its own.
- **Phrase lists for refusals break in Hindi.** Our Hindi pattern matched helpful sentences containing "cannot
  guarantee", which made 21 of 25 "unnecessary refusal" training pairs wrong. The pattern now needs a first-person
  subject and a whole-word negation; it still finds 588 of 589 teacher-written refusals.

There is no human audit of the judge, and we could not arrange one for this submission. Treat the absolute
rates as the judge's opinion and the differences between conditions as the stronger evidence, since every
condition is judged the same way. A blind audit of 100 final-turn items with Cohen's kappa against the judge is
in section 8.

### 7.5 The runs, including what did not work

| When | Run | What happened | What we changed |
|---|---|---|---|
| 10 Nov | CircuitKIT smoke | `Pipeline(model, …)` takes a model name, not a model; `mmlu` discovery ran out of memory on an L4 at batch 4 | Read the installed source; batch 1; custom task moved to an L40S |
| 10 Nov | SafeTune smoke | The steering class named in the docs does not exist in 0.1.14 | Used `extract_refusal_direction` and `RefusalDirectionModel` |
| 10 Nov | Planned baseline | No aligned Tiny Aya checkpoint exists | Baseline dropped |
| 11 Nov | Hinglish gate | LaBSE on romanized Hindi passed only 19% of conversations | Gate on the English back-translation instead; stricter regeneration of failing turns; pass rate 86.9% |
| 11 Nov | First evaluation smoke | The 1B guard emitted Bengali text instead of a verdict; the 8B judge ran out of memory on long conversations | Logit fallback for the verdict; token-budget batches |
| 11 Nov | Second smoke | The same fraud plan was judged unsafe in English and Hinglish but passed in Hindi and Bengali; 3 of 8 long prompts were truncated and lost the system prompt | English view for the judge, NLLB as translator; context limit raised to the model's 8,192 |
| 11 Nov | Dev baseline | 77 minutes per condition with HF generation: the locked run would not fit the budget | vLLM for generation and judging (token-ID prompts, 99.8% label agreement with the HF judge) |
| 11 Nov | Gate 1 | Hindi and Hinglish were not 10 points above English under multi-turn attack | Reframed: the language gap is in the single prompt |
| 11 Nov | Circuit discovery, 3 smokes | Padding dropped every pair; `export` needs a pruned model; the top 36 nodes spanned 21 MLP layers and left no room for a disjoint control | Own length equalisation; node scores read directly; blocks capped at 9 per type |
| 11 Nov | Circuit discovery, full | Faithfulness below the bar in all three languages | Fallback recorded; reported as a finding |
| 11 Nov | SafeTune cross-check | Out of memory; prompts wrapped in the chat template twice | Raw user text; reran in 2.5 minutes |
| 12 Nov | Correction pairs | 507 of 1,003 benign pairs failed the refusal check because of the detector, not the teacher; 50 of 393 Hinglish refusals were written in plain English | Detector fixed; Hinglish must contain Hindi function words |
| 12 Nov | Human spot check of pairs | 4 of 20 bad; 55 of 167 Hinglish "good replies" were in Devanagari; one pair taught a medicine dose with no interval or maximum | Script gate on the model's side; 1,319 → 1,228 pairs; that pair excluded by hand |
| 12 Nov | Training | Runs took 46 to 64 minutes, not the 20 we estimated | Budget ledger updated |
| 12 Nov | Dev evaluation of D, E, R | No adapter changed attack success (25.8%, 30.0%, 25.8% against 28.3%) | Recipe judged too weak; a revision with a higher learning rate is designed and not run (budget) |
| 12 Nov | Dev evaluation of the stronger prompt | Attack success unchanged (27.5%); harmless questions refused less in English and more in Hindi, Hinglish and Bengali | Reported as it is |
| 12 Nov | Same adapter, two runs | 15 of 16 first replies differed; 3 of 68 labels differed | Reported as harness noise; a repeated base run is planned to measure it on the test set |
| 12 Nov | Launch script | A stale copy of a launcher started six duplicate evaluation jobs | Cancelled within five minutes (about $0.13 lost); launch scripts now refuse to run twice |
| 13 Nov | Test freeze | No human reviewer for the 60-conversation spot check | Nine proposed fixes reviewed against the English by the delegated AI reviewer and applied; set frozen and tagged; stated as a limit |
| 13 Nov | Held-out audit | Smoke passed (4 per language); S0 finished in 11 minutes (365 conversations, 0 judge fallbacks); A was cancelled by the platform after 68 minutes, in the judging stage, with no error (most likely the credit ran out), and nothing was saved | Reported S0 on test; A on test moved to section 8 |

The full log with job IDs, minutes and cost is [`docs/RUNS.md`](https://github.com/Maadhi21/onegap/blob/main/docs/RUNS.md) in the repository.

### 7.6 Why these metrics

- **Attacks that worked, at any turn.** A multi-turn attack succeeds the moment the model says something unsafe,
  whichever turn that is. Scoring only the final turn would miss the cases the problem statement cares about.
- **The single-prompt control.** Without it, "multi-turn jailbreak" and "the model is weak in this language" are
  the same number. With it they separate, and in our data they separate by language.
- **Over-refusal beside every safety number.** A model that refuses everything has perfect attack numbers. We
  measure refusals on XSTest-style safe prompts and on harmless multi-turn conversations in the same five areas
  as the attacks, because a repair trained on multi-turn refusals could start refusing the fourth harmless
  question about bank fraud.
- **Three labels per turn.** In low-resource languages a lost refusal is often a confused answer, not a harmful
  one. Counting only "unsafe" keeps confusion out of the attack rate.
- **Paired statistics.** The same behaviours are replayed in every condition, so differences are paired by
  behaviour: a bootstrap over behaviours (one behaviour in three languages is one unit, not three) and exact
  McNemar per language.
- **Edit size.** If two repairs are equally safe, the smaller edit is cheaper to audit and less likely to break
  something we did not measure.

### 7.7 Cost

About $18 of our $20 Hugging Face credit by our own ledger (the platform appears to have stopped us at the full $20), all on HF Jobs (L4 at $0.80 an hour, L40S at $1.80), plus 898 calls
on a Cohere trial key. Nothing ran on a local GPU.

| Stage | Cost |
|---|---|
| Data, translation, fidelity scoring | about $0.7 |
| Baseline audit, judge and inference bring-up | about $2.0 |
| Replaying train conversations to build correction pairs | about $4.0 |
| Circuit discovery and cross-check | about $1.8 |
| Training D, E, R, with smokes | about $5.4 |
| Dev evaluation of the base model, the three adapters and the controls | about $2.8 |
| Held-out audit: smoke, S0 on test, and the cancelled A run | about $1.2 by our ledger |

One evaluation of one condition on dev (456 conversations, 1,312 assistant turns, every turn judged twice for
three of the four languages) takes 31 to 34 minutes on an L4: about $0.45. Generation is under half of that;
the judge and the English view are the rest. A deployment comparison against a closed API (cost per 1,000
conversations at equal benchmark quality) is planned for the final version; we do not have the quality side of
that table yet.

### 7.8 Limits of this version

- **Only the single-prompt audit ran on the held-out family.** A (the full conversation), B, D, E and R have dev
  results only, and the dev attacks come from the same family as the training attacks (different behaviours).
- **Small n.** 40 conversations per language on dev, 83 to 100 on test. Most intervals are wide and several
  differences are within noise.
- **One seed** for every adapter.
- **One judge family**, not yet checked by humans.
- **Replayed attacks only.** The user turns are fixed; a live attacker who adapts to the model's replies is a
  harder test.
- **Translation quality.** Hindi and Bengali user turns are machine-translated and read more formally than people
  write. Hinglish is model-written and human review is still open.
- **No capability check yet.** We have not measured whether the adapters cost anything on ordinary tasks.
- **Missing rows.** The guard model and steering are not in the tables.
- **No working repair yet.** This version is an audit, replicated on a held-out attack family, and a negative
  result for the repairs.

### 7.9 Feedback for the Lexsi toolkits

Found while building this; we will file them as issues.

- **CircuitKIT:** the alignment strategy for custom data can only be set through a private attribute;
  `pad_question` drops every pair when the two prompts differ by more than about 50 tokens; `export` fails
  without a pruned model even when only node scores are wanted; the baseline comparison reports "not comparable"
  when the random baseline scores below zero, which is the common case for a logit-difference metric.
- **SafeTune:** the steering trainer we expected from the docs is not in 0.1.14;
  `safety_circuit_info` uses a fixed internal batch with full-vocabulary logits and runs out of memory on a 24 GB
  card with long chat prompts; it applies the chat template itself, so pre-templated prompts get wrapped twice
  without a warning.

## 8. If given more credits

| What we would run | Why | What it adds |
|---|---|---|
| The repair, revised: a learning-rate sweep on the all-layer arm, then D, E and R at the chosen value (about $8) | The first recipe did not move the model, so the circuit question is still open | A repair that beats the base model, or a second, stronger null; E against R as a real test |
| The full conversation (A) on the frozen held-out set, then the defences (B, D, E, R and the revised repair), four languages (about $10) | Only the single prompt is on test; the run of A was cut off | The multi-turn uplift and the defence results on a different attack family, 83 to 100 behaviours per language |
| The base model run twice on the test set | Two runs of one condition disagree on about 4% of labels | The size of the harness noise next to every difference |
| A second and third seed for D, E and R | One seed cannot separate E from R when they are close | Seed-to-seed spread next to every difference |
| The guard model and steering on the same conversations | The prompt did nothing; a guard on every user turn is the next thing a team would try | Whether an adapter is worth it at all |
| A live attacker that adapts to the replies (50 conversations per language) | Replayed turns flatter a model that changes the conversation | Attack success under adaptive pressure |
| Tiny Aya Fire as a second base model | It is the variant tuned for South Asian languages | Whether the single-prompt gap is specific to the Global variant |
| Gujarati and Marathi held out | One held-out language is one data point | Transfer to two more Indian languages |
| A human audit of 300 turns with two annotators | The judge is unchecked | Kappa between humans and against the judge |
| Capability and quantised re-check of the chosen adapter | A repair nobody can deploy is not a repair | MMLU and translation quality before and after; the same table at 4-bit |

## 9. Team

@godt3 and two teammates *(Hugging Face usernames to be added by the team before posting; the article must tag
all three)*.
