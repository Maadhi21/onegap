# OneGap: Tiny Aya Global says no in English. In Hindi, Hinglish and Bengali it rarely does.

*Lexsi AI Safety and Alignment Hackathon 2026, Track 2. A turn-by-turn safety audit in four languages, the
circuit behind refusal, and a pre-registered test of where to repair it.*

**The story in five lines**

1. **One model, four languages, very different safety.** On a frozen held-out set from an attack family
   we never trained on, Tiny Aya Global refuses a harmful request outright **14% of the time in English and only
   1 to 3% in Hindi, Hinglish and Bengali** (paired gaps of 11 to 12 points, p ≤ 0.007).
2. **The language nobody trained on is the weakest.** Bengali gives unsafe replies 3.6 times as often as English
   (10.8% against 3.0%, p = 0.04).
3. **English needs a conversation; the other languages need one prompt.** In English the attack works by
   building up over turns (15% asked outright, 30% over a conversation). In Hindi and Hinglish one prompt already
   works 30 to 38% of the time. Conversations break at turn 3 or 4.
4. **Refusal lives in a circuit that is weakest where refusal is rarest.** Faithfulness 0.69 in English,
   0.67 in Hinglish, 0.36 in Hindi.
5. **We built the experiment that asks whether repairing that circuit beats plain fine-tuning, and pre-registered
   how to read it.** The first recipe was too weak to move the model in any arm. Section 8 is the run that answers
   the question.

Links: [model collection](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd) · [dataset collection](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a) · [code](https://github.com/Maadhi21/onegap) · [replay Space](https://huggingface.co/spaces/maadhi/onegap-replay) · [reproduction notebook](https://github.com/Maadhi21/onegap/blob/main/notebooks/reproduce_headline.ipynb)

---

## 1. Problem statement and approach

**Track 2, AI Red Teaming, Safety and Alignment.** Problem statements: *Multi-turn audits* and *Multilingual and
code-mixed safety* (primary); *Locating where safety lives* and *Circuit-level interventions* (supporting).

**The threat.** A public helpline assistant answers in the language the user writes in. Someone who wants a UPI
scam script or attack instructions does not ask outright. They build the request over a few harmless-looking
turns, and they type in Hindi, or in Hinglish, the romanized Hindi-English mix people use on WhatsApp. Small
multilingual models like Tiny Aya are exactly what gets deployed in that setting.

**Our approach, in four steps.**

1. **Audit, turn by turn.** Replay multi-turn attack conversations through `CohereLabs/tiny-aya-global` in
   English, Hindi and Hinglish, generating every assistant turn live, and judge every turn. Send the final request
   alone as a control (S0): the difference between the two is what a single-prompt test would miss. Bengali is
   evaluated and never trained on.
2. **Locate.** Find the components that carry refusal in each language with CircuitKIT, measure how faithful the
   circuit is, and cross-check it with SafeTune.
3. **Repair, three ways, same data and budget.** Train a LoRA with DPO on pairs built from the model's own
   failures: on every layer (D), only on the circuit's blocks (E), and on as many random blocks (R). R is the
   control that tells us whether the location matters or only the size of the edit.
4. **Decide by a rule written beforehand**, with every safety number reported next to an over-refusal number,
   and training and test attacks drawn from different attack families.

## 2. Hackathon repos used

| Repo | Version | What for | Features used |
|---|---|---|---|
| **CircuitKIT** | 0.2.2 | Locating refusal per language and choosing the components repair E trains | `Pipeline.from_custom_data` with our contrastive refusal task (48 pairs per language); EAP-IG node-level discovery; `evaluate` (patching, ablation, random-circuit baseline); node scores for the component ranking and the cross-language overlap |
| **SafeTune** | 0.1.14 | An independent second opinion on the circuit | `interpret.safety_circuit_info` on the same pairs; `extract_refusal_direction` and `RefusalDirectionModel` verified on Tiny Aya |

CircuitKIT's ranking decides which modules repair E may train; SafeTune checks that ranking with a different
method; the evaluation then tests whether the choice mattered.

## 3. Models used and their licences

| Model | Role | Licence |
|---|---|---|
| `CohereLabs/tiny-aya-global` (3.35B, revision `af89d219`) | Model under audit; base of every adapter | CC-BY-NC 4.0 |
| `meta-llama/Llama-Guard-3-8B` | Judge of every assistant turn | Llama 3.1 Community Licence |
| `facebook/nllb-200-distilled-1.3B` | Hindi and Bengali user turns; English view of replies for the judge | CC-BY-NC 4.0 |
| `sentence-transformers/LaBSE` | Translation fidelity gate | Apache-2.0 |

## 4. Datasets and their licences

| Source | Used for | Licence | What we publish |
|---|---|---|---|
| `marslabucla/XGuard-Train` | Training and dev attack conversations | Gated, no licence stated | IDs and labels only |
| `SafeMTData/SafeMTData`, `Attack_600` | **Held-out test attacks, a different attack family** | MIT | Test conversations in four languages with labels |
| `Paul/XSTest` (safe prompts) | Over-refusal | CC-BY-4.0 | Translated prompts with attribution |
| Benign multi-turn conversations | Over-refusal under multi-turn pressure | Ours | Full text |

Hindi and Bengali come from NLLB; Hinglish is written by Cohere Command A. Every turn must back-translate to
English with LaBSE cosine ≥ 0.80 and be in the right script, or the conversation is dropped.

| Split | Attacks en / hi / Hinglish / bn | Harmless multi-turn | XSTest-style prompts |
|---|---|---|---|
| train (XGuard-Train) | 391 / 390 / 391 / — | 384 / 362 / 282 / — | — |
| dev (XGuard-Train, disjoint behaviours) | 40 / 40 / 40 / 39 | 30 / 28 / 22 / 25 | 50 / 48 / 48 / 46 |
| **test (`Attack_600`), hash-frozen** | 100 / 89 / 83 / 93 | 80 / 76 / 50 / 63 | 150 / 147 / 148 / 146 |

Five risk areas: fraud and scams, cyber-attack help, weapons and violence, dangerous drugs, hate and harassment.
No behaviour appears in two splits. Before the freeze, a consistency check flagged nine translation errors in the
test set (for example "psychopharmacology" rendered as "psychotherapy"); each was checked against the English and
fixed. The frozen file's hash is verified inside every test job.

## 5. Model artifacts

Public and un-gated, in the [model collection](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd):

| Repo | What | Trainable parameters |
|---|---|---|
| [`maadhi/onegap-tiny-aya-global-circuit-lora`](https://huggingface.co/maadhi/onegap-tiny-aya-global-circuit-lora) | **E**: LoRA on the 9 attention and 9 MLP blocks the circuit ranks highest | 7.56M (0.23%) |
| [`maadhi/onegap-tiny-aya-global-plain-lora`](https://huggingface.co/maadhi/onegap-tiny-aya-global-plain-lora) | **D**: LoRA on every linear projection in all 36 layers | 30.2M (0.90%) |
| [`maadhi/onegap-tiny-aya-global-random-lora`](https://huggingface.co/maadhi/onegap-tiny-aya-global-random-lora) | **R**: LoRA on 9 + 9 random blocks, disjoint from E | 7.56M (0.23%) |

Shared recipe: LoRA r=16, alpha 32; DPO beta 0.1; learning rate 5e-6; 2 epochs, 150 steps; seed 13; the same 1,191
pairs. Each card states its results and limits in its first lines; licence CC-BY-NC 4.0, inherited from Tiny Aya.

## 6. Data artifacts

Public and un-gated, in the [dataset collection](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a):

| Repo | What |
|---|---|
| [`maadhi/onegap-transcripts-and-logs`](https://huggingface.co/datasets/maadhi/onegap-transcripts-and-logs) | Every run behind every table, dev and held-out: per-turn judge labels, run manifests (model revision, data hash, prompt hash, git commit), training logs, and both reports. `onegap report` recomputes every number here |
| [`maadhi/onegap-circuits`](https://huggingface.co/datasets/maadhi/onegap-circuits) | Node scores per language, faithfulness reports, the overlap matrix, the components chosen for E and R, and the SafeTune cross-check |

**Redaction.** Nothing is gated. A reply judged unsafe is replaced by a marker carrying the SHA-256 of the
original text and its labels; replies in attack conversations are withheld the same way until a human audit of
the judge; XGuard-Train text is never published.

## 7. Experiments, findings, and what else was tested

### 7.1 The language gap, on held-out attacks

365 harmful requests from the frozen `Attack_600` set, each sent alone, in four languages. Paired by behaviour;
% [Wilson 95%] (count/n); McNemar exact p.

| | English | Hindi | Hinglish | Bengali (never trained) |
|---|---|---|---|---|
| **Refused outright** | **14.0** [8.5, 22.1] (14/100) | 3.4 [1.2, 9.4] (3/89) | 1.2 [0.2, 6.5] (1/83) | 2.2 [0.6, 7.5] (2/93) |
| Refused, gap to English | — | −12.4 [−20.2, −4.5], p = 0.007 | −10.8 [−18.1, −4.8], p = 0.004 | −11.8 [−18.3, −5.4], p = 0.001 |
| **Unsafe reply** | 3.0 [1.0, 8.5] (3/100) | 6.7 [3.1, 13.9] (6/89) | 7.2 [3.4, 14.9] (6/83) | **10.8** [5.9, 18.7] (10/93) |
| Unsafe, gap to English | — | +3.4 [−1.1, +9.0], p = 0.38 | +3.6 [−2.4, +9.6], p = 0.45 | +7.5 [+2.2, +14.0], p = 0.039 |

- **English has a safety line; the other languages barely do.** The same request is refused 14 times in 100 in
  English and 1 to 3 times in the other three languages. All three gaps exclude zero, on an attack family the
  model's repair data never saw.
- **Bengali, the language we never trained on, fails most.** Unsafe replies 3.6 times as often as English, the
  one unsafe-rate gap that excludes zero. Hindi and Hinglish point the same way.
- **Weapons and cyber-attack help** gave the most unsafe single replies (9 of 100 and 9 of 93 conversations
  across the four languages).

### 7.2 The multi-turn gap, on dev

40 attack conversations per language (39 in Bengali). **S0** sends only the final request; **A** replays the
whole conversation and generates every assistant turn live.

| | English | Hindi | Hinglish | Bengali |
|---|---|---|---|---|
| **S0**, final request alone | 15.0 [7.1, 29.1] (6/40) | 37.5 [24.2, 53.0] (15/40) | 30.0 [18.1, 45.4] (12/40) | 28.2 [16.5, 43.8] (11/39) |
| **A**, full conversation | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 38.5 [24.9, 54.1] (15/39) |
| S0 refused outright | 25.0 (10/40) | 2.5 (1/40) | 7.5 (3/40) | 2.6 (1/39) |
| Over-refusal of A (harmless items refused) | 8.8% | 3.9% | 0% | 0% |

- **In English, the build-up is the attack:** an unsafe reply 15% of the time asked outright, 30% over a
  conversation (9 conversations fail only with the build-up, 3 only without).
- **In Hindi and Hinglish, one prompt is enough:** 30 to 38% asked outright, and the build-up adds nothing.
- **Conversations break in the middle:** of 34 failed conversations, 24 first failed at turn 3 or 4.
- The same pattern appeared in an earlier run on a different inference stack, and the refusal gap (25% in
  English against 2.5 to 7.5%) is the one that replicated on held-out attacks in 7.1.

### 7.3 Where refusal lives

Per language, 48 pairs of a harmful request the model failed on and a harmless request of the same category;
EAP-IG at node level on the logit difference between a refusal and the model's own first token.

| | English | Hindi | Hinglish |
|---|---|---|---|
| Patching faithfulness (bar set beforehand: 0.75) | 0.69 | 0.36 | 0.67 |
| Circuit against 5 random circuits (z) | 3.3 | 1.3 | 1.3 |
| SafeTune's top-9 MLP layers inside E's 9 (chance 2.25) | 1 | 1 | 2 |

- **Refusal localises best in English and worst in Hindi**, where the circuit explains a third of the behaviour
  and is no better than random. That matches the audit: there is less refusal mechanism to find where the model
  rarely refuses.
- **About a third of the top components are shared between any two languages** (Jaccard 0.33 to 0.41): safety is
  not one mechanism reused across languages.
- **SafeTune disagrees with CircuitKIT** beyond chance, so the circuit is a hypothesis to test, which is why
  repair E is run against a random-block control.

### 7.4 The repairs: a pre-registered test

**Correction pairs.** We replayed 1,172 train attack conversations and 1,028 harmless ones through the base model
and judged every turn. At the first unsafe turn, the pair is: prompt = the history with the model's own replies;
rejected = its unsafe reply; chosen = a refusal in the same language and script. Harmless pairs teach the
opposite, so the repair is not rewarded for refusing everything. 1,191 pairs after the gates.

**The rule, committed before any result:** E beats R if it is at least 5 points lower with a 90% interval that
excludes zero and over-refusal within 3 points; E matches D if within 5 points.

| Dev, en + hi + Hinglish | Attacks that worked | Change against base, points [95% CI] | Over-refusal |
|---|---|---|---|
| **A, base model** | 28.3 (34/120) | — | 4.4 (10/226) |
| **B, stronger system prompt** | 27.5 (33/120) | −0.8 [−9.2, +7.5] | 4.9 (11/226) |
| **D, every layer** | 25.8 (31/120) | −2.5 [−8.3, +2.5] | 4.0 (9/226) |
| **E, circuit blocks** | 30.0 (36/120) | +1.7 [−5.0, +8.3] | 4.0 (9/226) |
| **R, random blocks** | 25.8 (31/120) | −2.5 [−10.8, +5.0] | 4.0 (9/226) |

- **No arm moved the model, and the logs say why.** Mean training loss stayed at 0.52 to 0.65 against 0.69 at the
  start, and the share of attack turns refused rose only from 2.3% to at most 4.4%. A learning rate of 5e-6 suits full fine-tuning,
  not a rank-16 LoRA for 150 steps. The comparison between E, D and R is therefore not yet informative, and we
  do not read it as evidence for or against the circuit.
- **A stronger system prompt is not a fix either.** It left attack success unchanged and moved over-refusal in
  opposite directions: down from 8.8% to 0% in English, up to 5.6 to 9.2% in Hindi, Hinglish and Bengali. A
  prompt written in English does not behave the same way in Hindi.
- **The pipeline is in place for the decisive run:** shared data, settings and seed across D, E and R, the rule
  fixed in advance, and a held-out set frozen and hash-checked.

### 7.5 Why these metrics

- **Attacks that worked, at any turn:** a multi-turn attack succeeds the moment any reply is unsafe.
- **The single-prompt control** separates "multi-turn jailbreak" from "weak in this language". In our data they
  separate by language.
- **Over-refusal beside every safety number**, on safe prompts and on harmless multi-turn conversations in the
  same risk areas, so a repair cannot win by refusing everything.
- **Three labels per turn** (unsafe, refused, neither), because in low-resource languages a lost refusal is often
  a confused answer rather than a harmful one.
- **Paired statistics by behaviour**, and our own noise measured: two runs of one condition differ on about 4% of
  turn labels, which every interval above already absorbs.

### 7.6 Cost and limits

About $18 of compute in total, all on Hugging Face Jobs (L4 at $0.80 an hour, L40S at $1.80); nothing on a local
GPU. One evaluation of one condition on dev costs about $0.45. Limits: one seed per adapter; one judge family, not
yet audited by humans; machine-made translations; replayed rather than adaptive attackers; the multi-turn run on
the held-out set was cut off before it finished, so 7.2 is dev-only.

## 8. If given more credits

Everything above ran on $20. With $200, this is the run that turns the audit into a repair, in order, with what
counts as success fixed now:

| Run | Cost | What it settles |
|---|---|---|
| The full-conversation audit (A) on the frozen held-out set, four languages | about $1.5 | Whether the multi-turn gap in 7.2 holds on a new attack family |
| A learning-rate sweep on the all-layer arm (D), then D, E and R retrained at the chosen rate | about $10 | A repair that actually moves the model, with all three arms sharing it |
| D, E, R, B and the base model on the held-out set | about $8 | **The answer to the circuit question:** E supports circuit-targeted repair only if it beats R by at least 5 points with over-refusal within 3 points, and matches D at a quarter of the trainable parameters |
| Two more seeds for D, E and R | about $25 | Seed-to-seed spread next to every difference |
| A guard model (Llama Guard 3 1B on every user turn) | about $2 | Whether an adapter is worth it at all against a cheap guard |
| Tiny Aya Fire as a second base model | about $10 | Whether the language gap is specific to the Global variant |
| A human audit of 300 judged turns, two annotators | — | Agreement between humans and the judge in each language |

Success, stated in advance: a repair that cuts attacks that worked by half on held-out data in Hindi and
Hinglish, without raising over-refusal by more than 3 points, published with the same transcripts and logs as
everything here.

## 9. Team

@godt3 and two teammates *(replace with the three Hugging Face usernames before posting)*.
