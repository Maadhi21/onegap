# OneGap

**A turn-by-turn safety audit of Tiny Aya Global in English, Hindi and Hinglish, with Bengali held out, and a
test of whether repairing only the components a refusal circuit points to works better than plain fine-tuning or
a random-layer control.**

Lexsi AI Safety and Alignment Hackathon 2026, Track 2 (AI Red Teaming, Safety and Alignment).

> **Status on 12 Nov 2026: development-set results only.** The held-out test set is not frozen yet and the
> locked test run has not happened. Every number below comes from the dev split. The whole project ran on a $20
> compute credit; about $17 is spent. Nothing here should be read as a final result.

## The question

A public helpline assistant answers in the language the user writes in. A user who wants a scam script or
attack instructions does not have to ask outright: the request can be built up over several harmless-looking
turns, and it can be typed in Hindi or in Hinglish, the romanized Hindi-English mix people use on WhatsApp.

We ask four things about `CohereLabs/tiny-aya-global` (3.35B parameters):

1. Does a multi-turn build-up get past the model when the same final request, asked alone, does not?
2. Is the answer different in Hindi and Hinglish than in English? Does it carry to Bengali, which we never train on?
3. Where does refusal live in the model, and is it the same place in each language?
4. If we train only the components the circuit points to (E), is that as good as training every layer (D), and
   better than training the same number of randomly chosen components (R)? Same data, same budget.

## What we have so far (dev split)

### The audit: one prompt against the full conversation

40 attack conversations per language (39 in Bengali) from XGuard-Train. S0 sends only the final request with no
history. A replays the whole conversation and generates every assistant turn live. Rates are attacks that
worked: an unsafe reply at any turn, judged by Llama Guard 3 8B, in % with Wilson 95% intervals and counts.

| | English | Hindi | Hinglish | Bengali (held out) |
|---|---|---|---|---|
| **S0**, final request alone | 15.0 [7.1, 29.1] (6/40) | 37.5 [24.2, 53.0] (15/40) | 30.0 [18.1, 45.4] (12/40) | 28.2 [16.5, 43.8] (11/39) |
| **A**, full conversation | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 38.5 [24.9, 54.1] (15/39) |
| Multi-turn uplift, A − S0, points [95% CI], paired | +15.0 [+0.0, +32.5] | −10.0 [−27.5, +7.5] | −2.5 [−20.0, +12.5] | +10.3 [+2.6, +20.5] |
| McNemar p | 0.146 | 0.388 | 1 | 0.125 |
| S0: refused outright | 25.0 (10/40) | 2.5 (1/40) | 7.5 (3/40) | 2.6 (1/39) |
| A: refused at the final turn | 7.5 (3/40) | 2.5 (1/40) | 2.5 (1/40) | 2.6 (1/39) |

- **In English the build-up is what works.** Asked outright, the model gives an unsafe reply 15% of the time
  and refuses 25% of the time. Over a full conversation it gives an unsafe reply 30% of the time and refuses
  the final request 7.5% of the time.
- **In Hindi and Hinglish a single prompt is enough.** It already works 30 to 38% of the time, the model
  almost never refuses outright (1 of 40 in Hindi), and the build-up adds nothing.
- **When a conversation fails, it is usually at turn 3 or 4:** 24 of the 34 failed conversations in English,
  Hindi and Hinglish.

An earlier run of the same two conditions on a different inference stack gave the same picture for English
(12.5% against 27.5%) and for the single-prompt gap (30% in Hindi and Hinglish). The Bengali uplift did not
repeat (0.0 points the first time), so we do not claim it. With 40 conversations per language the intervals are
wide; the frozen test set, 100 conversations per language from a different attack family, decides.

### The defences: a stronger prompt (B) and three adapters (D, E, R) against the base model

**Neither the stronger system prompt nor any of the three adapters changed attack success.** Attacks that
worked, dev:

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

Every difference from the base model is inside its interval. The stronger prompt (a paragraph telling the model
to watch for requests built up over several turns and to refuse in the user's language) left attack success at
27.5%. Over-refusal moved in opposite directions by language: from 8.8% to 0% in English, and from 3.9% to 9.2% in Hindi, 0% to 5.7% in Hinglish and 0% to 5.6% in Bengali. The adapters did not move over-refusal. Our reading of the adapters: the first recipe was too
weak to change the model. Training barely moved the loss (0.69 at the start; mean 0.52
for D, 0.63 for E, 0.65 for R), and the share of attack turns the model refuses went from 2.3% to at most 4.4%.
The learning rate (5e-6) is a full fine-tuning value and is too low for LoRA. Whether the circuit is the right
place to train (E against R) cannot be read from adapters that do nothing, so that question is still open.

**The harness has noise of its own.** Two runs of the same adapter on the same 16 harmless conversations gave
the same first reply in 1 case of 16 and a different label on 3 turns of 68. Batched greedy decoding in bf16 is
not reproducible run to run on this stack. A difference of a few conversations out of 40 between two
conditions is what two runs of one condition also show.

Next would be one revision of the shared recipe (a higher learning rate, the same for D, E and R), chosen on
dev. It costs about $8, more than what is left of our budget, so it waits.

Full tables, paired tests and the pre-registered decision rule: [`docs/PROGRESS.md`](docs/PROGRESS.md).

### Where refusal lives

CircuitKIT (EAP-IG, node level, 48 contrastive pairs per language) and a SafeTune cross-check:

| | English | Hindi | Hinglish |
|---|---|---|---|
| Patching faithfulness (our bar: 0.75) | 0.69 | 0.36 | 0.67 |
| Circuit against 5 random circuits (z) | 3.3 | 1.3 | 1.3 |
| SafeTune's top-9 MLP layers inside E's 9 (chance: 2.25) | 1 | 1 | 2 |

Overlap of the top-36 components (Jaccard): English and Hindi 0.33, English and Hinglish 0.36, Hindi and
Hinglish 0.41. The circuit did not reach our faithfulness bar in any language, it barely beats random circuits
in Hindi and Hinglish, and the two tools do not agree beyond chance. We kept the CircuitKIT ranking for E,
report these numbers as they are, and let E against R on held-out data decide whether the location matters.

## Published artifacts

| | |
|---|---|
| Models ([collection](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd)) | [E, circuit blocks](https://huggingface.co/maadhi/onegap-tiny-aya-global-circuit-lora) · [D, every layer](https://huggingface.co/maadhi/onegap-tiny-aya-global-plain-lora) · [R, random blocks](https://huggingface.co/maadhi/onegap-tiny-aya-global-random-lora) |
| Data ([collection](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a)) | [transcript labels, manifests, training logs, report](https://huggingface.co/datasets/maadhi/onegap-transcripts-and-logs) · [refusal circuits](https://huggingface.co/datasets/maadhi/onegap-circuits) |
| Write-up | [`docs/article.md`](docs/article.md), abstract in [`docs/abstract.md`](docs/abstract.md) |

To recompute every table from the published labels:

```bash
hf download maadhi/onegap-transcripts-and-logs --repo-type dataset --local-dir pub
uv run onegap report pub/runs/20261112T102654-run-99f85c1b13a7 pub/runs/20261112T094810-run-0887f455fe6c \
    pub/runs/20261112T102710-run-99f85c1b13a7 pub/runs/20261112T094834-run-0887f455fe6c \
    pub/runs/20261112T094846-run-0887f455fe6c pub/runs/20261112T094854-run-0887f455fe6c --name check
```

## What is in this repo

```
configs/        data sizes, decoding, judges, circuit settings, the three training configs (D, E, R)
prompts/        system prompt A (all conditions), prompt B (prompt-only defence), judge rubric
src/onegap/     data building, multi-turn runner, judges, metrics, statistics, circuits, training, report
scripts/        HF Jobs launcher and helpers
tests/          unit tests for everything that runs without a GPU
docs/           PROGRESS.md (milestones and decisions), RUNS.md (every run, failed ones included),
                API_NOTES.md (what the toolkit and dataset APIs really look like), article.md, abstract.md
```

Conditions: **A** base model; **S0** final request alone; **B** stronger system prompt; **G** Llama Guard 3 1B
screening every user turn; **C** SafeTune steering; **D** LoRA DPO on every layer; **E** LoRA DPO on the
circuit's blocks; **R** LoRA DPO on randomly chosen blocks, same count as E. G passed a smoke run only and C
has not been run, so neither is in the tables.

## Reproduce

### 1. Recompute the tables from run folders (no GPU, about a minute)

```bash
uv sync --extra local --extra dev
uv run pytest                      # unit tests
uv run onegap report runs/<A run> runs/<D run> runs/<E run> runs/<R run> --name dev_v1
```

`onegap report` reads each run's `run_manifest.json` and `transcripts.jsonl`, refuses to call runs comparable
if the data file, decoding, judge, model revision or system prompt differ, and writes
`runs/report/dev_v1.{json,md}`. Every number in the article and the cards is pasted from that output.

### 2. Rerun an evaluation (one HF Job per condition, L4, about 34 minutes, about $0.45)

All GPU work runs as Hugging Face Jobs; nothing loads a model locally. You need `HF_TOKEN`, access to the gated
models (`CohereLabs/tiny-aya-global`, `meta-llama/Llama-Guard-3-8B`, `meta-llama/Llama-Guard-3-1B`) and a private
dataset repo for job inputs and outputs (`ONEGAP_HF_WORK_REPO`).

```bash
uv run onegap check-env
uv run --extra local python scripts/launch_job.py run --flavor l4x1 --venv vllm \
    --inputs data/generated/conversations.jsonl -- --condition A --split dev --limit 4      # smoke first
uv run --extra local python scripts/launch_job.py run --flavor l4x1 --venv vllm --timeout 3h \
    --inputs data/generated/conversations.jsonl -- --condition E --split dev --adapter-run <train run id>
uv run python scripts/pull_run.py --light <run id>
```

Decoding is greedy with `max_new_tokens=384`; the model revision is pinned in `configs/eval.yaml`.

### 3. Rebuild everything

| Step | Command | Where |
|---|---|---|
| Sample seeds, map categories | `onegap data` | local, Cohere API |
| Hindi and Bengali | `onegap translate` | HF Job, L4 |
| Hinglish and its back-translation | `onegap codemix`, `onegap codemix-back`, `onegap score-hinglish-back` | local Cohere, CPU job |
| Benign conversations | `onegap benign` | local, Cohere API |
| Fidelity gate, records, split check | `onegap curate`, `onegap check-splits <file>` | local |
| Freeze the test set | `onegap freeze`, then tag `test-freeze-v1` | local, after human review |
| Replay train conversations through A | `onegap run --condition A --split train` | HF Jobs, L4 |
| Correction pairs | `onegap prefdata` | local, Cohere API |
| Circuits and cross-check | `onegap circuits`, `onegap circuit-check` | HF Jobs, L40S and L4 |
| Train D, E, R | `make train-D`, `train-E`, `train-R` | HF Jobs, L40S, 46 to 64 minutes each |
| Evaluate and report | `onegap run`, `onegap report` | HF Jobs, then local |

The run log with flavors, minutes and cost for every job is [`docs/RUNS.md`](docs/RUNS.md). Total spend so far
is about $17 of Hugging Face credit.

Section numbers in code comments ("spec 8.3") refer to our internal implementation plan; what matters from it
is in this README and in `docs/PROGRESS.md`. The project was built with Claude Code as a coding assistant, and
every run, including the failed ones, is in the run log.

**What will not reproduce exactly.** Rerunning a condition gives the same rates within noise, not the same
text: see "The harness has noise of its own" above. Compare runs with `onegap report`, which pairs conditions
by behaviour and reports intervals, and do not expect a rerun to match a published hash.

## Rules we hold ourselves to

1. Training attacks come only from XGuard-Train; test attacks only from SafeMTData `Attack_600`, a different
   attack family. Bengali never appears in training data.
2. The frozen test set is hash-checked before every evaluation and never edited after the freeze.
3. Every decision is made on dev. The locked test run happens once.
4. D, E and R share data, pair count, epochs, steps, learning rate, beta and seed. Only the trained modules differ.
5. All conditions see the same user turns, system prompt (except B), decoding and judges.
6. The guard model (Llama Guard 3 1B) is never the judge (Llama Guard 3 8B).
7. Everything published is un-gated. Unsafe model replies are replaced by a marker with the SHA-256 of the
   original text, the judge labels, the category and the turn.
8. Every safety number is reported next to an over-refusal number.

The rule for reading D, E and R was written down before any of the three had a result
([`docs/PROGRESS.md`](docs/PROGRESS.md), "M7 — Decision rule").

## Toolkits

| Toolkit | Version | Used for |
|---|---|---|
| CircuitKIT | 0.2.2 | Custom contrastive refusal task per language, EAP-IG discovery, faithfulness (patching, ablation, random-circuit baseline), component ranking for E |
| SafeTune | 0.1.14 | Independent cross-check of the circuit (`safety_circuit_info`); refusal-direction steering for condition C (API verified in a smoke job, not run yet) |
| AuditKIT | 1.1.7 | Installed and API-checked; the scoring bridge is not written yet |
| CuratorKIT | 0.1.24 | Installed and API-checked; not yet used on the correction pairs |

What did not work is in [`docs/RUNS.md`](docs/RUNS.md): API mismatches, out-of-memory runs, a judge that missed
a Bengali scam plan until we changed the translator, a refusal detector with false positives in Hindi, and
correction pairs we had to throw out.

## Licences

Code: Apache-2.0 (see `LICENSE`). Adapters are derivatives of Tiny Aya Global and carry CC-BY-NC 4.0. Model
outputs in the transcripts come from Tiny Aya Global (CC-BY-NC 4.0). Sources: SafeMTData `Attack_600` (MIT),
XSTest (CC-BY-4.0), XGuard-Train (gated, no licence stated: we publish IDs and labels for it, never its text),
NLLB-200 (CC-BY-NC 4.0), Llama Guard 3 (Llama 3.1 and 3.2 Community Licences), Lexsi toolkits (LSAL).

## Team

Listed in the article (section 9). Built during the Lexsi AI Safety and Alignment Hackathon, 10 to 15 Nov 2026.
