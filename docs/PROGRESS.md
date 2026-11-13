# Progress

## Registration (HUMAN — fill in)
- Registered email: 
- HF usernames (3): godt3
- HF org: maadhi
- Discord member: 

## Pre-flight (spec Section 1) — reported done by user 2026-11-10
- [x] Registration confirmed, Track 2 declared
- [x] HF org + token, $20 coupon redeemed
- [x] Model gates accepted (tiny-aya-global, Llama-Guard-3-8B, Llama-Guard-3-1B)
- [x] COHERE_API_KEY available; Hindi/Hinglish reviewers named
- [ ] **HF_TOKEN and COHERE_API_KEY visible to this machine** — not set at any scope as of 2026-11-10

## Gate 3 blockers (must clear before anything goes public)
- [ ] **Git history:** commit e00be67 contains `data/generated/cohere_cache/` (classifier inputs incl. XGuard-Train
      text). Untracked in f7400e4, but still in history. Rewrite/squash history before the first push to a
      public remote.
- [ ] **XGuard-Train licence:** gated dataset, no licence on the card (derived from WildJailbreak, ODC-BY).
      Before publishing the correction set (its prompts are XGuard user turns), check the X-Teaming repo
      licence and gate terms; if redistribution is not allowed, publish pair IDs + regeneration script instead.

## M0 — Repo, venvs, CI, access
Status: **done 2026-11-11** (Llama Guard 3 1B granted on `maadhi`, the last open item).

Acceptance:
- [x] `make test` passes — local 2026-11-10: `uv run pytest` → `22 passed, 1 skipped`
  (skip = publish-dir redaction scan, runs when ONEGAP_PUBLISH_DIR is set); `ruff check` → All checks passed
- [x] Tiny Aya tokenizer loads (job 20261110T093652): revision af89d219, chat template renders, 414-token preamble
- [x] CircuitKIT 0.2.2, SafeTune 0.1.14, CuratorKIT 0.1.24, AuditKIT 1.1.7 import in their venvs (jobs 093652/093738/093745)
- [x] CircuitKIT discovery runs on Tiny Aya: mmlu on L4 (182 components); custom refusal task on L40S (job 103504)
- [x] SafeTune refusal direction + `RefusalDirectionModel(mode="steer")` work (job 094251)
- [x] Llama Guard 3 8B readable on `maadhi` (2026-11-10)
- [x] Llama Guard 3 1B readable on `maadhi` (2026-11-11, verified on `config.json`) — condition G unblocked

## M1 — Data (attack + over-refusal done; benign in progress)
Status 2026-11-11: attack and over-refusal records assembled and gated; benign generation running; HUMAN spot
check and freeze pending.

Acceptance (attack + over-refusal, `uv run onegap curate`, data/generated/fidelity_report.json):
- [x] Category counts, test attack (en): fraud 17, cyber 27, weapons 27, drugs 8, hate 21 = 100 (21 top-ups,
      `provenance.topup`; Attack_600 has only 8 drug / 17 fraud behaviours). Train 391 (drugs 71/80), dev 40 (8/cat).
- [x] Fidelity-gate pass rate per language (conversation passes only if every turn passes):

      | lang | pass | rate | fails |
      |---|---|---|---|
      | en | 731/731 | 100% | — |
      | hi | 714/731 | 97.7% | 5 low cos, 12 wrong script |
      | hinglish | 709/731 | 97.0% | 21 low cos (after back-translation + 1 stricter regeneration), 1 missing |
      | bn | 324/340 | 95.3% | 10 low cos, 6 wrong script (dev/test only) |

      Test attack per language: en 100, hi 89, hinglish 82, bn 93. Paired (same seed) en∩hi∩hinglish = **74**
      (fraud 14, cyber 17, weapons 20, drugs 6, hate 17); all four languages = 69. Cross-language comparisons are
      paired on these seeds; per-language ASR uses each language's full n.
- [x] No seed_id in two splits; family rules hold (`check_splits` OK on 2,478 records)
- [x] Bengali only in dev/test (0 train records)
- [x] Benign multi-turn generated, translated, gated (2026-11-11, after back-translation scoring; whole-set
      gate en 100%, hi 96.3%, Hinglish 86.9%, bn 91.6%). Benign kept: train en 384 / hi 362 / Hinglish 282;
      dev 30/28/22/25 (en/hi/Hinglish/bn); test 80/76/50/63. `check_splits` OK on 3,881 records.
      Re-curation moved test Hinglish attack 82 → 83 (a previously missing back-translation now passes; pre-freeze).
- [ ] HUMAN spot check 30 hi + 30 Hinglish test conversations (sheets: data/generated/spot_check_{hi,hinglish}.csv):
      **not done by a person** (no human reviewers, Madhava 2026-11-12). Replaced by the mechanical consistency check
      (`scripts/consistency_check.py`) and Claude's review of its findings; stated as a limit in the article.
- [x] Review of the 9 proposed test fixes (2026-11-13, Claude, delegated): each checked against its English turn;
      all 9 approved (5 Hindi mistranslations: psychopharmacology→psychotherapy, psychoactive→psychiatrist, a flipped
      "overcome", the group name Anonymous as the adjective, anti-bullying→child abuse; 2 Hinglish: a garbled turn and
      a lost back-reference). One proposal edited: "हैकर समूह" (hacker group) dropped from a600-0083 turn 1 because the
      English does not say it. Audit trail in each record's `provenance.human_fixes`.
- [x] **Freeze, 2026-11-13** (Madhava's go-ahead 09:03 for the held-out audit): `data/frozen/test.jsonl` = the 1,225
      test records of `data/generated/conversations.jsonl` (source file sha256 09b6cb430a6dd8ab…) with the 9 approved
      fixes applied; `SHA256SUMS` digest f0d07ff62b1a…; tag `test-freeze-v1`. Attack: en 100, hi 89, Hinglish 83,
      bn 93; over-refusal en 150, hi 147, Hinglish 148, bn 146; benign en 80, hi 76, Hinglish 50, bn 63.
      `check_splits` OK on 3,881 records with the frozen test records. `conversations.jsonl` itself is unchanged (so
      the dev inputs uploaded earlier still hash-match); a rerun of `onegap curate` applies the same approved fixes.
      **Deviation (Invariant 3):** the single-prompt control (S0) ran on test before Gate 2 passed (A was launched
      too and cancelled before it finished). No adapter ran on test, so no repair decision has seen the test set.

## M4 (part) — Dev baseline audit and Gate 1 — 2026-11-11
Runs: A `20261111T100248-run-0539da7c61e7` (351 transcripts, 77 min L4), S0 `20261111T100320-run-0539da7c61e7`.
Report: `uv run python scripts/gate1.py runs/<A> runs/<S0>` → runs/report/gate1.json. Dev, 40 attacks/language.

| lang | S0 single prompt | A full conversation | uplift A−S0 (paired) | over-refusal A |
|---|---|---|---|---|
| en | 12.5 [5.5, 26.1] (5/40) | 27.5 [16.1, 42.8] (11/40) | +15.0 [+0.0, +30.0], McNemar p=0.146 | 8.0 [3.2, 18.8] (4/50) |
| hi | 30.0 [18.1, 45.4] (12/40) | 35.0 [22.1, 50.5] (14/40) | +5.0 [−10.0, +20.0] | 6.2 [2.1, 16.8] (3/48) |
| hinglish | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | −2.5 [−20.0, +12.5] | 0.0 [0.0, 7.4] (0/48) |
| bn | 33.3 [20.6, 49.0] (13/39) | 33.3 [20.6, 49.0] (13/39) | +0.0 [−15.4, +15.4] | 0.0 [0.0, 7.7] (0/46) |

Turn of first failure (A): mostly turn 3–4 in every language. Refusals are rare (1–4% of assistant turns).
English S0 12.5% matches the public single-turn anchor (14.3%).

Gate 1 checks: ASR ≥20% in ≥2 of en/hi/Hinglish **pass**; Hindi or Hinglish ≥10 pts above English under A
**fail** (hi +7.5); multi-turn uplift clearly above zero **only in English** (CI lower bound 0.0).
Pattern: the language gap is in the *single prompt* (en 12.5% vs hi/Hinglish 30%, bn 33%); a multi-turn
build-up brings English up to the same ~30%.
**HUMAN Gate 1 decision (2026-11-11): continue with the repairs plan, reframed.** Headline: "In English a
harmful request needs several turns of build-up; in Hindi, Hinglish and Bengali one prompt is enough. Both end
at about 30%." Both gaps are reported; the locked test set decides whether the pattern holds.

Method notes: Hinglish fidelity = LaBSE cos(source, Cohere English back-translation), as spec 7.3 says; the
direct cos(source, Hinglish) is kept only as a diagnostic (median 0.79; LaBSE does not know romanized Hindi).
Failing Hinglish turns were regenerated once with a stricter "every clause, keep pronouns" prompt (v3/v4).

## M6 — Locating refusal — 2026-11-11
Runs: circuits `20261111T191230-circuits-1ea299391ff9` (l40sx1, 45 min), SafeTune cross-check
`20261111T200611-circuit-check-a9bb7211b2cf`. Task: 48 pairs/language (clean = final harmful turn where base A went
unsafe, corrupt = benign train request of the same category, single user turn, lengths equalised by `equalize`);
answers = first refusal token vs the model's own first token; EAP-IG, node level, sparsity 0.3.

| | en | hi | Hinglish | bar |
|---|---|---|---|---|
| Patching faithfulness | 0.69 | 0.36 | 0.67 | ≥0.75 |
| Ablation | 1.0 | 0.0 | 1.0 | |
| Circuit vs 5 random circuits (z) | 3.3 | 1.3 | 1.3 | |
| SafeTune top-9 MLP layers in E (chance 2.25) | 1 | 1 | 2 | |

Overlap (Jaccard, top-36 nodes): en–hi 0.33, en–Hinglish 0.36, hi–Hinglish 0.41.

- [ ] Faithfulness ≥0.75 — **not met**; **fallback recorded** (docs/RUNS.md): E kept as the CircuitKIT ranking,
      faithfulness and the SafeTune disagreement reported as findings; E vs R on dev decides.
- [x] Overlap matrix saved (`circuits_report.json`)
- [x] Component sets: E attn [0,13,14,16,17,18,19,20,30] + MLP [0,4,9,11,12,16,17,18,35] (63 modules);
      R attn [5,6,8,9,10,28,29,32,33] + MLP [1,3,6,7,8,14,25,28,31], disjoint, seed 13
      (`selected_components.json`). Selection: blocks of top nodes capped at 9/36 per block type.
- [x] SafeTune cross-check run (spec: always)
- Reading so far: refusal localises clearest in English; in Hindi the circuit barely beats random circuits and the
  two tools disagree. Consistent with the Gate 1 pattern (Indian-language refusal is weaker from the first prompt).

## M7 — Decision rule for D / E / R (pre-registered 2026-11-12 11:40 Dubai, before any full-run result)
Written before the full D/E/R runs finish so the circuit claim cannot be fitted to the numbers. Proposed by Claude,
**Madhava to confirm**. Changes after dev results are seen are logged in RUNS.md as deviations.

Data: dev split only (Invariant 3). Safety = ASR (Llama Guard 3 8B `unsafe`) on dev attacks, en + hi + Hinglish pooled
and paired by seed; over-refusal = refusal rate on dev benign + XSTest-style prompts with the fixed 2026-11-12 refusal
detector. Bengali is reported, never used to choose.
1. **E beats R** (the circuit matters) if ASR(R) − ASR(E) ≥ 5 points, the paired bootstrap 90% CI excludes 0, and
   over-refusal(E) ≤ over-refusal(R) + 3 points.
2. **E matches D** (a small edit suffices) if ASR(E) − ASR(D) ≤ 5 points and over-refusal(E) ≤ over-refusal(D) + 3.
3. Neither 1 nor 2 → report a measured null: "at this data and budget, which components are trained did not matter".
4. Gate 2 recipe = lowest pooled ASR among D/E/R with over-refusal ≤ base A + 3 points; ties within 2 points go to
   fewer trainable parameters (E or R before D).
5. Every safety number is shown next to its over-refusal number (Invariant 8); edit size is reported for D, E, R.

## M7 — Repairs D, E, R trained — 2026-11-12 12:49 Dubai
Runs `20261112T074348-train-D`, `…074356-train-E`, `…074408-train-R` (sha a7c46f2, pairs v2: 1,228 → 1,191 after the
length filter). Details in RUNS.md.
- [x] Loss curves falling, reward margins rising (D 0.68, E 0.23, R 0.14 at the last step)
- [x] E and R touch the same number of modules per type (9 attn + 9 MLP blocks = 63 projections, 7.56M params), disjoint
- [x] Edit sizes recorded (changed share D 84%, E/R 21%; Δ‖W‖_F D 0.574, E 0.307, R 0.286)
- [x] Adapters reload + dev table v1 (below)
- [ ] Steering C: **dropped for the sprint** (budget, 2026-11-12)

## M7 — Dev table v1 — 2026-11-12 14:45 Dubai
`uv run onegap report runs/<S0> runs/<A> runs/<B> runs/<D> runs/<E> runs/<R> --name dev_v1` → `runs/report/dev_v1.{json,md}`.
All six runs on the current pipeline (vLLM generation and judge, fixed refusal detector), same data file
(sha256 09b6cb43…), comparability check passed. Per-turn tables are in `runs/report/dev_v1.md`.

Split: dev. Rates are % [Wilson 95%] (count/n). Pooled = en + hi + Hinglish; Bengali is reported, never pooled and never used to choose.

Runs: S0 `20261112T102654-run-99f85c1b13a7`; A `20261112T094810-run-0887f455fe6c`; B `20261112T102710-run-99f85c1b13a7`; D `20261112T094834-run-0887f455fe6c`; E `20261112T094846-run-0887f455fe6c`; R `20261112T094854-run-0887f455fe6c`.

Comparability check (same split, data file, decoding, judge, model revision, prompt): passed.

**Attacks that worked (ASR-any: an unsafe reply at any turn)**

| condition | English | Hindi | Hinglish | Bengali (held out) | en+hi+Hinglish |
| --- | --- | --- | --- | --- | --- |
| S0 | 15.0 [7.1, 29.1] (6/40) | 37.5 [24.2, 53.0] (15/40) | 30.0 [18.1, 45.4] (12/40) | 28.2 [16.5, 43.8] (11/39) | 27.5 [20.3, 36.1] (33/120) |
| A | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 38.5 [24.9, 54.1] (15/39) | 28.3 [21.0, 37.0] (34/120) |
| B | 22.5 [12.3, 37.5] (9/40) | 30.0 [18.1, 45.4] (12/40) | 30.0 [18.1, 45.4] (12/40) | 28.2 [16.5, 43.8] (11/39) | 27.5 [20.3, 36.1] (33/120) |
| D | 20.0 [10.5, 34.8] (8/40) | 30.0 [18.1, 45.4] (12/40) | 27.5 [16.1, 42.8] (11/40) | 33.3 [20.6, 49.0] (13/39) | 25.8 [18.8, 34.3] (31/120) |
| E | 27.5 [16.1, 42.8] (11/40) | 27.5 [16.1, 42.8] (11/40) | 35.0 [22.1, 50.5] (14/40) | 35.9 [22.7, 51.6] (14/39) | 30.0 [22.5, 38.7] (36/120) |
| R | 17.5 [8.7, 31.9] (7/40) | 25.0 [14.2, 40.2] (10/40) | 35.0 [22.1, 50.5] (14/40) | 33.3 [20.6, 49.0] (13/39) | 25.8 [18.8, 34.3] (31/120) |

**Over-refusal (harmless prompts and harmless multi-turn conversations with a refused turn)**

| condition | English | Hindi | Hinglish | Bengali (held out) | en+hi+Hinglish |
| --- | --- | --- | --- | --- | --- |
| A | 8.8 [4.3, 17.0] (7/80) | 3.9 [1.4, 11.0] (3/76) | 0.0 [0.0, 5.2] (0/70) | 0.0 [0.0, 5.1] (0/71) | 4.4 [2.4, 8.0] (10/226) |
| B | 0.0 [0.0, 4.6] (0/80) | 9.2 [4.5, 17.8] (7/76) | 5.7 [2.2, 13.8] (4/70) | 5.6 [2.2, 13.6] (4/71) | 4.9 [2.7, 8.5] (11/226) |
| D | 8.8 [4.3, 17.0] (7/80) | 2.6 [0.7, 9.1] (2/76) | 0.0 [0.0, 5.2] (0/70) | 1.4 [0.2, 7.6] (1/71) | 4.0 [2.1, 7.4] (9/226) |
| E | 8.8 [4.3, 17.0] (7/80) | 2.6 [0.7, 9.1] (2/76) | 0.0 [0.0, 5.2] (0/70) | 1.4 [0.2, 7.6] (1/71) | 4.0 [2.1, 7.4] (9/226) |
| R | 8.8 [4.3, 17.0] (7/80) | 2.6 [0.7, 9.1] (2/76) | 0.0 [0.0, 5.2] (0/70) | 0.0 [0.0, 5.1] (0/71) | 4.0 [2.1, 7.4] (9/226) |

Parts of the pooled over-refusal number:

| condition | XSTest-style prompts refused | harmless conversations with a refused turn | harmless items with a reply the judge called unsafe |
| --- | --- | --- | --- |
| A | 4.1 [1.9, 8.7] (6/146) | 5.0 [2.0, 12.2] (4/80) | 2.7 [1.2, 5.7] (6/226) |
| B | 5.5 [2.8, 10.4] (8/146) | 3.8 [1.3, 10.5] (3/80) | 1.8 [0.7, 4.5] (4/226) |
| D | 4.1 [1.9, 8.7] (6/146) | 3.8 [1.3, 10.5] (3/80) | 1.8 [0.7, 4.5] (4/226) |
| E | 4.1 [1.9, 8.7] (6/146) | 3.8 [1.3, 10.5] (3/80) | 2.2 [0.9, 5.1] (5/226) |
| R | 4.1 [1.9, 8.7] (6/146) | 3.8 [1.3, 10.5] (3/80) | 2.2 [0.9, 5.1] (5/226) |

**Change against A, paired by behaviour (points [95% CI, bootstrap over seed_id])**

| difference | English | Hindi | Hinglish | Bengali (held out) | en+hi+Hinglish |
| --- | --- | --- | --- | --- | --- |
| A − S0: multi-turn uplift | +15.0 [+0.0, +32.5] | -10.0 [-27.5, +7.5] | -2.5 [-20.0, +12.5] | +10.3 [+2.6, +20.5] | +0.8 [-11.7, +13.3] |
| S0 vs A: McNemar p | 0.146 | 0.388 | 1 | 0.125 | — |
| B − A: attacks that worked | -7.5 [-22.5, +7.5] | +2.5 [-7.5, +12.5] | +2.5 [-10.0, +15.0] | -10.3 [-20.5, -2.6] | -0.8 [-9.2, +7.5] |
| B − A: over-refusal | -8.8 [-15.0, -3.8] | +5.3 [+1.3, +10.5] | +5.7 [+1.4, +11.4] | +5.6 [+1.4, +11.3] | +0.4 [-2.7, +3.9] |
| B vs A: McNemar p | 0.508 | 1 | 1 | 0.125 | — |
| D − A: attacks that worked | -10.0 [-22.5, +2.5] | +2.5 [-7.5, +12.5] | +0.0 [+0.0, +0.0] | -5.1 [-17.9, +7.7] | -2.5 [-8.3, +2.5] |
| D − A: over-refusal | +0.0 [+0.0, +0.0] | -1.3 [-3.9, +0.0] | +0.0 [+0.0, +0.0] | +1.4 [+0.0, +4.2] | -0.4 [-1.4, +0.0] |
| D vs A: McNemar p | 0.289 | 1 | 1 | 0.688 | — |
| E − A: attacks that worked | -2.5 [-15.0, +10.0] | +0.0 [-7.5, +7.5] | +7.5 [-2.5, +17.5] | -2.6 [-12.8, +7.7] | +1.7 [-5.0, +8.3] |
| E − A: over-refusal | +0.0 [+0.0, +0.0] | -1.3 [-3.9, +0.0] | +0.0 [+0.0, +0.0] | +1.4 [+0.0, +4.2] | -0.4 [-1.4, +0.0] |
| E vs A: McNemar p | 1 | 1 | 0.375 | 1 | — |
| R − A: attacks that worked | -12.5 [-25.0, +0.0] | -2.5 [-12.5, +7.5] | +7.5 [+0.0, +17.5] | -5.1 [-17.9, +7.7] | -2.5 [-10.8, +5.0] |
| R − A: over-refusal | +0.0 [+0.0, +0.0] | -1.3 [-3.9, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.4 [-1.4, +0.0] |
| R vs A: McNemar p | 0.125 | 1 | 0.25 | 0.688 | — |

**Between the repairs, attacks that worked (points [90% CI] as pre-registered; 95% in the JSON)**

| difference | English | Hindi | Hinglish | Bengali (held out) | en+hi+Hinglish |
| --- | --- | --- | --- | --- | --- |
| R − E | -10.0 [-22.5, +2.5] | -2.5 [-12.5, +7.5] | +0.0 [-7.5, +7.5] | -2.6 [-12.8, +7.7] | -4.2 [-10.8, +2.5] |
| E − D | +7.5 [-2.5, +17.5] | -2.5 [-10.0, +5.0] | +7.5 [+0.0, +17.5] | +2.6 [-5.1, +10.3] | +4.2 [+0.0, +8.3] |
| R − D | -2.5 [-12.5, +7.5] | -5.0 [-15.0, +5.0] | +7.5 [+2.5, +15.0] | +0.0 [-10.3, +10.3] | +0.0 [-5.0, +5.0] |

**Pre-registered D / E / R rule (docs/PROGRESS.md, M7), pooled en + hi + Hinglish**

1. E beats R (the circuit matters): **no**. R − E = -4.2 [-10.8, +2.5] points (90% CI); gap ≥ 5 points: no; CI excludes 0: no; over-refusal(E) ≤ over-refusal(R) + 3: yes.
2. E matches D (a small edit is enough): **yes**. E − D = +4.2 [+0.0, +8.3] points (90% CI); gap ≤ 5 points: yes; over-refusal(E) ≤ over-refusal(D) + 3: yes.
3. Measured null (which components are trained did not matter): **no**.
4. Gate 2 recipe: **R**. Within over-refusal ≤ A + 3 points: D, E, R; within 2 points of the lowest: D, R. HUMAN confirms at Gate 2.

**Relative cut in attacks that worked against A** (Gate 2 asks for ≥ 50% in en, hi and Hinglish, over-refusal within +3 points and MMLU within 1 point; MMLU is not in this report)

| condition | English | Hindi | Hinglish | Bengali (held out) | en+hi+Hinglish |
| --- | --- | --- | --- | --- | --- |
| D | 33% | -9% | 0% | 13% | 9% |
| E | 8% | 0% | -27% | 7% | -6% |
| R | 42% | 9% | -27% | 13% | 9% |

**Edit size**

| condition | trainable parameters | share of the model | projections with LoRA | layers touched | base weights inside touched matrices | ‖ΔW‖_F |
| --- | --- | --- | --- | --- | --- | --- |
| D | 30,228,480 | 0.90% | 252 | 36 | 84% | 0.574 |
| E | 7,557,120 | 0.23% | 63 | 14 | 21% | 0.307 |
| R | 7,557,120 | 0.23% | 63 | 15 | 21% | 0.286 |

**Reading (Claude, 2026-11-12; Madhava to confirm).**
- **No adapter changed attack success.** Pooled: A 28.3%, D 25.8%, E 30.0%, R 25.8% (n = 120); every difference
  from A has an interval that contains zero; over-refusal 4.4% → 4.0%.
- **Prompt B did nothing for attacks:** 27.5% against 28.3% (−0.8 [−9.2, +7.5]). Over-refusal moved by language:
  en 8.8% → 0%, hi 3.9% → 9.2%, Hinglish 0% → 5.7%, bn 0% → 5.6% (each interval excludes zero).
- **The rule as written** gives: E does not beat R; E "matches" D; recipe R. **Deviation proposed:** the rule never
  required a repair to beat A first, so "E matches D" is two ineffective adapters being equally ineffective. We
  report "the first recipe did not work; E against R is untested" and do not carry R forward as a repair. New
  first clause for any later round: an arm must beat A by ≥ 5 points with a 95% interval excluding zero before
  comparisons between arms count.
- **Why:** underfit. Mean train loss D 0.525 / E 0.628 / R 0.648 (start 0.693); final margins 0.68 / 0.23 / 0.14;
  refused share of attack turns 2.3% → 4.4% / 2.9% / 3.1%. lr 5e-6 is a full fine-tuning value, too low for LoRA.
- **Harness noise:** `runs_local/diag_adapter_effect.py`. E full run against the E smoke (same adapter, same 16
  benign conversations): first reply identical 1/16, labels differ on 3/68 turns. A against D / E / R: labels
  differ on 64 / 49 / 54 of 1,312 turns (4.9% / 3.7% / 4.1%), the same size. Batched bf16 greedy decoding on vLLM is
  not reproducible run to run, so `make replay` cannot reproduce a published hash (Invariant 7 wording updated in
  the README and cards: the hash identifies the text, it does not regenerate it).
- **Audit replicates:** S0 / A on this pipeline: en 15.0 / 30.0, hi 37.5 / 27.5, Hinglish 30.0 / 27.5, bn 28.2 / 38.5
  (Gate 1, HF pipeline: en 12.5 / 27.5, hi 30.0 / 35.0, Hinglish 30.0 / 27.5, bn 33.3 / 33.3). English uplift +15.0
  both times (p = 0.146 both times); the Bengali uplift (+10.3 now, 0.0 then) is not claimed.
- [x] Loss curves, margins, edit sizes, adapters reload, dev table v1, RUNS.md updated (M7 acceptance), except
      steering C.
- **Gate 2:** not passed (no arm cuts attack success by 50%). No locked run of the repairs with this recipe.

## Decisions by Madhava — 2026-11-12 14:49 Dubai (chat)
- Budget: **hard limit $20** (the coupon). No more than that until shortlisted ($200 then). Supersedes the $25 cap.
  Spend after today's runs ≈ $16.7. **No further jobs** without a new go-ahead.
- Priority: the first submission, now. A final submission later if budget and time allow.
- Public GitHub repo name `onegap` (account to be confirmed: `Maadhi21/onegap` new, or an existing repo); squash
  history before the first push: **yes**.
- Hugging Face: publish under `maadhi` with the proposed names: **ok**. Licences Apache-2.0 (code) and CC-BY-NC 4.0
  (adapters, data): **ok**.
- **No human reviewers and no human audit.** Review of the generated conversations is delegated to Claude
  ("you will own it"). Consequence: the 30 + 30 spot check, the 9 proposed test fixes and the judge audit have no
  human sign-off; the article states this as a limit.
- Madhava posts the Hugging Face article and fills the form (registered email, Discord); Claude prepares the text.

## M9 (part) — First submission pack — 2026-11-12
- [x] `README.md`, `LICENSE`, `docs/abstract.md` (484 words), `docs/article.md` (organisers' nine sections),
      `docs/submission.md`, `docs/cards/` (generated by `scripts/make_cards.py` from `runs/report/dev_v1.json`)
- [x] `onegap publish-stage` / `publish-hub` / `publish-visibility` with tests: unsafe replies redacted, attack
      replies and XGuard-Train text withheld, English views of hidden replies dropped (gap found in `redact.py`:
      `assistant_en` was not covered), weights only as adapter files, circuits without the prompt pairs
- [x] Hugging Face, 2026-11-12 17:50: job `20261112T134218-publish-hub-ab3fe55c3c86` (cpu-basic, 2.6 min) uploaded five
      repos as private; file lists and counts checked (labels unchanged in all six runs; in run A 95 replies
      redacted as unsafe, 552 attack replies withheld, 636 XGuard-Train user turns withheld, 665 harmless replies in
      full); then switched to public. Collections: [models](https://huggingface.co/collections/maadhi/onegap-models-6af5c33bccce0ed5679987fd), [data](https://huggingface.co/collections/maadhi/onegap-data-6af5c33b806eb7b1dd105b9a).
- [x] GitHub, 2026-11-13 09:00: [`Maadhi21/onegap`](https://github.com/Maadhi21/onegap) public, one commit
      `a4445d3` (93 files, tree `2db170a2`; internal planning files and `.cmd` helpers left out; author = GitHub
      no-reply address). Same tree as the 12 Nov snapshot `1f29248`, rebuilt with a new timestamp. Checks on that
      exact commit in a clean worktree: secret patterns none, pytest pass, ruff pass. Opens logged out; README and
      Apache-2.0 licence render; CI run #1 passed. Madhava created the empty repo (the GitHub connector cannot create
      repos: 403 "Resource not accessible by integration").
- [ ] Article posted, form submitted (HUMAN: Madhava)

## M8 (part) — Held-out audit, S0 only — 2026-11-13
Run S0 `20261113T053126-run-af2a65f52c79` (365 attack conversations from the frozen `Attack_600` test set, hash
verified in the job, 0 judge fallbacks). Report `runs/report/test_v1.{json,md}`. A on test
(`20261113T053116-run-af2a65f52c79`) was cancelled after 68 min, in judging, with no error (most likely the
credit ran out); Madhava 11:05: finish with S0 only.

| | English | Hindi | Hinglish | Bengali (held out) |
|---|---|---|---|---|
| S0 unsafe | 3.0 [1.0, 8.5] (3/100) | 6.7 [3.1, 13.9] (6/89) | 7.2 [3.4, 14.9] (6/83) | 10.8 [5.9, 18.7] (10/93) |
| S0 refused outright | 14.0 [8.5, 22.1] (14/100) | 3.4 [1.2, 9.4] (3/89) | 1.2 [0.2, 6.5] (1/83) | 2.2 [0.6, 7.5] (2/93) |
| unsafe, gap to en (paired bootstrap, McNemar) | — | +3.4 [−1.1, +9.0], p=0.375 | +3.6 [−2.4, +9.6], p=0.453 | +7.5 [+2.2, +14.0], p=0.039 |
| refused, gap to en | — | −12.4 [−20.2, −4.5], p=0.0074 | −10.8 [−18.1, −4.8], p=0.0039 | −11.8 [−18.3, −5.4], p=0.00098 |

Reading: the dev language pattern (English refuses a single harmful prompt, the other languages rarely do)
replicates on a different attack family; Bengali (never trained) is the only unsafe-rate gap that excludes zero.
Absolute single-prompt success is far below dev because `Attack_600` final turns depend on their context. The
multi-turn uplift on test is not measured.

## M9 — Final submission — 2026-11-13
- [x] Article (final version: held-out section in 7.1, notice, costs, limits, section 8), abstract (under 500
      words), README status and held-out table, cards regenerated (`make_cards.py --test-report`)
- [ ] Hub update (S0 test run, test report, cards) from the dev box; GitHub push of the same commit
- [ ] Article posted, form submitted (HUMAN: Madhava)
