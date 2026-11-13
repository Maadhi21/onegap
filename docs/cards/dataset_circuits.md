---
license: cc-by-nc-4.0
tags:
- interpretability
- circuits
- safety
- circuitkit
- safetune
- lexsi-hackathon-2026
pretty_name: OneGap refusal circuits
---

# OneGap: refusal circuits in Tiny Aya Global, per language

Where refusal sits in `CohereLabs/tiny-aya-global` for English, Hindi and Hinglish, as found by CircuitKIT 0.2.2 (EAP-IG, node level) on a contrastive refusal task, with a SafeTune 0.1.14 cross-check. These files decided which blocks the circuit-restricted adapter (E) and its random control (R) train.

## What is here

```
scores_{en,hi,hinglish}.json      node scores per language (attention heads and MLP blocks)
circuits_report.json              faithfulness per language (patching, ablation, random-circuit baseline) and the overlap matrix
selected_components.json          the blocks E trains, the blocks R trains, and the module names
safetune_crosscheck.json          SafeTune's layer ranking against E and R
```

## What it shows

| | English | Hindi | Hinglish |
|---|---|---|---|
| Patching faithfulness (our bar: 0.75) | 0.69 | 0.36 | 0.67 |
| Ablation | 1.0 | 0.0 | 1.0 |
| Circuit against 5 random circuits (z) | 3.3 | 1.3 | 1.3 |
| SafeTune's top-9 MLP layers inside E's 9 (chance: 2.25) | 1 | 1 | 2 |

Overlap of the top-36 nodes (Jaccard): English and Hindi 0.33, English and Hinglish 0.36, Hindi and Hinglish 0.41.

The circuit missed our faithfulness bar in every language, is weakest in Hindi, and the two tools do not agree beyond chance. We publish it as it came out.

## How it was made

48 pairs per language. Clean prompt: a harmful request (the final user turn of a training conversation on which the base model gave an unsafe reply). Corrupted prompt: a harmless request of the same category, padded to the same token length. Metric: logit difference between the first token of a refusal and the model's own first token. EAP-IG, sparsity 0.3, single user turn with the chat template.

The prompt pairs themselves are not published: the harmful side is XGuard-Train text (gated, no licence stated). The code that rebuilds them is in the repository: https://github.com/Maadhi21/onegap.

E: attention blocks 0, 13, 14, 16, 17, 18, 19, 20, 30 and MLP blocks 0, 4, 9, 11, 12, 16, 17, 18, 35. R: attention 5, 6, 8, 9, 10, 28, 29, 32, 33 and MLP 1, 3, 6, 7, 8, 14, 25, 28, 31.

Adapters trained from these sets: [maadhi/onegap-tiny-aya-global-circuit-lora](https://huggingface.co/maadhi/onegap-tiny-aya-global-circuit-lora) and [maadhi/onegap-tiny-aya-global-random-lora](https://huggingface.co/maadhi/onegap-tiny-aya-global-random-lora).

Built for the Lexsi AI Safety and Alignment Hackathon 2026, Track 2.
