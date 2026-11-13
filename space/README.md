---
title: OneGap replay
emoji: 🗣️
colorFrom: red
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Turn-by-turn safety labels for Tiny Aya in 4 languages
datasets:
  - maadhi/onegap-transcripts-and-logs
models:
  - CohereLabs/tiny-aya-global
---

Replay of the published OneGap transcripts: per-turn judge labels for Tiny Aya Global under the base model, a
safety prompt and three LoRA repairs, in English, Hindi, Hinglish and Bengali. Read-only, CPU only; nothing is
generated live. Code: https://github.com/Maadhi21/onegap
