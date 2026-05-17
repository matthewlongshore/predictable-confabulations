# Data

## Folder map

```
data/
├── verified/             Per-reference SourceVerify (SVRIS) outputs, one CSV per open-weights model.
├── aggregated/           Cross-model summaries used in the paper (model summary, MTQ, OpenAlex counts, title-match robustness matrix).
├── relevance/            Kimi K2 topic-relevance verdicts.
├── human_validation/     Human authenticity reviews (model column anonymized).
├── SCHEMA.md             Column dictionaries for every CSV in this folder.
└── README.md             (this file)
```

## What is and isn't published

### Tier A — open-weights models (raw outputs included)

31 models with publicly released weights. Their per-reference SVRIS outputs live in `verified/`:

DeepSeek R1 · DeepSeek V3 · DeepSeek V4 Pro · Gemma 3 (4B / 12B / 27B) · Gemma 4 31B · GPT-OSS 120B · Kimi K2 1T-A32B · Llama 3.1 (8B / 70B / 405B base / 405B Hermes) · Llama 3.2 1B · Llama 3.3 70B · Llama 4 Maverick · Llama 4 Scout · MiniMax M2.5 · Mistral Large 2 123B · Mistral Medium 3.1 250B · Mistral Small 3.2 24B · Mixtral 8x22B · Mixtral 8x7B · Qwen3 (8B / 14B / 32B × think / nothink) · Qwen3 30B-A3B · Qwen3.5 397B

### Tier B — closed-API models (aggregate scores only)

Raw model outputs from closed APIs cannot be redistributed under their providers' Terms of Service. These models appear only in `aggregated/model_summary.csv`:

GPT-5 · GPT-5 Mini · GPT-5 Nano · GPT-5.4 (OpenAI) · Claude Opus 4.6 · Claude Sonnet 4.6 (Anthropic) · Grok 3 (xAI)

## Methodology — short version

- **Generation.** Each model was prompted, temperature 0 (where supported), to "List 10 different relevant scholarly references" for each of 24 topics. The full prompt verbatim is in the paper Methods.
- **Verification.** Each reference was submitted to SourceVerify in batches of 5. SVRIS extracts the citation's title, authors, year, venue, and identifier, then searches OpenAlex, Google Scholar, Google Books, and DOI registries for matches. Each field is classified as MATCH, ABBREV, CONTAINS, CONTRADICTION, UNCONFIRMED, or ABSENT.
- **Per-reference authenticity score** ([0, 1], higher = more authentic):
  ```
  score = max(0, Σ(wᵢ × vᵢ) / Σ(wᵢ))    over non-ABSENT fields
  error_distance = 1 − score
  ```
  Weights: Title 0.25, Identifier 0.25, Authors 0.20, Year 0.15, Venue 0.15.
  Verdict values: MATCH +1.0, ABBREV +0.75, CONTAINS +0.5, UNCONFIRMED 0.0, CONTRADICTION −1.0.
- **Topic relevance.** Kimi K2 (temperature 0, OpenRouter) rated each (topic, parsed title) pair YES / PARTIAL / NO. Scored YES=1.0, PARTIAL=0.5, NO=0.0.
- **Quality.** `quality = authenticity × relevance`.
- **Dedup.** Within each (model, topic) cell, normalised titles (lowercased, punctuation- and whitespace-stripped) are collapsed to a single recall event.

## A note on reliability

SVRIS is non-deterministic at the margin: retesting yields r ≈ 0.675, and retest errors are asymmetric (a re-run only ever downgrades, never upgrades). Reported authenticity and quality scores are therefore **lower bounds** on the true recall rate.

The human-validation audit (`human_validation/`) found:
- No false positives in 326 reviews: SVRIS never certified a fabricated reference as real.
- 17 false negatives, all of which had at least one factual defect (wrong venue, year, coauthor, or drifted title). Specificity 100%, recall 88.9%, Cohen's κ = 0.887.

Field-level details and the audit history live in the paper Methods.

## Anonymization in `human_validation/`

The `model` column in `human_authenticity_reviews.csv` has been replaced with stable pseudonyms (`model_anon_NN`) so that human-reviewer judgements cannot be retrospectively used to identify per-model authentication errors for closed-API models whose raw outputs we are not permitted to redistribute. The pseudonym mapping is **not** included.

## License

CC-BY-4.0. See [`../LICENSE`](../LICENSE).

## Citation

See top-level [`README.md`](../README.md).
