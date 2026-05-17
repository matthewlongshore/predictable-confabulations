# Schema

Column dictionaries for every CSV / JSON file in `data/`.

## `verified/<model>_nogeo_results_verified_v2.csv`

One row per generated reference. Each model has its own file.

| column | description |
|---|---|
| `reference` | raw APA-formatted citation as emitted by the model |
| `topic` | one of 24 topics |
| `model` | model identifier (matches the filename stem) |
| `svris_status` | `verified` / `verified-with-error` / `unverified` / `needs-human` |
| `svris_reference` | SVRIS's normalised form of the citation (for joining) |
| `field_title`, `field_authors`, `field_year`, `field_venue`, `field_id` | per-field verdict: `MATCH` / `ABBREV` / `CONTAINS` / `CONTRADICTION` / `UNCONFIRMED` / `ABSENT` |
| `candidate_title`, `candidate_authors`, `candidate_year`, `candidate_venue`, `candidate_doi`, `candidate_url`, `candidate_source` | best matching record SVRIS found (or empty if none) |
| `candidate_matches`, `candidate_contradictions`, `candidate_contains`, `candidate_absent`, `candidate_unconfirmed` | counts of fields in each verdict category |
| `cmp_<field>_citation`, `cmp_<field>_candidate` | side-by-side values used for the field-level comparison |
| `match_set_size`, `contained_set_size`, `total_candidates` | retrieval-set sizes |
| `sources_queried` | which sources returned hits (`openalex`, `google_scholar`, `google_books`, `crossref`) |
| `processing_time_ms` | SVRIS wall time for this reference |
| `svris_message` | SVRIS diagnostic message (mostly empty) |
| `score` | continuous authenticity score, `[0, 1]`. Formula below. |
| `error_distance` | `1 − score` |

**Authenticity formula** (over non-ABSENT fields):

```
score          = max(0, Σ(wᵢ · vᵢ) / Σ(wᵢ))
error_distance = 1 − score
```

Field weights: Title 0.25, Identifier 0.25, Authors 0.20, Year 0.15, Venue 0.15.
Verdict values: `MATCH` +1.0, `ABBREV` +0.75, `CONTAINS` +0.5, `UNCONFIRMED` 0.0, `CONTRADICTION` −1.0. `ABSENT` is excluded from both numerator and denominator. A contradicted field is worse than a missing one.

## `aggregated/model_summary.csv`

One row per model (all 38, both tiers).

| column | description |
|---|---|
| `model` | display name |
| `family` | Llama / Qwen / Gemma / Mistral / DeepSeek / GPT-5 / Claude / Mixtral / Kimi / Llama4 / Grok / GPT-OSS / MiniMax |
| `architecture` | `dense` / `MoE` |
| `active_params_B` | active parameters in billions |
| `total_params_B` | total parameters in billions (= `active_params_B` for dense) |
| `n_refs` | total references analysed |
| `authenticity` | mean per-reference SVRIS score |
| `relevance` | mean K2 relevance score |
| `quality` | `authenticity × relevance` |
| `usable_rate` | fraction of references that are both SVRIS-verified (any variant) **and** K2-positive |
| `k2_yes_pct`, `k2_partial_pct`, `k2_no_pct` | K2 verdict breakdown (%) |

## `aggregated/model_topic_quality_matrix_dedup.csv`

Long format: one row per (model × topic). 910 of 912 possible cells are populated across 38 models × 24 topics (2 cells had zero output: GPT-5 Nano on "Biometric voter registration", DeepSeek V3 on "School dropout"). Columns include `model`, `model_key`, `active_params_B`, `total_params_B`, `architecture`, `topic`, `openalex_works`, `n_refs`, `authenticity`, `relevance`, and the derived `quality` (= authenticity × relevance). Cell value of quality after collapsing within-cell normalised-title duplicates.

## `aggregated/openalex_topic_counts.json`

| key | description |
|---|---|
| `<topic>` | mapping from topic name to OpenAlex scholarly-work count, queried 2026-03-08 using the `title_and_abstract.search` filter |

This is the `S` axis in `quality = m·log₁₀(P_total) + n·log₁₀(S) + c`.

## `aggregated/title_match_matrix.csv`

Same shape as `model_topic_quality_matrix_dedup.csv`, but each cell value is the **binary title-match rate** (fraction of references whose `field_title` ∈ {`MATCH`, `ABBREV`}). Used for the robustness check that re-fits the sigmoid against a strictly less-informative metric.

## `relevance/k2_relevance_results.csv`

Kimi K2 topic-relevance judgements. Filtered to Tier A models only (Tier B titles would expose closed-API outputs).

| column | description |
|---|---|
| `model` | matches `verified/` filename stem |
| `topic` | one of 24 |
| `reference` | full APA citation string |
| `extracted_title` | parsed title used as the K2 input |
| `verdict` | `YES` / `PARTIAL` / `NO` |

Judge spec: Kimi K2 (`moonshotai/kimi-k2`), temperature 0, accessed via OpenRouter. Prompt: see `scripts/relevance/judge_k2_openrouter.py`.

## `human_validation/human_authenticity_reviews.csv`

326 human reviews of individual references, 5 reviewers.

| column | description |
|---|---|
| `ref_id` | stable identifier |
| `reviewer` | reviewer name |
| `model` | **anonymized** as `model_anon_NN` |
| `topic` | one of 24 |
| `human_paper_exists` | reviewer's overall verdict: `yes` / `yes_errors` / `no` |
| `svris_status` | the SVRIS status at review time |
| `svris_score` | the SVRIS score at review time |
| `k2_verdict` | K2 verdict at review time |
| `title_match`, `authors_match`, `year_correct`, `venue_match`, `is_relevant` | field-level reviewer judgements: `exact` / `wrong` / `partial` / blank |
| `notes` | free-text reviewer comments (model-name strings have been scrubbed) |
| `created_at` | review submission timestamp (UTC) |
| `reference` | full APA citation string |

Paper analyses use the independent set (one reviewer's reviews excluded as an internal sanity check), `n = 301` ratings, 4 reviewers, 288 unique references. Confusion-matrix summary: TP=136, FP=0, TN=148, FN=17, Cohen's κ=0.887. Details in the paper Methods.
