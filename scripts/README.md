# Scripts

End-to-end reproducibility code for *Predictable confabulations*. All scripts read API credentials from environment variables — there are no embedded keys.

## Required environment variables

| var | needed by | where to get it |
|---|---|---|
| `OPENROUTER_API_KEY` | all `scripts/generation/generate_references_openrouter_*.py`, `scripts/relevance/judge_k2_openrouter.py` | <https://openrouter.ai/keys> |
| `DEEPINFRA_API_KEY` | `scripts/generation/generate_references_deepinfra_*.py` | <https://deepinfra.com/dash/api_keys> |
| `GROQ_API_KEY` | `scripts/generation/generate_references_groq*.py` | <https://console.groq.com/keys> |
| `TOGETHER_API_KEY` | `scripts/generation/generate_references_together_*.py` | <https://api.together.ai/settings/api-keys> |
| `SVRIS_API_KEY` | `scripts/verification/verify_production.py` | request access via <https://sourceverify.ai> |

Closed-API generators (OpenAI, Anthropic, xAI) are **not** distributed in this bundle — the corresponding model outputs are likewise withheld per provider Terms of Service.

## Folder map

```
scripts/
├── generation/      One generator per model. Reads topics, hits the provider API at temp 0,
│                    writes raw output CSVs to data/working/<model>_nogeo_results.csv.
├── verification/    verify_production.py — submits references to SourceVerify in batches of 5,
│                    polls every 15s, writes *_verified_v2.csv with status / score / per-field verdicts.
├── relevance/       judge_k2_openrouter.py — sends (topic, extracted_title) to Kimi K2,
│                    writes k2_relevance_results.csv with YES / PARTIAL / NO verdicts.
├── analysis/        Build the aggregated data files used in the paper.
└── plotting/        Generate the figures used in the paper.
```

## Run order

```
1. scripts/generation/generate_references_*.py        # per model — generates raw CSVs
2. scripts/verification/verify_production.py          # all generated refs → SVRIS → *_verified_v2.csv
3. scripts/relevance/judge_k2_openrouter.py           # extracted titles → K2 → k2_relevance_results.csv
4. scripts/analysis/build_final_data.py               # collects per-model CSVs into all_references.csv
5. scripts/analysis/rebuild_mtq.py                    # (model × topic) quality matrix + dedup variant
6. scripts/analysis/score_references.py               # standalone scorer if you want to re-derive the column
7. scripts/plotting/plot_sigmoid_dense_moe.py         # fits sigmoid, renders figure
```

## Notes on individual scripts

- **`scripts/analysis/citation_count_analysis.py`** — for correctly-recalled references (SVRIS status `verified` or `verified-with-error`), looks up each title in OpenAlex and analyses the citation-count tail. Used for the within-topic test of the training-data-representation hypothesis.

## Important production behaviour

- **SourceVerify is non-deterministic at the margin** (retest r ≈ 0.675). A re-run can only ever downgrade a verdict, never upgrade it. Reported authenticity is a lower bound.
- **SourceVerify rate-limits during the day.** `verify_production.py` defaults to waiting until 01:00 UTC. Pass `--now` to start immediately.
- **Mac sleep kills long verification runs.** Prefix with `caffeinate -i` when running on macOS.
- The SVRIS submission batch is 5 references, polled every 15 seconds. The script prints job IDs to stdout — record them so you can cancel stuck jobs via `POST /api/verify-reference/cancel` with `{"jobIds": [...]}`.

## Hard-coded paths

Some scripts contain absolute paths under `/Users/lilo/DATA/10x24/` from the original development environment. When adapting to your own machine, search-and-replace those with the appropriate relative paths or environment variables. These paths do not affect correctness but break out-of-the-box reproducibility.
