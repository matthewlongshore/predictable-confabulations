# Predictable Confabulations — Data & Reproducibility Bundle

Data and code accompanying:

> **Predictable confabulations: factual recall by LLMs scales with model size and topic frequency.**
> Smith, Shock, Segun, Olatunji, Bissyandé. arXiv preprint, 2026.

We prompted 38 large language models for 10 scholarly references on each of 24 topics, then verified every reference against OpenAlex, Google Scholar, Google Books, and DOI registries via [SourceVerify](https://sourceverify.ai) (SVRIS). This repository contains the verified per-reference outputs for the 31 open-weights models, aggregate scores for all 38 models, the topic-relevance judgements from Kimi K2, the human-validation review set, and the scripts needed to reproduce the pipeline end-to-end.

## Contents

- [`data/`](data/) — verified references, aggregate scores, topic metadata, K2 relevance judgements, human-validation reviews. See [`data/README.md`](data/README.md).
- [`scripts/`](scripts/) — generation, verification, K2 judging, aggregation, plotting. See [`scripts/README.md`](scripts/README.md).
- [`LICENSE`](LICENSE) — CC-BY-4.0 for the data and any non-API-output text produced by these scripts.

## Quick facts

- 38 models, 24 topics, ~10 references per (model, topic) cell.
- 8,913 references produced; 8,829 analysed after collapsing within-cell title duplicates.
- Two predictors (total parameter count, topic frequency in OpenAlex) explain 60% of cross-model quality variance; 74–94% within a single model family.
- Verification is non-deterministic at the margin (retest r=0.675); reported authenticity is a lower bound.

## Citation

```bibtex
@article{smith2026recall,
  title         = {Predictable confabulations: factual recall by LLMs scales with model size and topic frequency},
  author        = {Smith, Shock, Segun, Olatunji, and Bissyand\'e},
  year          = {2026},
  archivePrefix = {arXiv}
}
```

arXiv ID: _to be assigned_.

Dataset version: v1.0 · Last updated: 2026-05-17
