#!/usr/bin/env python3
"""
Score verified references using field-level error distance measure.

d = 1 - max(0, Σ(wᵢ × vᵢ) / Σ(wᵢ))   for non-ABSENT fields

Weights: Title=0.25, Authors=0.20, Year=0.15, Venue=0.15, Identifier=0.25
Verdicts: MATCH=1, ABBREV=0.75, CONTAINS=0.5, CONTRADICTION=-1, UNCONFIRMED=0, ABSENT=excluded
"""

import csv
import sys

WEIGHTS = {
    "title": 0.25,
    "authors": 0.20,
    "year": 0.15,
    "venue": 0.15,
    "id": 0.25,
}

VERDICT_SCORES = {
    "MATCH": 1.0,
    "ABBREV": 0.75,
    "CONTAINS": 0.5,
    "CONTRADICTION": -1.0,
    "UNCONFIRMED": 0.0,
    "ABSENT": None,  # excluded
}

FIELD_COLUMNS = {
    "title": "field_title",
    "authors": "field_authors",
    "year": "field_year",
    "venue": "field_venue",
    "id": "field_id",
}


def score_reference(row):
    """Compute quality score and error distance for a single reference."""
    numerator = 0.0
    denominator = 0.0

    for field, col in FIELD_COLUMNS.items():
        verdict = row.get(col, "").strip().upper()
        if not verdict or verdict == "ABSENT":
            continue  # excluded from calculation

        w = WEIGHTS[field]
        v = VERDICT_SCORES.get(verdict, 0.0)
        if v is None:
            continue

        numerator += w * v
        denominator += w

    if denominator == 0:
        return 0.0, 1.0  # no fields to score

    raw = numerator / denominator
    score = max(0.0, raw)
    distance = 1.0 - score
    return round(score, 3), round(distance, 3)


def main():
    infile = sys.argv[1] if len(sys.argv) > 1 else "/Users/lilo/DATA/10x24/GOOD DATA/test_local_v2_results.csv"

    with open(infile, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Scoring {len(rows)} references from {infile}\n")
    print(f"{'Score':>5}  {'d':>4}  {'Status':<20}  {'T':>5} {'A':>5} {'Y':>5} {'V':>5} {'I':>5}  Reference")
    print("─" * 120)

    scores_by_status = {}

    for row in rows:
        status = row.get("svris_status", "")
        if not status:
            continue

        score, d = score_reference(row)
        ref = row.get("reference", "")[:60]

        ft = row.get("field_title", "")[:5]
        fa = row.get("field_authors", "")[:5]
        fy = row.get("field_year", "")[:5]
        fv = row.get("field_venue", "")[:5]
        fi = row.get("field_id", "")[:5]

        print(f"{score:>5.2f}  {d:>4.2f}  {status:<20}  {ft:>5} {fa:>5} {fy:>5} {fv:>5} {fi:>5}  {ref}")

        if status not in scores_by_status:
            scores_by_status[status] = []
        scores_by_status[status].append(score)

    print(f"\n{'─' * 60}")
    print(f"\nScore ranges by SVRIS status:")
    for status in ["verified", "verified-with-error", "needs-human", "unverified", "FAILED"]:
        if status in scores_by_status:
            scores = scores_by_status[status]
            avg = sum(scores) / len(scores)
            lo = min(scores)
            hi = max(scores)
            print(f"  {status:<20}  n={len(scores):>2}  avg={avg:.2f}  range=[{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
