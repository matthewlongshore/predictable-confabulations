#!/usr/bin/env python3
"""
Verify v2 (no-geography, topic-only) references against production SourceVerify API.
Saves to *_verified_v2.csv. Resume-safe. Start it and walk away.

Usage:
    caffeinate -i python3 scripts/verify_production.py          # wait until 01:00 UTC
    caffeinate -i python3 scripts/verify_production.py --now     # start immediately

== V2 NO-GEO QUEUE ==
Done:      Llama 8B, 70B, 405B base, Scout, Maverick, GPT-5, Nano,
           Mistral Small, Medium, Perplexity Sonar, Gemma 4B, 12B, 27B,
           Qwen3 8B/14B/32B nothink = ~3,850 ✓
Tonight:   FAILED retries (8) + Mistral Medium gaps (10) = 18
Tomorrow:  Kimi K2 (~62 remaining) + Qwen3 8B think (193)  = 255
Later:     Qwen3 think 14B/32B (446) + 405B Hermes (201) + GPT-OSS (240)
           + DeepSeek R1 (222) + Mistral Large (240) + MiniMax M25 (240) = 1,589
"""

import csv
import time
import sys
import os
from datetime import datetime, timedelta, timezone
import requests

# === PRODUCTION CONFIG ===
API_URL = "https://sourceverify.ai/api/verify-reference"
API_KEY = os.environ["SVRIS_API_KEY"]  # request access at https://sourceverify.ai
BATCH_SIZE = 20         # production limit bumped to 20
POLL_INTERVAL = 20      # longer poll = more polite to prod
MAX_REFS_PER_NIGHT = 250   # Qwen3 30B-A3B (240) regenerated
REGION_FILTER = None       # v2: no geography, verify all refs

FIELDNAMES = [
    # Original reference data
    "reference", "topic", "model",
    # SVRIS verdict
    "svris_status", "svris_reference",
    # Field-level verdicts (MATCH, CONTRADICTION, CONTAINS, ABSENT, etc.)
    "field_title", "field_authors", "field_year", "field_venue", "field_id",
    # Best candidate metadata (the real paper SVRIS matched against)
    "candidate_title", "candidate_authors", "candidate_year",
    "candidate_venue", "candidate_doi", "candidate_url",
    "candidate_source",        # CROSSREF, GOOGLE, SCHOLAR, DOI, URL
    # Candidate comparison summary
    "candidate_matches", "candidate_contradictions",
    "candidate_contains", "candidate_absent", "candidate_unconfirmed",
    # Per-field actual values from comparison (what was compared)
    "cmp_title_citation", "cmp_title_candidate",
    "cmp_authors_citation", "cmp_authors_candidate",
    "cmp_year_citation", "cmp_year_candidate",
    "cmp_venue_citation", "cmp_venue_candidate",
    "cmp_id_citation", "cmp_id_candidate",
    # Search metadata
    "match_set_size", "contained_set_size",
    "total_candidates", "sources_queried", "processing_time_ms",
    # Raw message (kept for reference, at the end)
    "svris_message",
    # Computed scores
    "score", "error_distance",
]

# --- Error distance scoring ---
# d = 1 - max(0, Σ(wᵢ × vᵢ) / Σ(wᵢ)) for non-ABSENT fields
FIELD_WEIGHTS = {"title": 0.25, "authors": 0.20, "year": 0.15, "venue": 0.15, "id": 0.25}
VERDICT_SCORES = {"MATCH": 1.0, "ABBREV": 0.75, "CONTAINS": 0.5, "CONTRADICTION": -1.0, "UNCONFIRMED": 0.0}
FIELD_COLS = {"title": "field_title", "authors": "field_authors", "year": "field_year",
              "venue": "field_venue", "id": "field_id"}


def compute_score(row):
    """Compute quality score and error distance from field-level verdicts."""
    num = 0.0
    den = 0.0
    for field, col in FIELD_COLS.items():
        verdict = (row.get(col) or "").strip().upper()
        if not verdict or verdict == "ABSENT":
            continue
        w = FIELD_WEIGHTS[field]
        v = VERDICT_SCORES.get(verdict, 0.0)
        num += w * v
        den += w
    if den == 0:
        return 0.0, 1.0
    raw = num / den
    score = round(max(0.0, raw), 3)
    return score, round(1.0 - score, 3)

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

DATA_DIR = "/Users/lilo/DATA/10x24/GOOD DATA"

# V2 no-geography queue — Llama family first (scaling validation),
# then other models. Resume-safe: skips any file already fully done.
# Hermes 405B added back to compare base vs fine-tune at same param count.
QUEUE = [
    # --- Active (May 2): Qwen3 30B-A3B regenerated with parser fix, 240 fresh refs ---
    ("Qwen3 30B-A3B",       "qwen3_30b_a3b_nogeo_results.csv"),               # 240
    # --- Done previously ---
    # ("DeepSeek V4 Pro",     "deepseek_v4_pro_1600b_a49b_nogeo_results.csv"),# 240 DONE
    # ("Llama 3.1 70B",       "llama_3_1_70b_nogeo_results.csv"),             # DONE
    # ("Claude Opus 4.6",     "claude_opus_46_nogeo_results.csv"),            # DONE
    # ("Gemma 4 26B-A4B",     "gemma_4_26b_a4b_nogeo_results.csv"),           # 240
    # ("Gemma 4 31B",         "gemma_4_31b_nogeo_results.csv"),              # 237
    # ("Llama 3.2 1B",        "llama_3_2_1b_nogeo_results.csv"),             # 58 remaining
    # --- Already done ---
    # ("Qwen3.5 397B",        "qwen35_397b_a17b_nogeo_results.csv"),          # 206 DONE
    # ("DeepSeek V3",         "deepseek_v3_671b_a37b_nogeo_results.csv"),     # 231 DONE
    # ("Mixtral 8x22B",       "mixtral_8x22b_141b_a39b_nogeo_results.csv"),   # 240 DONE
    # --- All below already done; commented out Apr 27 to enforce single-model run ---
    # ("Sonar Reasoning Pro", "perplexity_sonar_reasoning_pro_nogeo_results.csv"),
    # ("Claude Opus 4.6",    "claude_opus_46_nogeo_results.csv"),
    # ("Claude Sonnet 4.6",   "claude_sonnet_46_nogeo_results.csv"),
    # ("GPT-5.4",             "gpt_54_nogeo_results.csv"),
    # ("GPT-5 Mini",          "gpt_5_mini_nogeo_results.csv"),
    # ("MiniMax M25",         "minimax_m25_230b_a10b_nogeo_results.csv"),
    # ("GPT-OSS 120B",        "gpt_oss_120b_nogeo_results.csv"),
    # ("Mistral Large 2",     "mistral_large_2_123b_nogeo_results.csv"),
    # ("Qwen3 8B think",      "qwen3_8b_think_nogeo_results.csv"),
    # ("Qwen3 14B think",     "qwen3_14b_think_nogeo_results.csv"),
    # ("Qwen3 32B think",     "qwen3_32b_think_nogeo_results.csv"),
    # ("Qwen3 32B nothink",   "qwen3_32b_nothink_nogeo_results.csv"),
    # ("Qwen3 8B nothink",    "qwen3_8b_nothink_nogeo_results.csv"),
    # ("Qwen3 14B nothink",   "qwen3_14b_nothink_nogeo_results.csv"),
    # ("Perplexity Sonar",    "perplexity_sonar_nogeo_results.csv"),
    # ("GPT-5 Nano",          "gpt_5_nano_nogeo_results.csv"),
    # ("Llama 405B base",     "llama_3_1_405b_base_nogeo_results.csv"),
    # ("Mistral Medium 3.1",  "mistral_medium_31_250b_nogeo_results.csv"),
    # ("Gemma 3 4B",          "gemma_3_4b_nogeo_results.csv"),
    # ("Gemma 3 12B",         "gemma_3_12b_nogeo_results.csv"),
    # ("Gemma 3 27B",         "gemma_3_27b_nogeo_results.csv"),
    # ("Llama 8B",            "llama_3_1_8b_nogeo_results.csv"),
    # ("Llama 70B",           "llama_3_3_70b_nogeo_results.csv"),
    # ("Llama 4 Scout",       "llama_4_scout_109b_a17b_nogeo_results.csv"),
    # ("Llama 4 Maverick",    "llama_4_maverick_400b_a17b_nogeo_results.csv"),
    # ("GPT-5",               "gpt_5_nogeo_results.csv"),
    # ("Mistral Small 3.2",   "mistral_small_32_24b_nogeo_results.csv"),
    # ("Kimi K2",             "kimi_k2_1t_a32b_nogeo_results.csv"),
    # ("Llama 405B Hermes",   "llama_3_1_405b_nogeo_results.csv"),
    # ("DeepSeek R1",         "deepseek_r1_671b_a37b_nogeo_results.csv"),
]


def wait_until_target():
    """Wait until 01:00 UTC."""
    now_utc = datetime.now(timezone.utc)
    target = now_utc.replace(hour=1, minute=0, second=0, microsecond=0)
    if target <= now_utc:
        target += timedelta(days=1)
    wait_seconds = (target - now_utc).total_seconds()

    hours = int(wait_seconds // 3600)
    mins = int((wait_seconds % 3600) // 60)
    now_local = datetime.now()
    print(f"Current time: {now_local.strftime('%H:%M:%S')} local / {now_utc.strftime('%H:%M:%S')} UTC")
    print(f"Waiting until 01:00 UTC ({hours}h {mins}m)...")
    print(f"Will verify up to {MAX_REFS_PER_NIGHT} refs from {len(QUEUE)} queued files.\n")

    remaining = wait_seconds
    while remaining > 0:
        sleep_chunk = min(remaining, 600)
        time.sleep(sleep_chunk)
        remaining -= sleep_chunk
        if remaining > 0:
            h = int(remaining // 3600)
            m = int((remaining % 3600) // 60)
            print(f"  {h}h {m}m until 01:00 UTC...", flush=True)

    print(f"\n--- 01:00 UTC reached: {datetime.now().strftime('%H:%M:%S')} local ---\n")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def output_path_for(input_path):
    """*_results.csv -> *_results_verified_v2.csv"""
    return input_path.replace(".csv", "_verified_v2.csv")


def load_existing(output_path):
    """Load previously verified rows, deduplicating by reference text.
    Keeps the best result for each reference (good > FAILED > ERROR)."""
    try:
        rows = read_csv(output_path)
    except FileNotFoundError:
        return []
    # Deduplicate: keep best result per reference
    best = {}
    bad_statuses = ("ERROR", "FAILED", "")
    for r in rows:
        ref = r.get("reference", "")
        status = r.get("svris_status", "")
        existing = best.get(ref)
        if existing is None:
            best[ref] = r
        elif existing.get("svris_status", "") in bad_statuses and status not in bad_statuses:
            # Replace bad with good
            best[ref] = r
        elif existing.get("svris_status", "") in bad_statuses and status in bad_statuses:
            pass  # both bad, keep first
        # else: existing is good, keep it
    return [r for r in best.values() if r.get("svris_status") not in bad_statuses]


def submit_batch(references, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, headers=HEADERS,
                              json={"references": references}, timeout=60)
            if not r.ok:
                print(f"\n    API {r.status_code}: {r.text[:300]}", flush=True)
            r.raise_for_status()
            return r.json()["jobIds"]
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"\n    Submit error ({e}), retrying in {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise


def poll_results(job_ids, max_wait=600):
    """Poll for results. Give up after max_wait seconds."""
    retries = 0
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        try:
            r = requests.post(f"{API_URL}/results", headers=HEADERS,
                              json={"jobIds": job_ids}, timeout=60)
            r.raise_for_status()
            retries = 0
        except requests.exceptions.RequestException:
            retries += 1
            if retries >= 5:
                raise
            time.sleep(15 * retries)
            elapsed += 15 * retries
            continue

        jobs = r.json()["jobs"]
        ACTIVE_STATES = ("active", "waiting")
        if all(j["state"] not in ACTIVE_STATES for j in jobs):
            return jobs
        active = sum(1 for j in jobs if j["state"] in ACTIVE_STATES)
        done = len(jobs) - active
        states = {}
        for j in jobs:
            s = j["state"]
            states[s] = states.get(s, 0) + 1
        state_str = " | ".join(f"{k}: {v}" for k, v in sorted(states.items()))
        print(f"    waiting... ({active} active, {int(max_wait - elapsed)}s left) [{state_str}]", flush=True)

        # Bail early if most are done but a few are stuck (>300s)
        if active <= 2 and done >= 10 and elapsed >= 300:
            print(f"    WARNING: {active} job(s) stuck after {elapsed}s, moving on")
            return jobs

    # Timed out — return what we have
    print(f"    WARNING: timed out after {max_wait}s, returning partial results")
    return jobs


def extract_result_fields(job):
    """Extract all fields from a completed job result, including verificationGraph data."""
    result = job["result"]
    fields = result.get("field_results", {})
    vg = result.get("verificationGraph") or {}
    nodes = vg.get("nodes", [])
    metadata = vg.get("metadata", {})

    # Find the RESULT node to get bestCandidateId
    result_node = next((n for n in nodes if n.get("type") == "RESULT"), {})
    best_cand_id = result_node.get("bestCandidateId")

    # Find the best CANDIDATE node
    best_cand = {}
    if best_cand_id:
        best_cand = next((n for n in nodes if n.get("id") == best_cand_id), {})
    if not best_cand:
        # Fallback: first CANDIDATE node
        best_cand = next((n for n in nodes if n.get("type") == "CANDIDATE"), {})

    cf = best_cand.get("candidateFields", {})
    cs = best_cand.get("comparisonSummary", {})

    # Find COMPARISON nodes for the best candidate
    cmp_nodes = [n for n in nodes
                 if n.get("type") == "COMPARISON"
                 and n.get("candidateId") == best_cand.get("id")]
    # Index by fieldName
    cmps = {n["fieldName"]: n for n in cmp_nodes}

    # Helper to get comparison values for a field
    def cmp_val(field, key):
        node = cmps.get(field, {})
        val = node.get(key, "")
        if val is None:
            return ""
        return str(val)

    # Format authors list as semicolon-separated string
    cand_authors = cf.get("authors") or []
    if isinstance(cand_authors, list):
        cand_authors = "; ".join(cand_authors)

    sources = metadata.get("sourcesQueried", [])
    if isinstance(sources, list):
        sources = ", ".join(sources)

    return {
        # SVRIS verdict
        "svris_status": result.get("status", ""),
        "svris_reference": result.get("reference", ""),
        # Field-level verdicts
        "field_title": fields.get("title", ""),
        "field_authors": fields.get("authors", ""),
        "field_year": fields.get("year", ""),
        "field_venue": fields.get("venue", ""),
        "field_id": fields.get("identifier", ""),
        # Best candidate metadata
        "candidate_title": cf.get("title") or "",
        "candidate_authors": cand_authors,
        "candidate_year": cf.get("year") or "",
        "candidate_venue": cf.get("venue") or "",
        "candidate_doi": cf.get("doi") or "",
        "candidate_url": cf.get("url") or "",
        "candidate_source": best_cand.get("sourceType", ""),
        # Comparison summary
        "candidate_matches": cs.get("matches", ""),
        "candidate_contradictions": cs.get("contradictions", ""),
        "candidate_contains": cs.get("contains", ""),
        "candidate_absent": cs.get("absent", ""),
        "candidate_unconfirmed": cs.get("unconfirmed", ""),
        # Per-field actual values
        "cmp_title_citation": cmp_val("title", "citationValue"),
        "cmp_title_candidate": cmp_val("title", "candidateValue"),
        "cmp_authors_citation": cmp_val("authors", "citationValue"),
        "cmp_authors_candidate": cmp_val("authors", "candidateValue"),
        "cmp_year_citation": cmp_val("year", "citationValue"),
        "cmp_year_candidate": cmp_val("year", "candidateValue"),
        "cmp_venue_citation": cmp_val("venue", "citationValue"),
        "cmp_venue_candidate": cmp_val("venue", "candidateValue"),
        "cmp_id_citation": cmp_val("identifier", "citationValue"),
        "cmp_id_candidate": cmp_val("identifier", "candidateValue"),
        # Search metadata
        "match_set_size": result.get("match_set_size", ""),
        "contained_set_size": result.get("contained_set_size", ""),
        "total_candidates": metadata.get("totalCandidates", ""),
        "sources_queried": sources,
        "processing_time_ms": metadata.get("processingTimeMs", ""),
        # Raw message (kept for reference)
        "svris_message": result.get("message", ""),
    }


def process_file(name, input_path, max_refs_remaining):
    """Verify up to max_refs_remaining references. Returns count verified."""
    rows = read_csv(input_path)
    if REGION_FILTER:
        rows = [r for r in rows if r.get("region", "") == REGION_FILTER]
    out_path = output_path_for(input_path)

    results = load_existing(out_path)
    done_refs = {r["reference"] for r in results}
    skip = len(done_refs)
    remaining_in_file = len(rows) - skip
    to_verify = min(remaining_in_file, max_refs_remaining)

    if to_verify <= 0:
        print(f"\n  {name}: already fully verified ({skip}/{len(rows)}), skipping.")
        return 0

    print(f"\n{'='*60}")
    print(f"  {name}: {len(rows)} refs ({skip} done, verifying {to_verify} more)")
    print(f"  Output: {out_path}")
    print(f"  API: {API_URL}")
    print(f"{'='*60}")

    verified_this_run = 0

    # Collect refs that still need verification
    pending_rows = [r for r in rows if r["reference"] not in done_refs]

    total_batches = (len(pending_rows) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(pending_rows), BATCH_SIZE):
        if verified_this_run >= to_verify:
            break

        batch_num = i // BATCH_SIZE + 1
        batch_rows = pending_rows[i:i + BATCH_SIZE]
        refs = [r["reference"] for r in batch_rows]

        print(f"  [{batch_num}/{total_batches}] Submitting {len(refs)} refs...", end=" ", flush=True)

        try:
            job_ids = submit_batch(refs)
            jobs = poll_results(job_ids)

            for row, job in zip(batch_rows, jobs):
                if job["state"] == "completed" and job.get("result"):
                    row.update(extract_result_fields(job))
                else:
                    row["svris_status"] = "FAILED"
                    row["svris_message"] = job.get("error", "unknown error")
                    for f in FIELDNAMES:
                        if f not in ("reference", "topic", "model",
                                     "svris_status", "svris_message"):
                            row.setdefault(f, "")

                if row.get("svris_status") == "FAILED":
                    row["score"] = ""
                    row["error_distance"] = ""
                else:
                    s, d = compute_score(row)
                    row["score"] = s
                    row["error_distance"] = d
                results.append(row)
                verified_this_run += 1

            # Save after every batch
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                w.writeheader()
                w.writerows(results)

            statuses = {}
            for r in results:
                s = r["svris_status"]
                statuses[s] = statuses.get(s, 0) + 1
            status_str = " | ".join(f"{k}: {v}" for k, v in sorted(statuses.items()))
            print(f"done ({len(results)}/{len(rows)}) [{status_str}]")

        except Exception as e:
            print(f"ERROR: {e}")
            for row in batch_rows:
                row["svris_status"] = "ERROR"
                row["svris_message"] = str(e)
                for f in FIELDNAMES:
                    if f not in ("reference", "topic", "model",
                                 "svris_status", "svris_message"):
                        row.setdefault(f, "")
                results.append(row)
                verified_this_run += 1

            # Save even on error
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                w.writeheader()
                w.writerows(results)

    print(f"  {name}: verified {verified_this_run} refs this run (total: {len(results)}/{len(rows)})")
    return verified_this_run


def retry_failed():
    """Re-verify only FAILED refs across all verified files."""
    import glob
    files = glob.glob(os.path.join(DATA_DIR, "*_verified_v2.csv"))
    total = 0
    for vf in sorted(files):
        rows = read_csv(vf)
        failed = [r for r in rows if r.get("svris_status") == "FAILED"]
        if not failed:
            continue
        good = [r for r in rows if r.get("svris_status") != "FAILED"]
        model_name = os.path.basename(vf).replace("_nogeo_results_verified_v2.csv", "")
        print(f"\n  {model_name}: {len(failed)} FAILED refs to retry")

        refs = [r["reference"] for r in failed]
        for i in range(0, len(refs), BATCH_SIZE):
            batch_refs = refs[i:i + BATCH_SIZE]
            batch_rows = failed[i:i + BATCH_SIZE]
            print(f"    Submitting {len(batch_refs)} refs...", end=" ", flush=True)
            try:
                job_ids = submit_batch(batch_refs)
                jobs = poll_results(job_ids)
                for row, job in zip(batch_rows, jobs):
                    if job["state"] == "completed" and job.get("result"):
                        row.update(extract_result_fields(job))
                        s, d = compute_score(row)
                        row["score"] = s
                        row["error_distance"] = d
                        good.append(row)
                        print(f"{row['svris_status']}", end=" ", flush=True)
                    else:
                        # Still failed — keep with blank score
                        row["svris_status"] = "FAILED"
                        row["score"] = ""
                        row["error_distance"] = ""
                        good.append(row)
                        print("FAILED", end=" ", flush=True)
                print()
            except Exception as e:
                print(f"ERROR: {e}")
                for row in batch_rows:
                    row["score"] = ""
                    row["error_distance"] = ""
                    good.append(row)

        # Save
        with open(vf, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(good)
        total += len(failed)
        print(f"    Saved {vf}")

    print(f"\n  Retried {total} FAILED refs total.")


def main():
    if "--retry-failed" in sys.argv:
        retry_failed()
        return

    if "--now" not in sys.argv:
        wait_until_target()
    else:
        print("Skipping wait (--now flag)\n")

    total_verified = 0

    for name, filename in QUEUE:
        if total_verified >= MAX_REFS_PER_NIGHT:
            print(f"\n  Reached nightly limit ({MAX_REFS_PER_NIGHT}). Stopping.")
            break

        input_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(input_path):
            print(f"\n  {name}: file not found ({filename}), skipping.")
            continue

        budget = MAX_REFS_PER_NIGHT - total_verified
        count = process_file(name, input_path, budget)
        total_verified += count

    # Check for FAILED rows across all verified files
    import glob
    failed_summary = []
    for vf in sorted(glob.glob(os.path.join(DATA_DIR, "*_verified_v2.csv"))):
        rows = read_csv(vf)
        failed = [r for r in rows if r.get("svris_status") == "FAILED"]
        if failed:
            failed_summary.append((os.path.basename(vf), len(failed)))

    print(f"\n{'='*60}")
    print(f"  DONE: Verified {total_verified} references tonight.")
    print(f"  Finished at {datetime.now().strftime('%H:%M:%S')}")
    if failed_summary:
        total_failed = sum(c for _, c in failed_summary)
        print(f"\n  ⚠️  WARNING: {total_failed} FAILED refs need retry (scored as blank, not 0):")
        for fname, count in failed_summary:
            print(f"    {fname}: {count} FAILED")
        print(f"  Run: python3 scripts/verify_production.py --retry-failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
