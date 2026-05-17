"""Build FINAL DATA/ aggregates: all_references.csv + model_summary.csv."""
import csv
import glob
import os
from collections import defaultdict

ROOT = "/Users/lilo/DATA/10x24"
PER_MODEL = f"{ROOT}/FINAL DATA/reference_level/per_model"
OUT_REFS = f"{ROOT}/FINAL DATA/reference_level/all_references.csv"
OUT_SUMMARY = f"{ROOT}/FINAL DATA/aggregated/model_summary.csv"
K2 = f"{ROOT}/GOOD DATA/relevance_judge_k2_results.csv"
MATRIX = f"{ROOT}/data/final/aggregated/model_topic_quality_matrix.csv"

# K2 file uses short model keys; per_model CSVs use long keys. Map long→short.
ALIAS = {
    "deepseek_r1_671b_a37b": "deepseek_r1",
    "deepseek_v3_671b_a37b": "deepseek_v3",
    "kimi_k2_1t_a32b": "kimi_k2",
    "llama_3_1_405b": "llama_3_1_405b_hermes",
    "llama_4_maverick_400b_a17b": "llama_4_maverick",
    "llama_4_scout_109b_a17b": "llama_4_scout",
    "minimax_m25_230b_a10b": "minimax_m25",
    "mistral_large_2_123b": "mistral_large_2",
    "mistral_medium_31_250b": "mistral_medium_31",
    "mistral_small_32_24b": "mistral_small_32",
    "mixtral_8x22b_141b_a39b": "mixtral_8x22b",
    "qwen35_397b_a17b": "qwen35_397b",
    "deepseek_v4_pro_1600b_a49b": "deepseek_v4_pro",
}

k2_lookup = {}
k2_any_topic = {}  # fallback for cross-topic duplicate refs
with open(K2) as f:
    for row in csv.DictReader(f):
        key = (row["model"].strip(), row["topic"].strip(), row["reference"].strip())
        k2_lookup[key] = row["verdict"].strip()
        k2_any_topic[(row["model"].strip(), row["reference"].strip())] = row["verdict"].strip()

KEEP = ["reference", "topic", "model", "svris_status", "score", "error_distance",
        "field_title", "field_authors", "field_year", "field_venue", "field_id",
        "candidate_title", "candidate_doi", "candidate_url"]

files = sorted(glob.glob(f"{PER_MODEL}/*.csv"))
# Opus dual-file mess consolidated 2026-05-03: regular file is now canonical (233 rows,
# full schema). Manual file moved to data/deprecated/opus_dual_files_2026-05-03/.
skip_dups = {"claude_opus_46_nogeo_results_verified_v2_manual.csv",
             "perplexity_sonar_reasoning_pro_nogeo_results_verified_v2.csv"}
files = [f for f in files if os.path.basename(f) not in skip_dups]

total = 0
with open(OUT_REFS, "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(KEEP + ["k2_relevance", "source_file"])
    for fp in files:
        fname = os.path.basename(fp)
        with open(fp) as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_long = row.get("model", "").strip()
                model_short = ALIAS.get(model_long, model_long)
                topic = row.get("topic", "").strip()
                ref = row.get("reference", "").strip()
                k2 = k2_lookup.get((model_short, topic, ref), "") \
                  or k2_lookup.get((model_long, topic, ref), "") \
                  or k2_any_topic.get((model_short, ref), "") \
                  or k2_any_topic.get((model_long, ref), "")
                writer.writerow([row.get(c, "") for c in KEEP] + [k2, fname])
                total += 1
print(f"all_references.csv: {total} rows from {len(files)} files")

agg = defaultdict(lambda: {"auth": [], "rel": [], "qual": [], "n": 0,
                            "active": "", "total": "", "arch": ""})
with open(MATRIX) as f:
    for row in csv.DictReader(f):
        m = row["model"]
        agg[m]["active"] = row["active_params_B"]
        agg[m]["total"] = row["total_params_B"]
        agg[m]["arch"] = row["architecture"]
        try:
            agg[m]["auth"].append(float(row["authenticity"]))
            agg[m]["rel"].append(float(row["relevance"]))
            agg[m]["qual"].append(float(row["quality"]))
            agg[m]["n"] += int(row["n_refs"])
        except ValueError:
            pass

with open(OUT_SUMMARY, "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["model", "architecture", "active_params_B", "total_params_B",
                "n_refs", "authenticity", "relevance", "quality"])
    for m in sorted(agg):
        a = agg[m]
        mean = lambda xs: sum(xs) / len(xs) if xs else ""
        w.writerow([m, a["arch"], a["active"], a["total"], a["n"],
                    f"{mean(a['auth']):.4f}" if a["auth"] else "",
                    f"{mean(a['rel']):.4f}" if a["rel"] else "",
                    f"{mean(a['qual']):.4f}" if a["qual"] else ""])
print(f"model_summary.csv: {len(agg)} models")
