"""Rebuild model_topic_quality matrix (dedup + non-dedup) from verified_v2 + K2.
Handles K2-short-key ↔ main-matrix-long-key mapping."""
import csv, glob, os, re
from collections import defaultdict

ROOT = "/Users/lilo/DATA/10x24"

# Map K2 short key → verified/main matrix long key (canonical = long form)
ALIAS = {
    "deepseek_r1":     "deepseek_r1_671b_a37b",
    "deepseek_v3":     "deepseek_v3_671b_a37b",
    "kimi_k2":         "kimi_k2_1t_a32b",
    "llama_3_1_405b_hermes": "llama_3_1_405b",
    "llama_4_maverick": "llama_4_maverick_400b_a17b",
    "llama_4_scout":   "llama_4_scout_109b_a17b",
    "minimax_m25":     "minimax_m25_230b_a10b",
    "mistral_large_2": "mistral_large_2_123b",
    "mistral_medium_31": "mistral_medium_31_250b",
    "mistral_small_32": "mistral_small_32_24b",
    "mixtral_8x22b":   "mixtral_8x22b_141b_a39b",
    "qwen35_397b":     "qwen35_397b_a17b",
    "deepseek_v4_pro": "deepseek_v4_pro_1600b_a49b",
}

def norm_title(ref):
    m = re.search(r'\(\d{4}[a-z]?\)\.\s*(.+?)(?:\.\s+[A-Z]|\.$)', ref)
    t = (m.group(1) if m else ref[:120]).lower()
    t = re.sub(r'[^a-z0-9 ]', '', t)
    return re.sub(r'\s+', ' ', t).strip()

V = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}

# K2 verdicts: key by (resolved_model_key, topic, reference)
k2 = {}
for r in csv.DictReader(open(f"{ROOT}/GOOD DATA/relevance_judge_k2_results.csv")):
    if r["verdict"] not in V: continue
    mk = ALIAS.get(r["model"], r["model"])
    k2[(mk, r["topic"], r["reference"])] = V[r["verdict"]]

# Metadata from existing matrix (then add gemma_4_31b + mixtral_8x7b if absent)
meta, oa = {}, {}
for r in csv.DictReader(open(f"{ROOT}/data/final/aggregated/model_topic_quality_matrix.backup.csv")):
    meta[r["model_key"]] = {"model": r["model"], "active_params_B": r["active_params_B"],
                             "total_params_B": r["total_params_B"], "architecture": r["architecture"]}
    oa[r["topic"]] = r["openalex_works"]

meta.setdefault("mixtral_8x7b_47b_a13b", {"model": "Mixtral 8x7B", "active_params_B": "13",
                                           "total_params_B": "47", "architecture": "moe"})
meta.setdefault("grok_3", {"model": "Grok 3", "active_params_B": "",
                            "total_params_B": "", "architecture": "unknown"})
meta.setdefault("deepseek_v4_pro_1600b_a49b", {"model": "DeepSeek V4 Pro",
                            "active_params_B": "49", "total_params_B": "1600",
                            "architecture": "moe"})
meta.setdefault("llama_3_1_70b", {"model": "Llama 3.1 70B", "active_params_B": "70",
                            "total_params_B": "70", "architecture": "dense"})

# Aggregate per (model_key, topic). Two passes: with and without dedup.
raw  = defaultdict(list)   # no dedup: list of (auth, rel)
dedup = defaultdict(list)  # dedup
seen = defaultdict(set)    # (model, topic) -> set of normalized titles

for fp in glob.glob(f"{ROOT}/GOOD DATA/*_nogeo_results_verified_v2.csv"):
    for r in csv.DictReader(open(fp)):
        if not r.get("score"): continue
        mk = r.get("model", "")
        mk = ALIAS.get(mk, mk)
        rel = k2.get((mk, r["topic"], r["reference"]))
        if rel is None: continue
        a = float(r["score"])
        raw[(mk, r["topic"])].append((a, rel))
        nt = norm_title(r["reference"])
        if nt not in seen[(mk, r["topic"])]:
            seen[(mk, r["topic"])].add(nt)
            dedup[(mk, r["topic"])].append((a, rel))

def emit(agg, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","model_key","active_params_B","total_params_B","architecture",
                    "topic","openalex_works","n_refs","authenticity","relevance","quality"])
        for (mk, topic) in sorted(agg):
            if mk not in meta: continue
            m = meta[mk]; vals = agg[(mk, topic)]
            n = len(vals)
            auth = sum(a for a,_ in vals)/n
            rel  = sum(r for _,r in vals)/n
            qual = sum(a*r for a,r in vals)/n
            w.writerow([m["model"], mk, m["active_params_B"], m["total_params_B"],
                        m["architecture"], topic, oa.get(topic, ""), n,
                        f"{auth:.4f}", f"{rel:.4f}", f"{qual:.4f}"])

emit(raw,   f"{ROOT}/data/final/aggregated/model_topic_quality_matrix.csv")
emit(dedup, f"{ROOT}/data/final/aggregated/model_topic_quality_matrix_dedup.csv")

dropped = sum(len(raw[k]) - len(dedup[k]) for k in raw)
models_raw   = len({k[0] for k in raw})
models_dedup = len({k[0] for k in dedup})
print(f"Rebuilt matrix: {len(raw)} rows ({models_raw} models)")
print(f"Deduped:        {len(dedup)} rows ({models_dedup} models), {dropped} dupes dropped")
