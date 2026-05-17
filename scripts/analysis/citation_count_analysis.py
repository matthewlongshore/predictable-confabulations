#!/usr/bin/env python3
"""
Test prediction: models preferentially recall highly-cited papers.
Smaller models should only get the most-cited papers right (strongest signal).
Larger models should also recall less-cited papers.

Uses OpenAlex API to look up citation counts for verified references.
"""

import json
import time
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

DATA = Path("/Users/lilo/DATA/10x24/data/working")
CACHE = Path("/Users/lilo/DATA/10x24/data/citation_cache.json")
PLOTS = Path("/Users/lilo/DATA/10x24/plots")

# Stopwords stripped before computing title-overlap to avoid spurious matches
# driven by closed-class words (the, of, and, etc.). Kept short and conservative.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "the", "to", "with", "that", "this", "these",
    "those", "into", "onto", "upon", "between", "among", "across", "within",
    "via", "vs", "versus",
}
OVERLAP_THRESHOLD = 0.5  # content-word overlap (stopwords removed)


def content_words(text):
    """Lowercase tokens with stopwords removed."""
    return {w for w in re.findall(r"\w+", (text or "").lower()) if w not in STOPWORDS}


def passes_overlap(query_title, found_title, threshold=OVERLAP_THRESHOLD):
    """True if content-word overlap (intersection / query) >= threshold."""
    qw = content_words(query_title)
    fw = content_words(found_title)
    if not qw:
        return False
    return len(qw & fw) / len(qw) >= threshold

# All dense models with known parameter counts (matches Paper 1 primary set, n=16)
MODELS = [
    ("llama_3_2_1b_nogeo_results_verified_v2.csv", "Llama 1B", 1, "Llama"),
    ("gemma_3_4b_nogeo_results_verified_v2.csv", "Gemma 4B", 4, "Gemma"),
    ("llama_3_1_8b_nogeo_results_verified_v2.csv", "Llama 8B", 8, "Llama"),
    ("qwen3_8b_nothink_nogeo_results_verified_v2.csv", "Qwen3 8B", 8, "Qwen3"),
    ("gemma_3_12b_nogeo_results_verified_v2.csv", "Gemma 12B", 12, "Gemma"),
    ("qwen3_14b_nothink_nogeo_results_verified_v2.csv", "Qwen3 14B", 14, "Qwen3"),
    ("mistral_small_32_24b_nogeo_results_verified_v2.csv", "Mistral 24B", 24, "Mistral"),
    ("gemma_3_27b_nogeo_results_verified_v2.csv", "Gemma 27B", 27, "Gemma"),
    ("gemma_4_31b_nogeo_results_verified_v2.csv", "Gemma 4 31B", 31, "Gemma"),
    ("qwen3_32b_nothink_nogeo_results_verified_v2.csv", "Qwen3 32B", 32, "Qwen3"),
    ("llama_3_1_70b_nogeo_results_verified_v2.csv", "Llama 3.1 70B", 70, "Llama"),
    ("llama_3_3_70b_nogeo_results_verified_v2.csv", "Llama 3.3 70B", 70, "Llama"),
    ("mistral_large_2_123b_nogeo_results_verified_v2.csv", "Mistral 123B", 123, "Mistral"),
    ("mistral_medium_31_250b_nogeo_results_verified_v2.csv", "Mistral 250B", 250, "Mistral"),
    ("llama_3_1_405b_base_nogeo_results_verified_v2.csv", "Llama 405B", 405, "Llama"),
    ("llama_3_1_405b_nogeo_results_verified_v2.csv", "405B Hermes", 405, "Llama"),
    # MoE models
    ("gpt_oss_120b_nogeo_results_verified_v2.csv", "GPT-OSS", 120, "MoE"),
    ("minimax_m25_230b_a10b_nogeo_results_verified_v2.csv", "MiniMax", 230, "MoE"),
    ("llama_4_scout_109b_a17b_nogeo_results_verified_v2.csv", "Scout", 109, "MoE"),
    ("llama_4_maverick_400b_a17b_nogeo_results_verified_v2.csv", "Maverick", 400, "MoE"),
    ("kimi_k2_1t_a32b_nogeo_results_verified_v2.csv", "Kimi K2", 1000, "MoE"),
    ("deepseek_r1_671b_a37b_nogeo_results_verified_v2.csv", "DeepSeek R1", 671, "MoE"),
    ("qwen35_397b_a17b_nogeo_results_verified_v2.csv", "Qwen3.5", 397, "MoE"),
    ("mixtral_8x22b_141b_a39b_nogeo_results_verified_v2.csv", "Mixtral", 141, "MoE"),
]

FAMILY_COLORS = {
    "Llama": "#3b82f6", "Gemma": "#f59e0b", "Qwen3": "#8b5cf6",
    "Mistral": "#10b981", "MoE": "#ef4444",
}
FAMILY_MARKERS = {
    "Llama": "D", "Gemma": "s", "Qwen3": "o", "Mistral": "^", "MoE": "s",
}

OPENALEX_URL = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "HallucinationStudy/1.0 (research)"}


def extract_title(reference):
    """Extract the paper title from an APA-style reference string."""
    # Strip markdown italics markers
    ref = reference.replace('*', '')

    # Try to extract title between year and journal/publisher
    # Pattern: Author(s) (Year). Title. Journal...
    # or: Author(s) (Year). Title. Publisher.
    m = re.search(r'\(\d{4}\)\.\s*(.+?)(?:\.\s*(?:[A-Z]|In\s|http|\d))', ref)
    if m:
        title = m.group(1).strip()
        # Remove quotes if wrapped
        title = title.strip('"').strip("'").strip('\u201c').strip('\u201d')
        # Remove trailing period
        title = title.rstrip('.')
        if len(title) >= 10:
            return title

    # Fallback: everything between (Year). and the next period after 10+ chars
    m = re.search(r'\(\d{4}\)\.\s*(.{10,}?)\.', ref)
    if m:
        title = m.group(1).strip().strip('"').strip("'").strip('\u201c').strip('\u201d')
        if len(title) >= 10:
            return title

    # Last resort: grab everything after (Year). up to a comma or period
    m = re.search(r'\(\d{4}\)\.\s*(.{10,}?)(?:[.,]\s)', ref)
    if m:
        title = m.group(1).strip().strip('"').strip("'").strip('\u201c').strip('\u201d')
        if len(title) >= 10:
            return title

    return None


def search_openalex(title, max_retries=3):
    """Search OpenAlex for a paper by title. Returns (cited_by_count, year, found_title) or None."""
    if not title or len(title) < 10:
        return None

    # Clean title for search
    search_title = title[:200]  # API has limits

    for attempt in range(max_retries):
        try:
            r = requests.get(
                OPENALEX_URL,
                params={
                    "search": search_title,
                    "per_page": 3,
                    "select": "id,title,cited_by_count,publication_year",
                },
                headers=HEADERS,
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                return None

            # Check if first result title is a reasonable match using content-word overlap
            best = results[0]
            if passes_overlap(title, best.get("title")):
                return {
                    "cited_by_count": best.get("cited_by_count", 0),
                    "year": best.get("publication_year"),
                    "found_title": best.get("title"),
                    "openalex_id": best.get("id"),
                }
            return None

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None
    return None


def main():
    # Load citation cache
    cache = {}
    if CACHE.exists():
        with open(CACHE) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached citation lookups")

    all_data = []

    for fname, model_name, params, family in MODELS:
        fpath = DATA / fname
        if not fpath.exists():
            print(f"  SKIP: {fname}")
            continue

        df = pd.read_csv(fpath)
        print(f"\n{'='*60}")
        print(f"{model_name} ({len(df)} refs)")
        print(f"{'='*60}")

        # Inclusion: SVRIS status verified or verified_with_error (per IRR audit,
        # both buckets are 100% real papers; tighter than the old score>=0.5 cut).
        status_col = df["svris_status"].astype(str).str.lower()
        keep_mask = status_col.isin({"verified", "verified_with_error"})
        verified = df[keep_mask].copy()
        unverified = df[~keep_mask].copy()
        print(f"  Verified or verified_with_error: {len(verified)}")
        print(f"  Other status: {len(unverified)}")

        # Look up citations for verified refs
        citations_verified = []
        new_lookups = 0
        n_unmatched = 0
        for _, row in verified.iterrows():
            ref = row["reference"]
            title = extract_title(ref)
            if ref in cache:
                cached = cache[ref]
                # Re-validate cached result against the current (stricter) overlap rule.
                if cached and not passes_overlap(title, cached.get("found_title")):
                    cached = None
                result = cached
            else:
                result = search_openalex(title)
                cache[ref] = result
                new_lookups += 1
                if new_lookups % 10 == 0:
                    with open(CACHE, "w") as f:
                        json.dump(cache, f)
                    time.sleep(0.2)  # rate limiting
                time.sleep(0.15)

            if result and result.get("cited_by_count") is not None:
                citations_verified.append(result["cited_by_count"])
                all_data.append({
                    "model": model_name,
                    "params": params,
                    "family": family,
                    "reference": ref,
                    "score": row["score"],
                    "status": "verified",
                    "cited_by_count": result["cited_by_count"],
                    "topic": row["topic"],
                })
            else:
                n_unmatched += 1

        print(f"  New API lookups: {new_lookups}")
        print(f"  Unmatched in OpenAlex: {n_unmatched}/{len(verified)} "
              f"({100*n_unmatched/max(1,len(verified)):.1f}%)")
        if citations_verified:
            cv = np.array(citations_verified)
            print(f"  Verified citations:   n={len(cv)}/{len(verified)}, "
                  f"median={np.median(cv):.0f}, mean={np.mean(cv):.0f}, "
                  f"min={cv.min()}, max={cv.max()}")

    # Save cache
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nSaved {len(cache)} citation lookups to cache")

    # Save full data
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(Path("/Users/lilo/DATA/10x24/data/citation_counts.csv"), index=False)
    print(f"Saved {len(df_all)} rows to data/citation_counts.csv")

    # ── Analysis ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # 1. Per-model median citation count for verified refs
    print("\nMedian citation count of correctly recalled papers:")
    model_medians = []
    for fname, model_name, params, family in MODELS:
        subset = df_all[(df_all["model"] == model_name) & (df_all["status"] == "verified")]
        if len(subset) >= 5:
            med = subset["cited_by_count"].median()
            model_medians.append((model_name, params, med, len(subset), family))
            print(f"  {model_name:15s} ({params:>5}B): median={med:>8.0f} citations (n={len(subset)})")

    # 2. Correlation: model size vs median citation count
    if len(model_medians) >= 3:
        sizes = np.log10([m[1] for m in model_medians])
        medians = [m[2] for m in model_medians]
        rho, p = spearmanr(sizes, medians)
        print(f"\n  Spearman correlation (log₁₀ params vs median citations): ρ={rho:.3f}, p={p:.4f}")
        if rho < 0:
            print("  → CONFIRMED: Larger models recall less-cited (more obscure) papers")
        else:
            print("  → Not confirmed: larger models don't seem to recall more obscure papers")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Citation count distributions per model (box plot)
    ax = axes[0]
    plot_data = []
    plot_labels = []
    plot_colors = []
    for fname, model_name, params, family in MODELS:
        subset = df_all[(df_all["model"] == model_name) & (df_all["status"] == "verified")]
        if len(subset) >= 5:
            plot_data.append(np.log10(subset["cited_by_count"].values + 1))
            plot_labels.append(model_name)
            plot_colors.append(FAMILY_COLORS.get(family, "#6b7280"))

    if plot_data:
        bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True, vert=True)
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("log₁₀(citations + 1)", fontsize=12)
        ax.set_title("a  Citation counts of recalled papers", fontsize=12, fontweight="bold", loc="left")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): Median citations vs model size (colored by family)
    ax = axes[1]
    if len(model_medians) >= 3:
        # Add legend handles
        from matplotlib.lines import Line2D
        seen_families = set()
        legend_handles = []
        for name, params_b, med, n, fam in model_medians:
            c = FAMILY_COLORS.get(fam, "#6b7280")
            mk = FAMILY_MARKERS.get(fam, "o")
            ax.scatter(params_b, med, c=c, marker=mk, s=120, zorder=5,
                       edgecolors="white", linewidth=0.5)
            ax.annotate(name, (params_b, med), textcoords="offset points", xytext=(8, 5),
                        fontsize=7, fontweight="bold", color=c)
            if fam not in seen_families:
                seen_families.add(fam)
                legend_handles.append(Line2D([0], [0], marker=mk, color="w",
                                             markerfacecolor=c, markersize=10, label=fam))

        # Fit line in log space
        log_sizes = np.log10([m[1] for m in model_medians])
        log_meds = np.log10([m[2] for m in model_medians])
        m_fit, b_fit = np.polyfit(log_sizes, log_meds, 1)
        x_fit = np.linspace(log_sizes.min() - 0.1, log_sizes.max() + 0.1, 100)
        ax.plot(10**x_fit, 10**(m_fit * x_fit + b_fit), "--", color="#9ca3af",
                linewidth=1.5, alpha=0.7)

        ss_res = np.sum((log_meds - (m_fit * log_sizes + b_fit)) ** 2)
        ss_tot = np.sum((log_meds - np.mean(log_meds)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        ax.text(0.05, 0.05, f"slope={m_fit:.2f}\n$R^2$={r2:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters (B)", fontsize=12)
        ax.set_ylabel("Median citation count", fontsize=12)
        ax.set_title("b  Model size vs citation threshold", fontsize=12, fontweight="bold", loc="left")
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (c): CDF overlay — citation count distributions per model
    ax = axes[2]
    # Pick a subset for readability: smallest, mid, largest from each family
    cdf_models = ["Gemma 4B", "Llama 8B", "Qwen3 8B", "Llama 70B",
                  "Mistral 250B", "Llama 405B", "405B Hermes"]
    for fname, model_name, params, family in MODELS:
        subset = df_all[(df_all["model"] == model_name) & (df_all["status"] == "verified")]
        if len(subset) >= 5 and model_name in cdf_models:
            citations = np.sort(subset["cited_by_count"].values)
            cdf = np.arange(1, len(citations) + 1) / len(citations)
            c = FAMILY_COLORS.get(family, "#6b7280")
            # Vary line style by size
            lw = 1.5 if params < 30 else 2.5
            ls = "--" if params < 30 else "-"
            ax.plot(citations + 1, cdf, ls, color=c, linewidth=lw,
                    label=f"{model_name} (n={len(citations)})", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Citation count", fontsize=12)
    ax.set_ylabel("Cumulative fraction", fontsize=12)
    ax.set_title("c  CDF of citation counts (verified refs)", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    outpath = PLOTS / "citation_count_analysis.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
