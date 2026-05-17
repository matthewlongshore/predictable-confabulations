#!/usr/bin/env python3
"""Sigmoid quality fit on dense + MoE models with known param sizes.
Uses TOTAL params (better predictor for MoE per CLAUDE.md §5)."""
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expit, logit as logit_fn

models = {}
with open("data/final/aggregated/model_topic_quality_matrix.csv") as f:
    for row in csv.DictReader(f):
        m = row["model"]
        if m not in models:
            models[m] = {"active": row["active_params_B"], "total": row["total_params_B"],
                        "arch": row["architecture"], "qualities": []}
        if row["quality"]:
            models[m]["qualities"].append(float(row["quality"]))

dense_pts, moe_pts, unknown_pts = [], [], []
for m, d in models.items():
    if not d["qualities"]: continue
    avg_q = np.mean(d["qualities"])
    if d["arch"] == "dense" and d["total"]:
        if "Sonar" in m: continue
        dense_pts.append((float(d["total"]), avg_q, m))
    elif d["arch"] == "moe" and d["total"]:
        moe_pts.append((float(d["total"]), avg_q, m))
    elif d["arch"] == "unknown":
        unknown_pts.append((avg_q, m))

dense_pts.sort(); moe_pts.sort()

def sig(logP, a, g): return expit(a*logP + g)

all_x = np.array([np.log10(p) for p,_,_ in dense_pts+moe_pts])
all_y = np.array([q for _,q,_ in dense_pts+moe_pts])
popt, _ = curve_fit(sig, all_x, all_y, p0=[1.27,-3.0],
                    bounds=([0.3,-8.0],[4.0,0.0]), maxfev=10000)
a, g = popt
y_pred = sig(all_x, *popt)
ss_res = np.sum((all_y - y_pred)**2)
ss_tot = np.sum((all_y - all_y.mean())**2)
r2 = 1 - ss_res/ss_tot
print(f"Combined dense+MoE sigmoid: σ({a:.3f}·log₁₀(P_total) + {g:.3f})")
print(f"  n={len(all_x)} models, R²={r2:.3f}")
print(f"  Half-max: {10**(-g/a):.0f}B")
print(f"  95% ceiling: {10**((2.944-g)/a):.0f}B")

# Dense-only fit for comparison
dx = np.array([np.log10(p) for p,_,_ in dense_pts])
dy = np.array([q for _,q,_ in dense_pts])
popt_d, _ = curve_fit(sig, dx, dy, p0=[1.27,-3.0],
                      bounds=([0.3,-8.0],[4.0,0.0]), maxfev=10000)
print(f"Dense-only:                σ({popt_d[0]:.3f}·log₁₀(P) + {popt_d[1]:.3f})")

fig, ax = plt.subplots(figsize=(11,7))
xc = np.linspace(-0.2, 3.5, 400)
ax.plot(xc, sig(xc,*popt), 'b-', lw=2.5, alpha=0.55,
        label=f'Combined fit: σ({a:.2f}·log₁₀P {g:+.2f}), R²={r2:.2f}', zorder=1)
ax.plot(xc, sig(xc,*popt_d), 'b--', lw=1.2, alpha=0.4,
        label=f'Dense-only fit', zorder=1)

# Highlight Llama 3.1 family with connecting line
LLAMA_31_FAMILY = ("Llama 3.2 1B","Llama 3.1 8B","Llama 3.1 70B","Llama 3.1 405B Hermes")
fam = [(p,q,m) for p,q,m in dense_pts if m in LLAMA_31_FAMILY]
fam.sort()
if len(fam) >= 2:
    fx = [np.log10(p) for p,_,_ in fam]
    fy = [q for _,q,_ in fam]
    ax.plot(fx, fy, color='darkorange', lw=2, alpha=0.7, zorder=4,
            label='Llama 3.1 family')

for p,q,m in dense_pts:
    is_31 = m in LLAMA_31_FAMILY
    is_33 = m == "Llama 3.3 70B"
    color = 'darkorange' if is_31 else ('#c0392b' if is_33 else 'royalblue')
    size = 110 if (is_31 or is_33) else 85
    edge = 'black' if (is_31 or is_33) else 'white'
    ax.scatter(np.log10(p), q, c=color, s=size, zorder=6 if (is_31 or is_33) else 5,
               edgecolors=edge, linewidth=1.0 if (is_31 or is_33) else 0.5)
    # Custom label
    if is_31:
        short = m.replace("Llama 3.1 ","L3.1 ").replace("Llama 3.2 ","L3.2 ")
    elif is_33:
        short = "L3.3 70B"
    else:
        short = m.replace("Llama 3.1 ","Llama ").replace("Llama 3.3 ","Llama ")
        short = short.replace("Llama 3.2 ","Llama ").replace(" nothink","")
        short = short.replace(" think"," (think)").replace("Mistral ","Mist. ")
        short = short.replace("Gemma 3 ","Gemma ").replace("Gemma 4 ","Gemma4 ")
    fontw = 'bold' if (is_31 or is_33) else 'normal'
    fontc = '#cc6600' if is_31 else ('#8b1a1a' if is_33 else '#1f3b80')
    # 3.3 70B label below the point to avoid colliding with 3.1 70B
    yoff = -12 if is_33 else 4
    ax.annotate(short, (np.log10(p),q), textcoords="offset points",
                xytext=(6,yoff), fontsize=7.5 if (is_31 or is_33) else 7,
                color=fontc, alpha=0.9, fontweight=fontw)

for p,q,m in moe_pts:
    ax.scatter(np.log10(p), q, c='#d62728', s=85, zorder=5,
               marker='s', edgecolors='white', linewidth=0.5)
    short = m.replace("DeepSeek ","DS ").replace("Llama 4 ","L4 ")
    short = short.replace("Mixtral ","Mx ").replace("MiniMax ","MM ")
    short = short.replace("Kimi ","")
    ax.annotate(short, (np.log10(p),q), textcoords="offset points",
                xytext=(6,-10), fontsize=7, color='#8b1a1a', alpha=0.85)

ax.set_xlabel('log₁₀(Total Parameters, billions)', fontsize=12)
ax.set_ylabel('Quality (authenticity × relevance)', fontsize=12)
ax.set_title('Quality Sigmoid: Dense + MoE Models (known total params)', fontsize=13)
ax.set_xticks([0,0.5,1,1.5,2,2.5,3,3.2])
ax.set_xticklabels(['1B','3B','10B','30B','100B','300B','1T','1.6T'])
ax.set_xlim(-0.3, 3.4); ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.2)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor='royalblue',markersize=10,label=f'Dense (n={len(dense_pts)})'),
    Line2D([0],[0],marker='s',color='w',markerfacecolor='#d62728',markersize=10,label=f'MoE (n={len(moe_pts)})'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='darkorange',markersize=11,markeredgecolor='black',label='Llama 3.1 family'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#c0392b',markersize=11,markeredgecolor='black',label='Llama 3.3 70B'),
    Line2D([0],[0],color='blue',lw=2.5,alpha=0.55,label=f'Combined fit (R²={r2:.2f})'),
    Line2D([0],[0],color='blue',lw=1.2,alpha=0.4,linestyle='--',label='Dense-only fit'),
], loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('plots/sigmoid_dense_moe.png', dpi=150, bbox_inches='tight')
print("\nSaved plots/sigmoid_dense_moe.png")
