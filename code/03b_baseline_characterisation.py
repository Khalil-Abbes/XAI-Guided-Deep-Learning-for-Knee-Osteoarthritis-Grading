"""
03b_baseline_characterisation.py
----------------------------------
Reads val_gradcam_results.json (already computed by 03_gradcam_analysis.py)
and computes the full baseline faithfulness characterisation.

Outputs:
  - Console: full metrics report
  - data/xai_analysis/baseline_metrics.json  (for paper table)
  - data/xai_analysis/baseline_plots.png     (4-panel figure)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# ── 1. Load results ────────────────────────────────────────────────────────────
with open("data/xai_analysis/val_gradcam_results.json") as f:
    results = json.load(f)

correct   = [r for r in results if r["correct"]]
incorrect = [r for r in results if not r["correct"]]

JSER_THRESHOLD = 0.40
GRADES = [0, 1, 2, 3, 4]
GRADE_NAMES = ["KL-0\n(Healthy)", "KL-1\n(Doubtful)", "KL-2\n(Mild)",
               "KL-3\n(Moderate)", "KL-4\n(Severe)"]

# ── 2. Global metrics ──────────────────────────────────────────────────────────
unfaithful = [r for r in correct if r["jser_hband"] < JSER_THRESHOLD]

mean_jser_all     = np.mean([r["jser_hband"] for r in results])
mean_jser_correct = np.mean([r["jser_hband"] for r in correct])
std_jser_correct  = np.std([r["jser_hband"]  for r in correct])
cbu_rate          = len(unfaithful) / len(correct)
faithfulness_score = 1 - cbu_rate

# Confidence-faithfulness Pearson correlation (correct preds only)
confidences = [r["confidence"]  for r in correct]
jser_vals   = [r["jser_hband"]  for r in correct]
pearson_r, pearson_p = stats.pearsonr(confidences, jser_vals)

# ── 3. Per-class metrics ───────────────────────────────────────────────────────
per_class = {}
for g in GRADES:
    g_correct    = [r for r in correct    if r["true_label"] == g]
    g_unfaithful = [r for r in unfaithful if r["true_label"] == g]
    g_all        = [r for r in results    if r["true_label"] == g]
    g_incorrect  = [r for r in incorrect  if r["true_label"] == g]

    per_class[g] = {
        "n_total":       len(g_all),
        "n_correct":     len(g_correct),
        "n_incorrect":   len(g_incorrect),
        "accuracy":      len(g_correct) / len(g_all) if g_all else 0,
        "n_unfaithful":  len(g_unfaithful),
        "cbu_rate":      len(g_unfaithful) / len(g_correct) if g_correct else 0,
        "mean_jser":     float(np.mean([r["jser_hband"] for r in g_correct])) if g_correct else 0,
        "std_jser":      float(np.std([r["jser_hband"]  for r in g_correct])) if g_correct else 0,
        "mean_conf":     float(np.mean([r["confidence"] for r in g_correct])) if g_correct else 0,
    }

# ── 4. JSER distribution by confidence quartile ────────────────────────────────
# Splits correct predictions into 4 confidence bins and checks if
# high-confidence predictions are also more faithful
conf_arr = np.array([r["confidence"] for r in correct])
jser_arr = np.array([r["jser_hband"] for r in correct])
quartiles = np.percentile(conf_arr, [25, 50, 75])
q_labels  = ["Low\n(<25%)", "Mid-Low\n(25-50%)", "Mid-High\n(50-75%)", "High\n(>75%)"]
q_masks   = [
    conf_arr <  quartiles[0],
    (conf_arr >= quartiles[0]) & (conf_arr < quartiles[1]),
    (conf_arr >= quartiles[1]) & (conf_arr < quartiles[2]),
    conf_arr >= quartiles[2],
]
q_mean_jser = [float(np.mean(jser_arr[m])) for m in q_masks]
q_cbu_rate  = [float(np.mean(jser_arr[m] < JSER_THRESHOLD)) for m in q_masks]

# ── 5. Print full report ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("BASELINE FAITHFULNESS CHARACTERISATION (H-Band 105-160)")
print("=" * 60)
print(f"\n── GLOBAL ──")
print(f"  Val images total          : {len(results)}")
print(f"  Correct predictions       : {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
print(f"  Mean JSER (all val)       : {mean_jser_all:.4f}")
print(f"  Mean JSER (correct only)  : {mean_jser_correct:.4f}  ± {std_jser_correct:.4f}")
print(f"  CbU Rate                  : {cbu_rate*100:.1f}%  ({len(unfaithful)}/{len(correct)})")
print(f"  Faithfulness Score (FS)   : {faithfulness_score*100:.1f}%")
print(f"  Conf-JSER Pearson r       : {pearson_r:.4f}  (p={pearson_p:.4f})")

print(f"\n── PER KL GRADE ──")
print(f"  {'Grade':<8} {'N':>5} {'Acc':>7} {'MeanJSER':>10} {'StdJSER':>9} {'CbU%':>8}")
print(f"  {'-'*52}")
for g in GRADES:
    pc = per_class[g]
    print(f"  KL-{g}    {pc['n_total']:>5}  {pc['accuracy']*100:>6.1f}%"
          f"  {pc['mean_jser']:>8.4f}  {pc['std_jser']:>8.4f}  {pc['cbu_rate']*100:>7.1f}%")

print(f"\n── CONFIDENCE QUARTILE BREAKDOWN ──")
for i, label in enumerate(q_labels):
    print(f"  {label.replace(chr(10),' '):20s}  mean JSER={q_mean_jser[i]:.4f}  "
          f"CbU rate={q_cbu_rate[i]*100:.1f}%")

# ── 6. Save metrics JSON ───────────────────────────────────────────────────────
import os
os.makedirs("data/xai_analysis", exist_ok=True)

metrics = {
    "global": {
        "n_total":          len(results),
        "n_correct":        len(correct),
        "accuracy":         len(correct) / len(results),
        "mean_jser_all":    float(mean_jser_all),
        "mean_jser_correct":float(mean_jser_correct),
        "std_jser_correct": float(std_jser_correct),
        "cbu_rate":         float(cbu_rate),
        "faithfulness_score": float(faithfulness_score),
        "pearson_r":        float(pearson_r),
        "pearson_p":        float(pearson_p),
        "jser_threshold":   JSER_THRESHOLD,
        "hband_rows":       "105-160",
    },
    "per_class": per_class,
    "confidence_quartiles": {
        "labels":     q_labels,
        "mean_jser":  q_mean_jser,
        "cbu_rate":   q_cbu_rate,
    }
}

with open("data/xai_analysis/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved: data/xai_analysis/baseline_metrics.json")

# ── 7. 4-panel figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Baseline Saliency Faithfulness Characterisation\n"
             "ResNet-50 (No XAI Guidance) — H-Band JSER (rows 105-160)",
             fontsize=13, fontweight="bold")

# Panel 1: Per-class mean JSER bar chart
ax1 = axes[0, 0]
means = [per_class[g]["mean_jser"] for g in GRADES]
stds  = [per_class[g]["std_jser"]  for g in GRADES]
colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]
bars = ax1.bar(GRADE_NAMES, means, yerr=stds, color=colors,
               capsize=5, width=0.6, edgecolor="white", linewidth=0.5)
ax1.axhline(JSER_THRESHOLD, color="black", linestyle="--",
            linewidth=1.5, label=f"Unfaithful threshold ({JSER_THRESHOLD})")
ax1.set_ylim(0, 0.75)
ax1.set_ylabel("Mean JSER (H-Band)")
ax1.set_title("Mean JSER per KL Grade (correct predictions)")
ax1.legend(fontsize=9)
for bar, val in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# Panel 2: CbU rate per class
ax2 = axes[0, 1]
cbu_rates = [per_class[g]["cbu_rate"] * 100 for g in GRADES]
bars2 = ax2.bar(GRADE_NAMES, cbu_rates, color=colors,
                width=0.6, edgecolor="white", linewidth=0.5)
ax2.set_ylabel("Correct-but-Unfaithful Rate (%)")
ax2.set_title("CbU Rate per KL Grade")
ax2.set_ylim(0, 100)
for bar, val in zip(bars2, cbu_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# Panel 3: JSER distribution (violin/box) per class — correct preds only
ax3 = axes[1, 0]
grade_jser_data = [[r["jser_hband"] for r in correct if r["true_label"] == g]
                   for g in GRADES]
bp = ax3.boxplot(grade_jser_data, tick_labels=[f"KL-{g}" for g in GRADES],
                 patch_artist=True, notch=False,
                 medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.axhline(JSER_THRESHOLD, color="red", linestyle="--",
            linewidth=1.5, label=f"Threshold ({JSER_THRESHOLD})")
ax3.set_ylabel("JSER (H-Band)")
ax3.set_title("JSER Distribution per KL Grade (correct predictions)")
ax3.legend(fontsize=9)

# Panel 4: Confidence vs JSER scatter (correct preds, coloured by grade)
ax4 = axes[1, 1]
for g, color in zip(GRADES, colors):
    g_data = [r for r in correct if r["true_label"] == g]
    ax4.scatter([r["confidence"] for r in g_data],
                [r["jser_hband"]  for r in g_data],
                alpha=0.3, s=10, color=color, label=f"KL-{g}")
ax4.axhline(JSER_THRESHOLD, color="black", linestyle="--",
            linewidth=1, label=f"Threshold ({JSER_THRESHOLD})")
ax4.set_xlabel("Prediction Confidence")
ax4.set_ylabel("JSER (H-Band)")
ax4.set_title(f"Confidence vs Faithfulness\n(Pearson r={pearson_r:.3f}, p={pearson_p:.3f})")
ax4.legend(fontsize=8, markerscale=2)

plt.tight_layout()
plt.savefig("data/xai_analysis/baseline_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/xai_analysis/baseline_plots.png")
