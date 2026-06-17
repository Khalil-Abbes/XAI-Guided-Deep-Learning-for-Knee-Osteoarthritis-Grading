"""
06_comparison.py
-----------------
Loads baseline and XAI-guided Grad-CAM results and produces the final
accuracy + faithfulness comparison for the paper.

Outputs:
  - Console: full side-by-side comparison table
  - data/xai_analysis/comparison_table.json   (paper Table 1)
  - data/xai_analysis/comparison_plots.png    (4-panel figure)
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── 1. Load both result sets ───────────────────────────────────────────────────
with open("data/xai_analysis/val_gradcam_results.json") as f:
    baseline = json.load(f)

with open("data/xai_analysis/val_gradcam_results_xai.json") as f:
    xai = json.load(f)

JSER_THRESHOLD = 0.40
GRADES         = [0, 1, 2, 3, 4]
GRADE_NAMES    = ["KL-0\n(Healthy)", "KL-1\n(Doubtful)", "KL-2\n(Mild)",
                  "KL-3\n(Moderate)", "KL-4\n(Severe)"]
COLORS         = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]

# ── 2. Helper — compute all metrics for one result set ────────────────────────
def compute_metrics(results):
    correct    = [r for r in results if r["correct"]]
    unfaithful = [r for r in correct if r["jser_hband"] < JSER_THRESHOLD]

    global_metrics = {
        "accuracy":          len(correct) / len(results),
        "mean_jser":         float(np.mean([r["jser_hband"] for r in results])),
        "mean_jser_correct": float(np.mean([r["jser_hband"] for r in correct])),
        "std_jser_correct":  float(np.std([r["jser_hband"]  for r in correct])),
        "cbu_rate":          len(unfaithful) / len(correct),
        "faithfulness_score": 1 - len(unfaithful) / len(correct),
        "pearson_r":         float(stats.pearsonr(
                                [r["confidence"] for r in correct],
                                [r["jser_hband"]  for r in correct])[0]),
    }

    per_class = {}
    for g in GRADES:
        g_correct    = [r for r in correct    if r["true_label"] == g]
        g_all        = [r for r in results    if r["true_label"] == g]
        g_unfaithful = [r for r in unfaithful if r["true_label"] == g]
        per_class[g] = {
            "accuracy":     len(g_correct) / len(g_all) if g_all else 0,
            "mean_jser":    float(np.mean([r["jser_hband"] for r in g_correct])) if g_correct else 0,
            "std_jser":     float(np.std([r["jser_hband"]  for r in g_correct])) if g_correct else 0,
            "cbu_rate":     len(g_unfaithful) / len(g_correct) if g_correct else 0,
            "n_correct":    len(g_correct),
            "n_unfaithful": len(g_unfaithful),
        }

    return global_metrics, per_class

m_base, pc_base = compute_metrics(baseline)
m_xai,  pc_xai  = compute_metrics(xai)

# ── 3. Console report ─────────────────────────────────────────────────────────
def delta(new, old, pct=False):
    d = new - old
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{sign}{d*100:.1f}pp"
    return f"{sign}{d:.4f}"

print()
print("=" * 70)
print("FINAL COMPARISON — BASELINE vs XAI-GUIDED (H-Band JSER 105-160)")
print("=" * 70)
print(f"\n{'Metric':<30} {'Baseline':>12} {'XAI-Guided':>12} {'Delta':>10}")
print("-" * 66)
print(f"{'Val Accuracy':<30} {m_base['accuracy']*100:>11.1f}% "
      f"{m_xai['accuracy']*100:>11.1f}% "
      f"{delta(m_xai['accuracy'], m_base['accuracy'], pct=True):>10}")
print(f"{'Faithfulness Score':<30} {m_base['faithfulness_score']*100:>11.1f}% "
      f"{m_xai['faithfulness_score']*100:>11.1f}% "
      f"{delta(m_xai['faithfulness_score'], m_base['faithfulness_score'], pct=True):>10}")
print(f"{'CbU Rate':<30} {m_base['cbu_rate']*100:>11.1f}% "
      f"{m_xai['cbu_rate']*100:>11.1f}% "
      f"{delta(m_xai['cbu_rate'], m_base['cbu_rate'], pct=True):>10}")
print(f"{'Mean JSER (correct)':<30} {m_base['mean_jser_correct']:>12.4f} "
      f"{m_xai['mean_jser_correct']:>12.4f} "
      f"{delta(m_xai['mean_jser_correct'], m_base['mean_jser_correct']):>10}")
print(f"{'Std JSER (correct)':<30} {m_base['std_jser_correct']:>12.4f} "
      f"{m_xai['std_jser_correct']:>12.4f} "
      f"{delta(m_xai['std_jser_correct'], m_base['std_jser_correct']):>10}")
print(f"{'Conf-JSER Pearson r':<30} {m_base['pearson_r']:>12.4f} "
      f"{m_xai['pearson_r']:>12.4f} "
      f"{delta(m_xai['pearson_r'], m_base['pearson_r']):>10}")

print(f"\n{'Per-Class Accuracy':}")
print(f"  {'Grade':<10} {'Base Acc':>10} {'XAI Acc':>10} {'Delta':>10} "
      f"{'Base CbU':>10} {'XAI CbU':>10} {'Delta':>10}")
print(f"  {'-'*62}")
for g in GRADES:
    print(f"  KL-{g}      "
          f"{pc_base[g]['accuracy']*100:>9.1f}% "
          f"{pc_xai[g]['accuracy']*100:>9.1f}% "
          f"{delta(pc_xai[g]['accuracy'], pc_base[g]['accuracy'], pct=True):>10}  "
          f"{pc_base[g]['cbu_rate']*100:>8.1f}% "
          f"{pc_xai[g]['cbu_rate']*100:>8.1f}% "
          f"{delta(pc_xai[g]['cbu_rate'], pc_base[g]['cbu_rate'], pct=True):>10}")

# ── 4. Save comparison JSON ────────────────────────────────────────────────────
comparison = {
    "baseline":    {"global": m_base, "per_class": pc_base},
    "xai_guided":  {"global": m_xai,  "per_class": pc_xai},
}
with open("data/xai_analysis/comparison_table.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("\nSaved: data/xai_analysis/comparison_table.json")

# ── 5. 4-panel comparison figure ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Baseline vs XAI-Guided — Accuracy & Faithfulness Comparison",
             fontsize=13, fontweight="bold")

x      = np.arange(len(GRADES))
width  = 0.35

# Panel 1: Per-class accuracy
ax1 = axes[0, 0]
ax1.bar(x - width/2, [pc_base[g]["accuracy"]*100 for g in GRADES],
        width, label="Baseline", color="#90CAF9", edgecolor="white")
ax1.bar(x + width/2, [pc_xai[g]["accuracy"]*100  for g in GRADES],
        width, label="XAI-Guided", color="#1565C0", edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels([f"KL-{g}" for g in GRADES])
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Per-Class Accuracy")
ax1.set_ylim(0, 100)
ax1.legend()
ax1.yaxis.set_major_formatter(mticker.PercentFormatter())

# Panel 2: Per-class CbU rate
ax2 = axes[0, 1]
ax2.bar(x - width/2, [pc_base[g]["cbu_rate"]*100 for g in GRADES],
        width, label="Baseline",   color="#EF9A9A", edgecolor="white")
ax2.bar(x + width/2, [pc_xai[g]["cbu_rate"]*100  for g in GRADES],
        width, label="XAI-Guided", color="#B71C1C", edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels([f"KL-{g}" for g in GRADES])
ax2.set_ylabel("CbU Rate (%)")
ax2.set_title("Per-Class Correct-but-Unfaithful Rate")
ax2.set_ylim(0, 100)
ax2.legend()
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

# Panel 3: Per-class mean JSER
ax3 = axes[1, 0]
ax3.bar(x - width/2, [pc_base[g]["mean_jser"] for g in GRADES],
        width, label="Baseline",   color="#A5D6A7", edgecolor="white",
        yerr=[pc_base[g]["std_jser"] for g in GRADES], capsize=4)
ax3.bar(x + width/2, [pc_xai[g]["mean_jser"]  for g in GRADES],
        width, label="XAI-Guided", color="#1B5E20", edgecolor="white",
        yerr=[pc_xai[g]["std_jser"]  for g in GRADES], capsize=4)
ax3.axhline(JSER_THRESHOLD, color="black", linestyle="--",
            linewidth=1.5, label=f"Threshold ({JSER_THRESHOLD})")
ax3.set_xticks(x)
ax3.set_xticklabels([f"KL-{g}" for g in GRADES])
ax3.set_ylabel("Mean JSER (H-Band)")
ax3.set_title("Per-Class Mean JSER ± Std")
ax3.set_ylim(0, 0.75)
ax3.legend()

# Panel 4: Global metrics bar chart
ax4 = axes[1, 1]
global_labels  = ["Val\nAccuracy", "Faithfulness\nScore", "Mean\nJSER"]
base_vals      = [m_base["accuracy"]*100,
                  m_base["faithfulness_score"]*100,
                  m_base["mean_jser_correct"]*100]
xai_vals       = [m_xai["accuracy"]*100,
                  m_xai["faithfulness_score"]*100,
                  m_xai["mean_jser_correct"]*100]
xg = np.arange(len(global_labels))
ax4.bar(xg - width/2, base_vals, width, label="Baseline",   color="#90CAF9", edgecolor="white")
ax4.bar(xg + width/2, xai_vals,  width, label="XAI-Guided", color="#1565C0", edgecolor="white")
ax4.set_xticks(xg)
ax4.set_xticklabels(global_labels)
ax4.set_ylabel("Value (%)")
ax4.set_title("Global Metrics Overview")
ax4.set_ylim(0, 100)
ax4.legend()
for i, (bv, xv) in enumerate(zip(base_vals, xai_vals)):
    ax4.text(i - width/2, bv + 0.5, f"{bv:.1f}", ha="center", va="bottom", fontsize=8)
    ax4.text(i + width/2, xv + 0.5, f"{xv:.1f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("data/xai_analysis/comparison_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/xai_analysis/comparison_plots.png")