"""
05_xai_evaluation.py
---------------------
Runs the same Grad-CAM + JSER pipeline as 03_gradcam_analysis.py
but on the XAI-guided model (xai_guided_resnet50.pth).

Outputs:
  - data/xai_analysis/val_gradcam_results_xai.json
  - data/xai_analysis/unfaithful_cases_xai_hband.json
  - data/xai_analysis/unfaithful_cases_xai_square.json
  - data/xai_analysis/unfaithful_examples_xai_hband.png
  - data/xai_analysis/unfaithful_examples_xai_square.png
  - Console: full faithfulness report (mirrors 03 output for direct comparison)
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── 1. Reproduce val split — identical to 01, 03, 04 ──────────────────────────
random.seed(42)

DATA_ROOT = "data/kneeosteoarthritis/data"
GRADES    = ["0", "1", "2", "3", "4"]

all_images = []
for g in GRADES:
    folder = os.path.join(DATA_ROOT, g)
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images.append((os.path.join(folder, fname), int(g)))

random.shuffle(all_images)
split    = int(0.8 * len(all_images))
val_data = all_images[split:]
print(f"Val set size: {len(val_data)}")

# ── 2. Load XAI-guided model ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("data/xai_guided_resnet50.pth", map_location=device, weights_only=True))
model.eval().to(device)

# ── 3. Transform ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 4. Grad-CAM ────────────────────────────────────────────────────────────────
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

# ── 5. Region definitions — identical to 03 ───────────────────────────────────
REGIONS = {
    "square": {
        "label":       "Square Centre Crop",
        "row_min": 56,  "row_max": 168,
        "col_min": 56,  "col_max": 168,
        "color":       "red",
        "description": "Rows 56-168, Cols 56-168 (centre 50% of image)"
    },
    "hband": {
        "label":       "Horizontal Band",
        "row_min": 105, "row_max": 160,
        "col_min": 0,   "col_max": 224,
        "color":       "lime",
        "description": "Rows 105-160, Full width (anatomical joint-space band)"
    }
}

JSER_THRESHOLD = 0.40

# ── 6. Main loop ───────────────────────────────────────────────────────────────
results = []

for i, (img_path, true_label) in enumerate(val_data):
    img          = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_label)]
    )[0]

    total_energy = grayscale_cam.sum() + 1e-8
    jser_vals = {}
    for key, r in REGIONS.items():
        region_energy  = grayscale_cam[r["row_min"]:r["row_max"],
                                       r["col_min"]:r["col_max"]].sum()
        jser_vals[key] = float(region_energy / total_energy)

    results.append({
        "path":        img_path,
        "true_label":  true_label,
        "pred_label":  pred_label,
        "confidence":  confidence,
        "jser_square": jser_vals["square"],
        "jser_hband":  jser_vals["hband"],
        "correct":     (pred_label == true_label)
    })

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(val_data)} ...")

print(f"Grad-CAM complete for {len(results)} images.")

# ── 7. Save results ────────────────────────────────────────────────────────────
os.makedirs("data/xai_analysis", exist_ok=True)
with open("data/xai_analysis/val_gradcam_results_xai.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: data/xai_analysis/val_gradcam_results_xai.json")

# ── 8. Faithfulness report — mirrors 03 exactly for direct comparison ──────────
correct_results = [r for r in results if r["correct"]]

for key, region in REGIONS.items():
    jser_key   = f"jser_{key}"
    unfaithful = [r for r in correct_results if r[jser_key] < JSER_THRESHOLD]

    print()
    print("=" * 55)
    print(f"FAITHFULNESS REPORT (XAI-GUIDED) — {region['label'].upper()}")
    print(f"Region: {region['description']}")
    print("=" * 55)
    print(f"Total val images        : {len(results)}")
    print(f"Correct predictions     : {len(correct_results)}  "
          f"({len(correct_results)/len(results)*100:.1f} %)")
    print(f"Unfaithful (CbU) cases  : {len(unfaithful)}  "
          f"({len(unfaithful)/len(correct_results)*100:.1f} % of correct preds)")
    print()

    all_jser     = [r[jser_key] for r in results]
    correct_jser = [r[jser_key] for r in correct_results]
    print(f"Mean JSER (all val)      : {np.mean(all_jser):.4f}")
    print(f"Mean JSER (correct only) : {np.mean(correct_jser):.4f}")
    print(f"Std  JSER (correct only) : {np.std(correct_jser):.4f}")
    print()
    print("Per-class mean JSER (correct predictions):")
    grade_dist = Counter(r["true_label"] for r in unfaithful)
    for g in range(5):
        g_jser         = [r[jser_key] for r in correct_results if r["true_label"] == g]
        n_unfaithful   = grade_dist.get(g, 0)
        total_in_grade = sum(1 for r in correct_results if r["true_label"] == g)
        pct            = n_unfaithful / total_in_grade * 100 if total_in_grade else 0
        mean_j         = np.mean(g_jser) if g_jser else 0
        print(f"  KL-{g}: mean JSER={mean_j:.4f}  "
              f"unfaithful={n_unfaithful}/{total_in_grade} ({pct:.1f} %)")

    out_path = f"data/xai_analysis/unfaithful_cases_xai_{key}.json"
    with open(out_path, "w") as f:
        json.dump(unfaithful, f, indent=2)
    print(f"\nSaved: {out_path}  ({len(unfaithful)} cases)")

# ── 9. Worst-5 unfaithful figures — same format as 03 ─────────────────────────
def overlay_cam(img_path, cam_map):
    bgr     = cv2.resize(cv2.imread(img_path), (224, 224))
    orig    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    heatmap = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    return orig, np.uint8(0.5 * orig + 0.5 * heatmap)

for key, region in REGIONS.items():
    jser_key   = f"jser_{key}"
    unfaithful = [r for r in correct_results if r[jser_key] < JSER_THRESHOLD]
    top5       = sorted(unfaithful, key=lambda x: x[jser_key])[:5]

    if not top5:
        print(f"No unfaithful cases for {key} — skipping figure.")
        continue

    fig, axes = plt.subplots(5, 3, figsize=(12, 22))
    fig.suptitle(
        f"5 Worst Unfaithful Cases (XAI-Guided) — {region['label']}\n"
        f"Region: {region['description']}",
        fontsize=12, fontweight="bold", y=1.01
    )

    for i, case in enumerate(top5):
        inp     = transform(Image.open(case["path"]).convert("RGB")).unsqueeze(0).to(device)
        cam_map = cam(input_tensor=inp,
                      targets=[ClassifierOutputTarget(case["pred_label"])])[0]
        orig, overlay = overlay_cam(case["path"], cam_map)

        axes[i, 0].imshow(orig)
        rect = patches.Rectangle(
            (region["col_min"], region["row_min"]),
            region["col_max"] - region["col_min"],
            region["row_max"] - region["row_min"],
            linewidth=2, edgecolor=region["color"], facecolor="none", linestyle="--"
        )
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f"Original  |  True KL-{case['true_label']}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(
            f"Grad-CAM  |  Pred KL-{case['pred_label']}  |  "
            f"{jser_key.upper()}={case[jser_key]:.3f}", fontsize=9
        )
        axes[i, 1].axis("off")

        bar_labels = ["Square\nCrop", "H-Band"]
        bar_vals   = [case["jser_square"], case["jser_hband"]]
        axes[i, 2].bar(bar_labels, bar_vals, color=["tomato", "limegreen"], width=0.4)
        axes[i, 2].axhline(JSER_THRESHOLD, color="black", linestyle="--",
                            linewidth=1, label=f"Threshold ({JSER_THRESHOLD})")
        axes[i, 2].set_ylim(0, 1)
        axes[i, 2].set_title("JSER Comparison", fontsize=9)
        axes[i, 2].legend(fontsize=7)
        axes[i, 2].set_ylabel("JSER")

    plt.tight_layout()
    out_fig = f"data/xai_analysis/unfaithful_examples_xai_{key}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}")
