"""
04_xai_guided_training.py
--------------------------
Retrains ResNet-50 with an auxiliary JSER loss that penalises the model
whenever its layer4 activation maps fall outside the anatomical H-Band region.

The JSER loss is computed directly from layer4[-1] activation maps (via a
forward hook) so it stays inside the computation graph and gradients flow
correctly. This replaces the broken pytorch-grad-cam approach.

Total loss: L_total = L_CE + lambda * L_JSER
            L_JSER  = 1 - JSER  (per image, averaged over batch)

Outputs:
  - data/xai_guided_resnet50.pth
  - data/xai_analysis/training_curves_xai.png
  - Console: per-epoch loss breakdown and val accuracy
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# ── 1. Reproducible split — identical to 01, 03, 05 ───────────────────────────
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
split         = int(0.8 * len(all_images))
train_samples = all_images[:split]
val_samples   = all_images[split:]
print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

# ── 2. Dataset ─────────────────────────────────────────────────────────────────
class KneeOADataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}

# ── 3. Transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 4. DataLoaders ─────────────────────────────────────────────────────────────
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels":       torch.tensor([x["labels"]       for x in batch])
    }

train_loader = DataLoader(KneeOADataset(train_samples, train_transform),
                          batch_size=64, shuffle=True,
                          collate_fn=collate_fn, num_workers=0,
                          pin_memory=True)

val_loader   = DataLoader(KneeOADataset(val_samples, val_transform),
                          batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=0,
                          pin_memory=True)

# ── 5. Model with forward hook on layer4[-1] ───────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 5)
model.to(device)

# Forward hook stores the activation map from layer4[-1]
# Shape will be (batch, 2048, 7, 7) — the spatial feature map before GAP
activation_store = {}

def hook_fn(module, input, output):
    activation_store["layer4"] = output  # stays in computation graph

hook = model.layer4[-1].register_forward_hook(hook_fn)

# ── 6. JSER loss from activation maps ─────────────────────────────────────────
# layer4 output: (B, 2048, 7, 7) — 7x7 spatial grid over the 224x224 image
# Each spatial cell covers 224/7 = 32 pixels
# H-Band rows 105-160 in pixel space maps to rows 3-5 in the 7x7 grid
# (105/32 = 3.28 → row 3,  160/32 = 5.0 → row 5)
HBAND_ROW_MIN = 3   # inclusive
HBAND_ROW_MAX = 5   # inclusive (rows 3, 4, 5 of the 7x7 grid)

def compute_jser_loss(activations):
    """
    activations: (B, 2048, 7, 7)
    Returns scalar JSER loss = 1 - mean(JSER) over the batch.
    Fully differentiable — stays in the computation graph.
    """
    # Mean over channels → (B, 7, 7) spatial attention map
    attn = activations.mean(dim=1)
    attn = F.relu(attn)                          # only positive activations

    total_energy = attn.sum(dim=(1, 2)) + 1e-8   # (B,)
    hband_energy = attn[:, HBAND_ROW_MIN:HBAND_ROW_MAX+1, :].sum(dim=(1, 2))  # (B,)

    jser     = hband_energy / total_energy        # (B,)  values in [0, 1]
    loss     = (1.0 - jser).mean()               # scalar
    return loss

# ── 7. Loss, optimiser, scheduler ─────────────────────────────────────────────
label_counts  = Counter([s[1] for s in train_samples])
total         = sum(label_counts.values())
class_weights = torch.tensor(
    [total / (5 * label_counts[i]) for i in range(5)], dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

LAMBDA = 0.5

# ── 8. Training loop ───────────────────────────────────────────────────────────
NUM_EPOCHS = 20
history    = {"train_loss": [], "ce_loss": [], "jser_loss": [], "val_acc": []}

for epoch in range(NUM_EPOCHS):
    model.train()
    running_total = 0.0
    running_ce    = 0.0
    running_jser  = 0.0
    correct       = 0
    total_samples = 0

    for batch in train_loader:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)   # forward hook fires here, fills activation_store

        loss_ce   = criterion(outputs, labels)
        loss_jser = compute_jser_loss(activation_store["layer4"])
        loss      = loss_ce + LAMBDA * loss_jser

        loss.backward()
        optimizer.step()

        running_total += loss.item()      * inputs.size(0)
        running_ce    += loss_ce.item()   * inputs.size(0)
        running_jser  += loss_jser.item() * inputs.size(0)
        correct       += (outputs.argmax(1) == labels).sum().item()
        total_samples += inputs.size(0)

    scheduler.step()

    train_acc   = correct / total_samples
    epoch_total = running_total / total_samples
    epoch_ce    = running_ce    / total_samples
    epoch_jser  = running_jser  / total_samples

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["pixel_values"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)
            out = model(inp)
            val_correct += (out.argmax(1) == lbl).sum().item()
            val_total   += lbl.size(0)
    val_acc = val_correct / val_total

    history["train_loss"].append(epoch_total)
    history["ce_loss"].append(epoch_ce)
    history["jser_loss"].append(epoch_jser)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
          f"Total={epoch_total:.4f}  CE={epoch_ce:.4f}  "
          f"JSER={epoch_jser:.4f} | ValAcc={val_acc:.4f}")

# ── 9. Remove hook, save model ─────────────────────────────────────────────────
hook.remove()
torch.save(model.state_dict(), "data/xai_guided_resnet50.pth")
print("Saved: data/xai_guided_resnet50.pth")

# ── 10. Training curves ────────────────────────────────────────────────────────
os.makedirs("data/xai_analysis", exist_ok=True)
epochs = range(1, NUM_EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("XAI-Guided Training Curves", fontsize=13, fontweight="bold")

axes[0].plot(epochs, history["train_loss"], label="Total Loss")
axes[0].plot(epochs, history["ce_loss"],    label="CE Loss")
axes[0].plot(epochs, history["jser_loss"],  label="JSER Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss Components")
axes[0].legend()

axes[1].plot(epochs, history["val_acc"], color="green", marker="o", markersize=3)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("Validation Accuracy per Epoch")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("data/xai_analysis/training_curves_xai.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/xai_analysis/training_curves_xai.png")

