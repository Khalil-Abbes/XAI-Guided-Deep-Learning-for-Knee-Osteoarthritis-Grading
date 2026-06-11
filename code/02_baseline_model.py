"""
02_baseline_model.py
---------------------
Trains a ResNet-50 baseline classifier on the Knee OA dataset (5 KL grades).
Uses ImageNet pre-trained weights, class-weighted cross-entropy loss to handle
class imbalance, and a StepLR scheduler over 20 epochs.

Outputs:
  - data/baseline_resnet50.pth       (saved model weights)
  - Console: per-epoch loss/accuracy and final validation accuracy
"""

import random
import zipfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Dataset class ─────────────────────────────────────────────────────────────
class KneeOADataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}

# ── Load & split data ─────────────────────────────────────────────────────────
zip_path = hf_hub_download(
    repo_id="SilpaCS/kneeosteoarthritis",
    filename="data.zip",
    repo_type="dataset"
)

extract_dir = Path("data/kneeosteoarthritis")
extract_dir.mkdir(parents=True, exist_ok=True)
if not any(extract_dir.iterdir()):
    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

data_root = extract_dir / "data"

all_samples = []
for label_folder in sorted(data_root.iterdir()):
    if not label_folder.is_dir():
        continue
    try:
        label = int(label_folder.name)
    except ValueError:
        continue
    for img_path in label_folder.glob("*.png"):
        all_samples.append((img_path, label))
    for img_path in label_folder.glob("*.jpg"):
        all_samples.append((img_path, label))

random.seed(42)
random.shuffle(all_samples)
train_size    = int(0.8 * len(all_samples))
train_samples = all_samples[:train_size]
val_samples   = all_samples[train_size:]
print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── DataLoaders ───────────────────────────────────────────────────────────────
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }

train_loader = DataLoader(KneeOADataset(train_samples, train_transform),
                          batch_size=64, shuffle=True,
                          collate_fn=collate_fn, num_workers=0,
                          pin_memory=True)
val_loader   = DataLoader(KneeOADataset(val_samples, val_transform),
                          batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=0,
                          pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_baseline_model(num_classes=5):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_baseline_model().to(device)
print(f"Using device: {device}")

# ── Class weights (handle imbalance) ─────────────────────────────────────────
label_counts = Counter([s[1] for s in train_samples])
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / (5 * label_counts[i]) for i in range(5)],
    dtype=torch.float
).to(device)
print(f"Class weights: {class_weights}")

# ── Training setup ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

NUM_EPOCHS = 20

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct, total_samples = 0.0, 0, 0

    for batch in train_loader:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * inputs.size(0)
        correct       += (outputs.argmax(1) == labels).sum().item()
        total_samples += inputs.size(0)

    scheduler.step()
    train_acc = correct / total_samples
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {running_loss/total_samples:.4f} | Acc: {train_acc:.4f}")

torch.save(model.state_dict(), "data/baseline_resnet50.pth")
print("Baseline model saved to data/baseline_resnet50.pth")

# ── Validation ────────────────────────────────────────────────────────────────
model.eval()
correct, total_samples = 0, 0

with torch.no_grad():
    for batch in val_loader:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(inputs)
        correct       += (outputs.argmax(1) == labels).sum().item()
        total_samples += inputs.size(0)

val_acc = correct / total_samples
print(f"Validation Accuracy: {val_acc:.4f}")
