"""
01_data_load.py
---------------
Downloads the Knee OA dataset from HuggingFace, extracts it, and builds
train/val DataLoaders ready for model training.

Outputs:
  - data/kneeosteoarthritis/data/{0,1,2,3,4}/  (extracted images)
  - Console: class distribution, split sizes, sanity check batch shape
"""


import zipfile
from collections import Counter
from pathlib import Path
import random

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Step 1: Download the raw zip ──────────────────────────────────────────────
zip_path = hf_hub_download(
    repo_id="SilpaCS/kneeosteoarthritis",
    filename="data.zip",
    repo_type="dataset"
)
print(f"Downloaded zip to: {zip_path}")

# ── Step 2: Extract it ────────────────────────────────────────────────────────
extract_dir = Path("data/kneeosteoarthritis")
extract_dir.mkdir(parents=True, exist_ok=True)

if not any(extract_dir.iterdir()):
    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print("Extraction done.")

# ── Step 3: Find the folder that actually contains 0/1/2/3/4 subfolders ───────
# The zip extracts to: data/kneeosteoarthritis/data/0/, /1/, etc.
data_root = extract_dir / "data"
print(f"\nUsing data root: {data_root}")
print("Subfolders found:", [f.name for f in sorted(data_root.iterdir()) if f.is_dir()])

# ── Step 4: Custom Dataset ────────────────────────────────────────────────────
class KneeOADataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples      # list of (Path, int)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}

# ── Step 5: Collect all samples from grade subfolders (0–4) ──────────────────
all_samples = []
for label_folder in sorted(data_root.iterdir()):
    if not label_folder.is_dir():
        continue
    try:
        label = int(label_folder.name)      # folder name is the KL grade 0-4
    except ValueError:
        continue
    for img_path in label_folder.glob("*.png"):
        all_samples.append((img_path, label))
    for img_path in label_folder.glob("*.jpg"):
        all_samples.append((img_path, label))

print(f"\nTotal images found: {len(all_samples)}")
label_dist = Counter([s[1] for s in all_samples])
print(f"Class distribution: {dict(sorted(label_dist.items()))}")

# ── Step 6: Split into train (80%) and val (20%) ──────────────────────────────
# Fixed seed = reproducible split
random.seed(42)
random.shuffle(all_samples)

train_size   = int(0.8 * len(all_samples))
train_samples = all_samples[:train_size]
val_samples   = all_samples[train_size:]
print(f"\nTrain: {len(train_samples)} | Val: {len(val_samples)}")

# ── Step 7: Transforms ────────────────────────────────────────────────────────
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

# ── Step 8: Create Dataset objects ───────────────────────────────────────────
train_dataset = KneeOADataset(train_samples, transform=train_transform)
val_dataset   = KneeOADataset(val_samples,   transform=val_transform)

# ── Step 9: DataLoaders ───────────────────────────────────────────────────────
# num_workers=0 required on Windows
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32,
                          shuffle=False, collate_fn=collate_fn, num_workers=0)

# ── Step 10: Sanity check ─────────────────────────────────────────────────────
batch = next(iter(train_loader))
print(f"\nBatch pixel_values shape: {batch['pixel_values'].shape}")   # should be [32, 3, 224, 224]
print(f"Batch labels:             {batch['labels']}")
print("\nSanity check PASSED ✓")