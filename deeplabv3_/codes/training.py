import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from tqdm import tqdm
from data_loader import BinarySegDataset  # your dataset class

# ---------------- CONFIG ----------------
IMG_DIR = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/data/raw_data/images"
MASK_DIR = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/data/raw_data/masks"
SAVE_DIR = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/models"

BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
NUM_CLASSES = 2  # 0 = background, 1 = foreground
VAL_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = True

# ---------------- DATA ----------------
dataset = BinarySegDataset(IMG_DIR, MASK_DIR)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- MODEL ----------------
model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, 1)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)
best_val_loss = float('inf')

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    # -------- TRAIN --------
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
    for imgs, masks in loop:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
            preds = model(imgs)["out"]
            loss = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=total_loss/(loop.n+1))

    avg_train_loss = total_loss / len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    total_correct = 0
    total_pixels = 0
    total_inter = 0
    total_union = 0
    total_dice = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)["out"]
            loss = criterion(preds, masks)
            val_loss += loss.item()

            pred_mask = torch.argmax(preds, dim=1)  # [N,H,W]

            # Pixel accuracy
            total_correct += (pred_mask == masks).sum().item()
            total_pixels += masks.numel()

            # IoU
            intersection = ((pred_mask == 1) & (masks == 1)).sum().item()
            union = ((pred_mask == 1) | (masks == 1)).sum().item()
            total_inter += intersection
            total_union += union

            # Dice coefficient
            dice = (2 * intersection) / (pred_mask.sum().item() + masks.sum().item() + 1e-6)
            total_dice += dice

    avg_val_loss = val_loss / len(val_loader)
    pixel_acc = total_correct / total_pixels
    iou = total_inter / (total_union + 1e-6)
    dice_coeff = total_dice / len(val_loader)

    scheduler.step(avg_val_loss)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Pixel Acc: {pixel_acc:.4f} | "
          f"IoU: {iou:.4f} | "
          f"Dice: {dice_coeff:.4f}\n")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/models/deeplab_mobilenetv3_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved Best Model: {save_path}")
