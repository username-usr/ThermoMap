import os
import random
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================
IMAGE_DIR = "feet_seg_dataset/images"
MASK_DIR  = "feet_seg_dataset/masks"

IMG_SIZE  = 256
BATCH_SIZE = 4
LR         = 1e-3
EPOCHS     = 30
VAL_SPLIT  = 0.2
RANDOM_SEED = 42
MODEL_OUT  = "feet_unet.pth"

# =========================================================
# DATASET
# =========================================================
class FeetSegDataset(Dataset):
    def __init__(self, image_paths, image_dir, mask_dir, img_size=256, augment=False):
        self.image_paths = image_paths
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # infer mask name: <stem>_mask.png
        stem, _ = os.path.splitext(img_name)
        mask_name = f"{stem}_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # read
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # resize to fixed size
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # normalize image to [0,1]
        img = img.astype(np.float32) / 255.0

        # binarize mask: 0 or 1
        mask = (mask > 127).astype(np.float32)

        # simple augmentation (horizontal flip)
        if self.augment and random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # to tensors: (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0)      # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)

        return img, mask

# =========================================================
# MODEL: SMALL UNET
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        self.down1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        # encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # bottleneck
        bn = self.bottleneck(p4)

        # decoder
        u4 = self.up4(bn)
        u4 = torch.cat([u4, c4], dim=1)
        c4 = self.conv4(u4)

        u3 = self.up3(c4)
        u3 = torch.cat([u3, c3], dim=1)
        c3 = self.conv3(u3)

        u2 = self.up2(c3)
        u2 = torch.cat([u2, c2], dim=1)
        c2 = self.conv2(u2)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, c1], dim=1)
        c1 = self.conv1(u1)

        out = self.out_conv(c1)
        return out

# =========================================================
# LOSS FUNCTIONS
# =========================================================
bce_loss_fn = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-7):
    """
    logits: (B,1,H,W) raw outputs
    targets: (B,1,H,W) 0/1
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

def combined_loss(logits, targets):
    return bce_loss_fn(logits, targets) + dice_loss(logits, targets)

# =========================================================
# TRAIN / VAL LOOPS
# =========================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)

def eval_one_epoch(model, loader, device):
    model.eval()
    loss_sum = 0.0
    dice_sum = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = combined_loss(logits, masks)
            loss_sum += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            inter = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2 * inter + 1e-7) / (union + 1e-7)
            dice_sum += dice.mean().item() * imgs.size(0)

    avg_loss = loss_sum / len(loader.dataset)
    avg_dice = dice_sum / len(loader.dataset)
    return avg_loss, avg_dice

# =========================================================
# MAIN
# =========================================================
def main():
    # fix seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # collect image names
    img_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        img_files.extend([os.path.basename(p) for p in glob(os.path.join(IMAGE_DIR, ext))])

    img_files = sorted(list(set(img_files)))
    print(f"Total images found: {len(img_files)}")

    if len(img_files) == 0:
        print("No images found in", IMAGE_DIR)
        return

    # split train/val
    train_imgs, val_imgs = train_test_split(
        img_files,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED
    )

    print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")

    # datasets & loaders
    train_ds = FeetSegDataset(train_imgs, IMAGE_DIR, MASK_DIR, IMG_SIZE, augment=True)
    val_ds   = FeetSegDataset(val_imgs,   IMAGE_DIR, MASK_DIR, IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # model, optimizer
    model = UNetSmall(in_ch=1, out_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = eval_one_epoch(model, val_loader, device)

        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}")

        # save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  -> New best model saved to {MODEL_OUT}")

    print("\nTraining complete.")
    print("Best validation Dice:", best_val_dice)

if __name__ == "__main__":
    main()
