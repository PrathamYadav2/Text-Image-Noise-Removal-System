"""
Training Script -- Text Image Noise Removal (U-Net)
====================================================
Usage:
    python train.py
    python train.py --epochs 100 --batch_size 16 --lr 0.0001
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model.unet import UNet, LightUNet, count_parameters


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    """Loads paired (noisy, clean) grayscale image patches."""

    def __init__(self, noisy_dir: Path, clean_dir: Path, img_size: int = 256):
        self.noisy_paths = sorted(noisy_dir.glob("*.png")) + sorted(noisy_dir.glob("*.jpg"))
        self.clean_paths = sorted(clean_dir.glob("*.png")) + sorted(clean_dir.glob("*.jpg"))

        assert len(self.noisy_paths) == len(self.clean_paths), (
            f"Mismatch: {len(self.noisy_paths)} noisy vs {len(self.clean_paths)} clean images"
        )
        assert len(self.noisy_paths) > 0, (
            "No images found! Run: python data/generate_dataset.py first."
        )

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),           # -> [0, 1] float tensor
        ])

    def __len__(self) -> int:
        return len(self.noisy_paths)

    def __getitem__(self, idx: int):
        noisy = Image.open(self.noisy_paths[idx]).convert("L")
        clean = Image.open(self.clean_paths[idx]).convert("L")
        return self.transform(noisy), self.transform(clean)


# ---------------------------------------------------------------------------
# Loss: L1 + SSIM
# ---------------------------------------------------------------------------

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural Similarity Loss (differentiable approximation)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = nn.functional.avg_pool2d(pred,   window_size, 1, window_size // 2)
    mu2 = nn.functional.avg_pool2d(target, window_size, 1, window_size // 2)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.avg_pool2d(pred   * pred,   window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = nn.functional.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu2_sq
    sigma12   = nn.functional.avg_pool2d(pred   * target, window_size, 1, window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """L1 + SSIM loss for sharper text reconstruction."""

    def __init__(self, alpha: float = 0.85):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.l1(pred, target) + (1 - self.alpha) * ssim_loss(pred, target)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    dataset = DenoisingDataset(
        noisy_dir=Path(args.noisy_dir),
        clean_dir=Path(args.clean_dir),
        img_size=args.img_size,
    )
    val_size  = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0)

    print(f"Train: {train_size} | Val: {val_size}")

    # --- Model ---
    if args.light:
        model = LightUNet(in_channels=1, out_channels=1).to(device)
        print("Using LightUNet (CPU-friendly, ~500K params)")
    else:
        model = UNet(in_channels=1, out_channels=1).to(device)
        print("Using full UNet (~31M params)")
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = CombinedLoss().to(device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    # --- Resume if checkpoint exists ---
    best_ckpt = ckpt_dir / "best_model.pth"
    start_epoch = 0
    if best_ckpt.exists() and args.resume:
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.4f}")

    print(f"\nTraining for {args.epochs} epochs...\n{'='*50}")
    t0 = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+args.epochs} [train]", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                val_loss += criterion(pred, clean).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        print(f"Epoch [{epoch+1:3d}] | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ---- Save best ----
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch":     epoch + 1,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  best_val_loss,
            }, best_ckpt)
            print(f"  >> Best model saved (val_loss={best_val_loss:.4f})")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {best_ckpt}")

    # ---- Plot loss curves ----
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    loss_plot = ckpt_dir / "loss_curve.png"
    plt.savefig(loss_plot)
    plt.close()
    print(f"Loss curve saved to {loss_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net document denoiser")
    parser.add_argument("--noisy_dir",      type=str, default="data/noisy")
    parser.add_argument("--clean_dir",      type=str, default="data/clean")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--img_size",       type=int, default=256)
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--light",          action="store_true", default=True,
                        help="Use LightUNet (~500K params) for fast CPU training (default: True)")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume from existing best_model.pth if available")
    args = parser.parse_args()
    train(args)
