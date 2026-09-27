"""
Evaluate — PSNR & SSIM metrics on test images
==============================================
Usage:
    python evaluate.py
    python evaluate.py --noisy_dir data/noisy --clean_dir data/clean --num_samples 50
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm import tqdm

from model.unet import UNet


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    model = UNet(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Run: python train.py  first.")
        return

    model = load_model(Path(args.checkpoint), device)
    print(f"✅ Model loaded from {args.checkpoint}")

    noisy_paths = sorted(Path(args.noisy_dir).glob("*.png")) + \
                  sorted(Path(args.noisy_dir).glob("*.jpg"))
    clean_paths = sorted(Path(args.clean_dir).glob("*.png")) + \
                  sorted(Path(args.clean_dir).glob("*.jpg"))

    if len(noisy_paths) == 0:
        print("❌ No images found. Run: python data/generate_dataset.py  first.")
        return

    # Sample a subset for evaluation
    pairs = list(zip(noisy_paths, clean_paths))
    if args.num_samples < len(pairs):
        pairs = random.sample(pairs, args.num_samples)

    to_tensor = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    psnr_noisy_list, psnr_clean_list = [], []
    ssim_noisy_list, ssim_clean_list = [], []

    for noisy_path, clean_path in tqdm(pairs, desc="Evaluating"):
        noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
        if noisy_img is None or clean_img is None:
            continue

        # Resize to same size
        noisy_img = cv2.resize(noisy_img, (args.img_size, args.img_size))
        clean_img = cv2.resize(clean_img, (args.img_size, args.img_size))

        # Run model
        inp = to_tensor(Image.fromarray(noisy_img)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
        pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Metrics: noisy vs clean (baseline)
        psnr_noisy_list.append(psnr(clean_img, noisy_img, data_range=255))
        ssim_noisy_list.append(ssim(clean_img, noisy_img, data_range=255))

        # Metrics: model output vs clean
        psnr_clean_list.append(psnr(clean_img, pred_np, data_range=255))
        ssim_clean_list.append(ssim(clean_img, pred_np, data_range=255))

    print("\n" + "=" * 50)
    print(f"{'Metric':<20} {'Noisy (baseline)':>18} {'Model Output':>14}")
    print("-" * 50)
    print(f"{'PSNR (dB)':<20} {np.mean(psnr_noisy_list):>18.2f} {np.mean(psnr_clean_list):>14.2f}")
    print(f"{'SSIM':<20} {np.mean(ssim_noisy_list):>18.4f} {np.mean(ssim_clean_list):>14.4f}")
    print("=" * 50)
    print(f"\nSamples evaluated: {len(psnr_clean_list)}")

    improvement_psnr = np.mean(psnr_clean_list) - np.mean(psnr_noisy_list)
    improvement_ssim = np.mean(ssim_clean_list) - np.mean(ssim_noisy_list)
    print(f"PSNR improvement : +{improvement_psnr:.2f} dB")
    print(f"SSIM improvement : +{improvement_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate denoiser PSNR/SSIM")
    parser.add_argument("--noisy_dir",   type=str, default="data/noisy")
    parser.add_argument("--clean_dir",   type=str, default="data/clean")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    evaluate(args)
