"""
CLI Inference — Text Image Noise Removal
=========================================
Cleans a single noisy scanned image using the trained U-Net model.

Usage:
    python inference.py --input noisy_scan.jpg --output cleaned.jpg
    python inference.py --input noisy_scan.jpg --output cleaned.jpg --show
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model.unet import UNet


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load trained U-Net from checkpoint."""
    model = UNet(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and wrapped checkpoint
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denoise_image(model: UNet, img_path: Path, device: torch.device, img_size: int = 256) -> np.ndarray:
    """
    Run the denoising model on a full-resolution image.
    Strategy: resize → infer → resize back to original resolution.
    """
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    orig_h, orig_w = original.shape

    # Pre-process
    to_tensor = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    pil_img = Image.fromarray(original)
    inp = to_tensor(pil_img).unsqueeze(0).to(device)   # (1, 1, H, W)

    # Inference
    with torch.no_grad():
        out = model(inp)

    # Post-process back to numpy uint8
    out_np = out.squeeze().cpu().numpy()               # (H, W) in [0, 1]
    out_np = (out_np * 255).astype(np.uint8)

    # Resize back to original resolution
    out_np = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return out_np


def main():
    parser = argparse.ArgumentParser(description="Denoise a scanned text image with U-Net")
    parser.add_argument("--input",      type=Path, required=True, help="Path to noisy input image")
    parser.add_argument("--output",     type=Path, default=None,  help="Path to save cleaned image")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--img_size",   type=int,  default=256,   help="Processing resolution")
    parser.add_argument("--show",       action="store_true",       help="Show before/after comparison")
    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Please train first: python train.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Input     : {args.input}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    model = load_model(args.checkpoint, device)
    print("Model loaded ✅")

    # Denoise
    cleaned = denoise_image(model, args.input, device, args.img_size)

    # Save output
    if args.output is None:
        args.output = args.input.parent / (args.input.stem + "_cleaned.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cleaned)
    print(f"Saved     : {args.output}")

    # Optional: side-by-side comparison
    if args.show:
        noisy_img = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
        h = max(noisy_img.shape[0], cleaned.shape[0])
        panel = np.hstack([
            cv2.copyMakeBorder(noisy_img, 0, h - noisy_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=255),
            np.ones((h, 10), dtype=np.uint8) * 180,   # separator
            cv2.copyMakeBorder(cleaned,   0, h - cleaned.shape[0],   0, 0, cv2.BORDER_CONSTANT, value=255),
        ])
        cv2.imshow("Noisy  |  Cleaned", panel)
        print("Press any key to close…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
