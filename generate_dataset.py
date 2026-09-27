"""
Synthetic Dataset Generator for Text Image Denoising
=====================================================
Generates paired (clean, noisy) images from clean text images.

Usage:
    python data/generate_dataset.py --source data/clean --output_noisy data/noisy --num 500

If you have no clean images yet, this script will generate simple text images
from scratch using PIL so you can get started immediately.
"""

import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: np.ndarray, std: float = 15.0) -> np.ndarray:
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
    out = img.copy()
    num_pixels = int(amount * img.size)
    # Salt
    coords = [np.random.randint(0, d, num_pixels) for d in img.shape]
    out[tuple(coords)] = 255
    # Pepper
    coords = [np.random.randint(0, d, num_pixels) for d in img.shape]
    out[tuple(coords)] = 0
    return out


def add_blur(img: np.ndarray, ksize: int = None) -> np.ndarray:
    if ksize is None:
        ksize = random.choice([3, 5])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_ink_blots(img: np.ndarray, n_blots: int = None) -> np.ndarray:
    """Simulate ink stains / coffee spots."""
    out = img.copy()
    h, w = out.shape
    n_blots = n_blots or random.randint(1, 5)
    for _ in range(n_blots):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(3, 25)
        color = random.randint(0, 80)          # dark blot
        alpha = random.uniform(0.3, 0.8)
        mask = np.zeros_like(out, dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        # Feather edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        out = (out * (1 - alpha * mask) + color * alpha * mask).astype(np.uint8)
    return out


def add_background_texture(img: np.ndarray, intensity: float = 0.1) -> np.ndarray:
    """Overlay paper-grain texture."""
    texture = np.random.uniform(200, 255, img.shape).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (21, 21), 0)
    out = img.astype(np.float32) * (1 - intensity) + texture * intensity
    return np.clip(out, 0, 255).astype(np.uint8)


def add_jpeg_artifacts(img: np.ndarray, quality: int = None) -> np.ndarray:
    quality = quality or random.randint(20, 60)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def add_random_lines(img: np.ndarray, n: int = None) -> np.ndarray:
    """Simulate scan lines / scratches."""
    out = img.copy()
    h, w = out.shape
    n = n or random.randint(1, 4)
    for _ in range(n):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        color = random.randint(0, 100)
        thickness = random.randint(1, 2)
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def apply_random_noise(img: np.ndarray) -> np.ndarray:
    """Apply a random combination of noise types."""
    transforms = [
        lambda x: add_gaussian_noise(x, std=random.uniform(5, 25)),
        lambda x: add_salt_pepper(x, amount=random.uniform(0.005, 0.03)),
        lambda x: add_blur(x),
        lambda x: add_ink_blots(x),
        lambda x: add_background_texture(x, intensity=random.uniform(0.05, 0.2)),
        lambda x: add_jpeg_artifacts(x),
        lambda x: add_random_lines(x),
    ]
    # Always apply at least 2 noise types
    chosen = random.sample(transforms, k=random.randint(2, 4))
    out = img.copy()
    for fn in chosen:
        out = fn(out)
    return out


# ---------------------------------------------------------------------------
# Text image generator (creates clean images if you have none)
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Scanning documents produces noisy images.",
    "Deep learning can restore degraded text.",
    "Python is a versatile programming language.",
    "Image processing improves OCR accuracy.",
    "Noise removal enhances document readability.",
    "Convolutional neural networks learn features.",
    "Encoder-decoder architectures work well here.",
    "This is a sample scanned document page.",
    "Text recognition requires clean input images.",
]


def generate_clean_text_image(size: tuple = (256, 256)) -> np.ndarray:
    """Create a synthetic clean text image."""
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)

    # Try to use a system font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=14)
        except IOError:
            font = ImageFont.load_default()

    # Draw several lines of random text
    y = 10
    for _ in range(random.randint(5, 12)):
        line = random.choice(SAMPLE_TEXTS)
        # Randomly shorten
        start = random.randint(0, max(0, len(line) - 30))
        line = line[start: start + random.randint(20, 50)]
        draw.text((random.randint(5, 20), y), line, fill=0, font=font)
        y += random.randint(16, 24)
        if y > size[1] - 20:
            break

    return np.array(img)


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    source_dir: Path,
    noisy_dir: Path,
    clean_out_dir: Path,
    num_synthetic: int = 200,
    img_size: int = 256,
):
    """
    Generate paired (clean, noisy) images.
    - If source_dir has images, uses them as clean sources.
    - Always generates `num_synthetic` additional synthetic clean images.
    """
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0

    # --- Process existing clean images if any ---
    existing = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
    if existing:
        print(f"Found {len(existing)} existing clean images in {source_dir}")
        for img_path in tqdm(existing, desc="Processing existing clean images"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            # Save multiple noisy variants per clean image
            for v in range(3):
                noisy = apply_random_noise(img)
                fname = f"img_{idx:05d}.png"
                cv2.imwrite(str(clean_out_dir / fname), img)
                cv2.imwrite(str(noisy_dir / fname), noisy)
                idx += 1

    # --- Generate synthetic text images ---
    print(f"Generating {num_synthetic} synthetic paired images...")
    for _ in tqdm(range(num_synthetic), desc="Synthetic generation"):
        clean = generate_clean_text_image((img_size, img_size))
        noisy = apply_random_noise(clean)
        fname = f"img_{idx:05d}.png"
        cv2.imwrite(str(clean_out_dir / fname), clean)
        cv2.imwrite(str(noisy_dir / fname), noisy)
        idx += 1

    print(f"\nDataset ready: {idx} pairs")
    print(f"   Clean  -> {clean_out_dir}")
    print(f"   Noisy  -> {noisy_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic noisy/clean document image pairs")
    parser.add_argument("--source",       type=Path, default=Path("data/clean"),
                        help="Directory with existing clean images (optional)")
    parser.add_argument("--output_clean", type=Path, default=Path("data/clean"),
                        help="Where to save/copy clean images for training")
    parser.add_argument("--output_noisy", type=Path, default=Path("data/noisy"),
                        help="Where to save noisy images")
    parser.add_argument("--num",          type=int,  default=500,
                        help="Number of synthetic pairs to generate (default: 500)")
    parser.add_argument("--size",         type=int,  default=256,
                        help="Image size in pixels (default: 256)")
    args = parser.parse_args()

    generate_dataset(
        source_dir=args.source,
        noisy_dir=args.output_noisy,
        clean_out_dir=args.output_clean,
        num_synthetic=args.num,
        img_size=args.size,
    )
