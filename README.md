# Text Image Noise Removal System

A deep-learning pipeline that **cleans noisy scanned text images** using a **U-Net** model.
Upload a dirty scan → get back a sharp, readable document image.

---

## 🗂️ Project Structure

```
text-denoiser/
├── data/
│   ├── clean/                  ← Clean ground-truth images (generated/downloaded)
│   ├── noisy/                  ← Noisy input images (generated)
│   └── generate_dataset.py     ← Synthetic dataset generator
├── model/
│   ├── __init__.py
│   └── unet.py                 ← U-Net architecture
├── train.py                    ← Training loop
├── inference.py                ← CLI: denoise a single image
├── app.py                      ← Flask web app
├── evaluate.py                 ← PSNR / SSIM metrics
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA from https://pytorch.org/get-started/locally/

---

## 🗄️ Dataset

### Option A — Generate Synthetic Dataset (quickest way to start)

```bash
python data/generate_dataset.py --num 500
```

This creates **500 paired** (clean, noisy) 256×256 grayscale images in `data/clean/` and `data/noisy/`.

Noise types applied:
- Gaussian noise
- Salt & pepper noise
- Ink blots / coffee stains
- Gaussian blur (scanner defocus)
- JPEG compression artifacts
- Random scan lines / scratches
- Paper background texture

### Option B — Use Real Public Datasets

| Dataset | Size | Link |
|---|---|---|
| **NoisyOffice** (UCI) | 72 pairs | https://archive.ics.uci.edu/dataset/318/noisyoffice |
| **Kaggle — Denoising Dirty Documents** | 144 pairs | https://www.kaggle.com/c/denoising-dirty-documents |
| **DIBCO benchmark** | ~100 pairs/year | http://utopia.duth.gr/~ipratika/DIBCO2019/benchmark/ |

Download and place:
- Clean images → `data/clean/`
- Noisy images → `data/noisy/`

---

## 🧠 Train the Model

```bash
python train.py
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--img_size` | 256 | Image resolution |
| `--resume` | off | Resume from last checkpoint |

**Example — train for 100 epochs with larger batch:**
```bash
python train.py --epochs 100 --batch_size 16
```

Best model is saved to `checkpoints/best_model.pth`.
Loss curve saved to `checkpoints/loss_curve.png`.

---

## 🔍 Inference (CLI)

Clean a single image:

```bash
python inference.py --input path/to/noisy_scan.jpg --output cleaned.jpg
```

Show side-by-side comparison window:

```bash
python inference.py --input noisy_scan.jpg --output cleaned.jpg --show
```

---

## 🌐 Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

- Drag & drop your noisy image
- See **before / after** side-by-side
- Download the cleaned result

---

## 📊 Evaluate (PSNR / SSIM)

```bash
python evaluate.py
```

Output example:
```
==================================================
Metric               Noisy (baseline)   Model Output
--------------------------------------------------
PSNR (dB)                        22.14          30.87
SSIM                             0.6812         0.9143
==================================================
PSNR improvement : +8.73 dB
SSIM improvement : +0.2331
```

---

## 🏗️ Architecture

```
Input (1×256×256 grayscale)
 ├─ Encoder Block 1  →  64 channels
 ├─ Encoder Block 2  → 128 channels
 ├─ Encoder Block 3  → 256 channels
 ├─ Encoder Block 4  → 512 channels
 ├─ Bottleneck       → 1024 channels
 ├─ Decoder Block 4  ←─ skip from Enc4
 ├─ Decoder Block 3  ←─ skip from Enc3
 ├─ Decoder Block 2  ←─ skip from Enc2
 ├─ Decoder Block 1  ←─ skip from Enc1
Output (1×256×256)  sigmoid → [0, 1]
```

**Loss:** `0.85 × L1 + 0.15 × (1 − SSIM)` — preserves both pixel accuracy and structural sharpness.

**Parameters:** ~31 million (base_features=64)

---

## 📋 Quick Workflow Summary

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate dataset
python data/generate_dataset.py --num 500

# 3. Train
python train.py --epochs 50

# 4. Evaluate
python evaluate.py

# 5. Denoise your own image
python inference.py --input my_scan.jpg --output cleaned.jpg

# OR use the web app
python app.py
```
