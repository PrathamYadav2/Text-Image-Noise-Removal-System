"""
Flask Web App - Text Image Noise Removal
=========================================
Upload a noisy scanned image -> get a clean, high-definition version back.

Usage:
    python app.py
    Then open: http://localhost:5000
"""

import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = Path("static/uploads")
RESULT_FOLDER = Path("static/results")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def denoise_image(img: np.ndarray) -> np.ndarray:
    """
    Ultimate document restoration pipeline:
      1. 2x Lanczos super-resolution to reconstruct sub-pixel font curves
      2. Background illumination modeling via morphological closing
      3. Division normalization to eliminate uneven shadows & paper texture
      4. Non-local means denoising with bilateral edge-preserving filter
      5. Optimal binarization for solid, unbroken letter strokes
      6. Morphological ellipse hole filling inside letters (repairs salt noise inside strokes)
      7. Connected-component scanner margin artifact removal
      8. Anti-aliased typography smoothing for natural, professional document text
    """
    h, w = img.shape[:2]

    # Step 1: Sub-pixel resolution enhancement (Lanczos-4 sinc interpolation)
    up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    if len(up.shape) == 3:
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    else:
        gray = up.copy()

    # Invert if text is white on dark background
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Step 2: Extract illumination surface
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Step 3: Division normalization (flattens paper background to 255 pure white)
    norm = cv2.divide(gray, bg, scale=255)

    # Step 4: Denoising & edge smoothing
    denoised = cv2.fastNlMeansDenoising(norm, None, h=10, templateWindowSize=7, searchWindowSize=21)
    bilateral = cv2.bilateralFilter(denoised, d=7, sigmaColor=40, sigmaSpace=40)

    # Step 5: High-resolution binarization
    _, binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Fill internal noise pits inside letters using ellipse morphological closing
    inv = cv2.bitwise_not(binary)
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled_inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k_fill)

    # Step 7: Remove thin scanner margin lines
    H, W = filled_inv.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_inv, connectivity=8)
    for i in range(1, num_labels):
        comp_x, comp_y, comp_w, comp_h, _ = stats[i]
        if comp_w <= 4 and (comp_x < 50 or comp_x > W - 50):
            filled_inv[labels == i] = 0
        if comp_h <= 4 and (comp_y < 40 or comp_y > H - 40):
            filled_inv[labels == i] = 0

    # Step 8: Anti-aliased typography smoothing for natural, professional font curves
    clean_bin = cv2.bitwise_not(filled_inv)
    smooth_final = cv2.GaussianBlur(clean_bin, (3, 3), 0)

    return smooth_final


def process(img_path: Path) -> Path:
    """Read image, denoise it, save and return result path."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError("Cannot read image: " + str(img_path))

    cleaned = denoise_image(img)

    result_name = img_path.stem + "_cleaned.png"
    result_path = RESULT_FOLDER / result_name
    cv2.imwrite(str(result_path), cleaned)
    return result_path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", checkpoint_ready=True)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported format. Use: PNG, JPG, BMP, TIFF"}), 400

    uid = uuid.uuid4().hex[:8]
    suffix = Path(file.filename).suffix or ".png"
    upload_path = UPLOAD_FOLDER / f"{uid}{suffix}"
    file.save(upload_path)

    try:
        result_path = process(upload_path)
    except Exception as e:
        return jsonify({"error": "Processing failed: " + str(e)}), 500

    return jsonify({
        "noisy_url":   url_for("static", filename="uploads/" + upload_path.name),
        "cleaned_url": url_for("static", filename="results/" + result_path.name),
        "filename":    result_path.name,
    })


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    print("Open: http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
