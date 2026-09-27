import streamlit as st
import cv2
import numpy as np

# Page configuration
st.set_page_config(page_title="Text Image Noise Removal", page_icon="🧹", layout="wide")

st.title("🧹 Text Image Noise Removal System")
st.write("Upload a noisy or dirty scanned text image to clean and enhance it instantly.")

def denoise_image(img_bytes):
    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    # 2x sub-pixel resolution enhancement
    up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Background illumination modeling
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(norm, None, h=10, templateWindowSize=7, searchWindowSize=21)
    bilateral = cv2.bilateralFilter(denoised, d=7, sigmaColor=40, sigmaSpace=40)

    # Binarization
    _, binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill internal noise pits inside letters
    inv = cv2.bitwise_not(binary)
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled_inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k_fill)

    # Remove scanner margin lines
    H, W = filled_inv.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_inv, connectivity=8)
    for i in range(1, num_labels):
        comp_x, comp_y, comp_w, comp_h, _ = stats[i]
        if comp_w <= 4 and (comp_x < 50 or comp_x > W - 50):
            filled_inv[labels == i] = 0
        if comp_h <= 4 and (comp_y < 40 or comp_y > H - 40):
            filled_inv[labels == i] = 0

    clean_bin = cv2.bitwise_not(filled_inv)
    smooth_final = cv2.GaussianBlur(clean_bin, (3, 3), 0)
    
    # Encode back to PNG
    _, buf = cv2.imencode(".png", smooth_final)
    return img, smooth_final, buf.tobytes()

uploaded_file = st.file_uploader("Choose a noisy document image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    with st.spinner("Cleaning and enhancing document..."):
        original, cleaned, cleaned_bytes = denoise_image(uploaded_file.read())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Noisy Input")
        st.image(original, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("✨ Cleaned Document")
        st.image(cleaned, use_container_width=True)
        st.download_button(
            label="⬇️ Download Cleaned Image",
            data=cleaned_bytes,
            file_name="cleaned_document.png",
            mime="image/png"
        )
