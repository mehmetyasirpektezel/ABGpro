import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageOps
import re
import shutil

# --- CLOUD CONFIG ---
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸")

st.title("🩸 Gobseck ABG Engine")
st.caption("Docent Dr. Pektezel's Professional Clinical Tool")

patterns = {
    'ph': [r'pH', r'p\.H'],
    'pco2': [r'pCO2', r'PCO2'],
    'hco3': [r'cHCO3', r'HCO3\(act\)', r'act\.HCO3', r'HCO3a', r'HCO3-'],
    'po2': [r'pO2', r'PO2'],
    'na': [r'Na\+', r'Sodium', r'Na'],
    'cl': [r'Cl-', r'Chloride', r'Cl'],
    'lactate': [r'Lactate', r'Lac', r'Laktat'],
    'po4': [r'PO4', r'Phosphate', r'Fosfat']
}

def clean_clinical_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aggressive contrast for thermal paper
    gray = cv2.convertScaleAbs(gray, alpha=1.9, beta=-60)
    # Denoise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11)

with st.sidebar:
    st.header("Patient Context")
    age = st.number_input("Age", value=60)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0)

# --- THE NATIVE ANDROID HOOK ---
st.info("Tap 'Browse' -> Select 'Camera' to use your high-res back lens.")
uploaded_file = st.file_uploader("Upload or Take Photo of ABG", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with st.spinner("Processing high-res capture..."):
        # Load and handle orientation
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img) # Corrects photo rotation
        
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = clean_clinical_image(frame)
        
        # OCR with optimized config
        text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6')
        
        d = {k: None for k in patterns.keys()}
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=]*([-+]?\d*\.\d+|\d+)', text, re.IGNORECASE)
                if m: d[key] = float(m.group(1)); break

        if not d['ph']:
            st.error("❌ OCR could not read the values. Ensure the paper is flat and text is sharp.")
            st.image(processed, caption="Machine Vision View (Check if text is clear here)")
        else:
            # Calculations
            ph, pco2, hco3 = d['ph'], d['pco2'], d['hco3']
            na, cl, lac = d.get('na', 140), d.get('cl', 104), d.get('lactate', 1.0)
            po2, po4 = d.get('po2', 90), d.get('po4', 3.0)
            
            aa_grad = (((fio2_pct/100) * 713) - (1.2 * pco2)) - po2
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6

            # Results
            st.success("✅ Analysis Complete")
            col1, col2 = st.columns(2)
            col1.metric("pH", ph)
            col2.metric("pCO2", pco2)
            
            st.divider()
            st.write(f"**Stewart cBE:** {cbe:.2f} mmol/L")
            st.write(f"**A-a Gradient:** {aa_grad:.1f} mmHg")
            st.write(f"**Anion Gap (Corr):** {(na - (cl + hco3)) + (2.5 * (4.0 - alb)):.1f}")
