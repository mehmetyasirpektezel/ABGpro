import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageOps
import re
import shutil

# --- CLOUD CONFIG ---
# Automatically locates the Linux Tesseract engine on the server
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

# --- UI SETUP ---
st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸")

st.title("🩸 Gobseck ABG Engine")
st.caption("Docent Dr. Pektezel's Professional Clinical Tool")

# UPGRADE 1: "Fuzzy" patterns to catch OCR subscript errors (e.g., pCOz, pCO)
patterns = {
    'ph': [r'pH', r'p\.H', r'PH'],
    'pco2': [r'pCO2', r'PCO2', r'pCOz', r'pCO', r'PCO'],
    'hco3': [r'cHCO3', r'HCO3', r'HCO', r'act\.HCO'],
    'po2': [r'pO2', r'PO2', r'pO', r'pOz', r'PO'],
    'na': [r'Na\+', r'Sodium', r'Na', r'NA'],
    'cl': [r'Cl-', r'Chloride', r'Cl', r'CL'],
    'lactate': [r'Lactate', r'Lac', r'Laktat'],
    'po4': [r'PO4', r'Phosphate', r'Fosfat']
}

def clean_clinical_image(image):
    """Aggressive filtering for thermal paper and shadow neutralization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.9, beta=-60)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Patient Context")
    age = st.number_input("Age", value=60)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0)

# --- NATIVE CAPTURE ---
st.info("Tap 'Browse files' -> 'Camera'. Try to keep dark background noise out of the frame.")
uploaded_file = st.file_uploader("Upload or Take Photo of ABG", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with st.spinner("Decoding Clinical Values..."):
        # Load image and force correct rotation from mobile EXIF data
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img) 
        
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = clean_clinical_image(frame)
        
        # UPGRADE 2: PSM 4 is designed for variable-size columns (like receipts)
        text = pytesseract.image_to_string(processed, config='--oem 3 --psm 4')
        
        d = {k: None for k in patterns.keys()}
        for key, regs in patterns.items():
            for r in regs:
                # UPGRADE 3: Added more allowed "noise" characters between the label and the number
                m = re.search(r + r'[\s:=,\|\~_*\-]*([-+]?\d*\.\d+|\d+)', text, re.IGNORECASE)
                if m: 
                    d[key] = float(m.group(1))
                    break

        # --- MATH GATEKEEPER ---
        if d['ph'] is None or d['pco2'] is None:
            st.error("❌ OCR missed primary values (pH or pCO2).")
            with st.expander("Machine Vision Debugger - View Raw Output"):
                st.image(processed, caption="What the computer saw")
                st.text("Raw Text Found:\n" + text) # Shows exactly what Tesseract read
        else:
            # Map safe fallbacks for secondary values
            ph = d['ph']
            pco2 = d['pco2']
            hco3 = d['hco3'] if d['hco3'] is not None else 24.0
            na = d['na'] if d['na'] is not None else 140.0
            cl = d['cl'] if d['cl'] is not None else 104.0
            lac = d['lactate'] if d['lactate'] is not None else 1.0
            po2 = d['po2'] if d['po2'] is not None else 90.0
            po4 = d['po4'] if d['po4'] is not None else 3.0
            
            # Clinical Engine Math
            aa_grad = (((fio2_pct/100) * 713) - (1.2 * pco2)) - po2
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6
            ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb))

            # UI Dashboard
            st.success("✅ Analysis Complete")
            col1, col2 = st.columns(2)
            col1.metric("pH", ph)
            col2.metric("pCO2", pco2)
            
            st.divider()
            st.write(f"**Stewart cBE:** {cbe:.2f} mmol/L")
            st.write(f"**A-a Gradient:** {aa_grad:.1f} mmHg")
            st.write(f"**Anion Gap (Corr):** {ag_corr:.1f} mmol/L")
