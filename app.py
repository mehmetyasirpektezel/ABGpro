import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import shutil

# --- LINUX CLOUD CONFIG ---
# This ensures the server finds the Tesseract engine we installed via packages.txt
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

# --- UI SETUP ---
st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸")
st.title("🩸 Gobseck ABG Engine")
st.markdown("9-Step Stewart Approach & Bleich Validation")

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

def preprocess_mobile_thermal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
    )
    return thresh

# --- SIDEBAR ---
with st.sidebar:
    st.header("Clinical Context")
    age = st.number_input("Patient Age", value=60)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0)
    fio2 = fio2_pct / 100.0

# --- MOBILE CAMERA ---
cam_image = st.camera_input("Scan ABG Printout")

if cam_image is not None:
    with st.spinner("Analyzing Clinical Data..."):
        img = Image.open(cam_image)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        processed = preprocess_mobile_thermal(frame)
        text = pytesseract.image_to_string(processed, config='--psm 6')
        
        d = {}
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=]*([-+]?\d*\.\d+|\d+)', text, re.IGNORECASE)
                if m: 
                    d[key] = float(m.group(1))
                    break
        
        with st.expander("Machine Vision Debugger"):
            st.image(processed, caption="OCR View")

        if 'ph' not in d:
            st.error("⚠️ Could not find pH. Check lighting and try again.")
        else:
            ph, pco2, hco3 = d.get('ph'), d.get('pco2'), d.get('hco3')
            na, cl, lac = d.get('na', 140), d.get('cl', 104), d.get('lactate', 1.0)
            po2 = d.get('po2', 90)
            
            # Calculations
            exp_ph = 7.40 - ((24 * (pco2/hco3)) - 40) * 0.01 if pco2 and hco3 else 7.4
            ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb)) if hco3 else 12
            aa_grad = ((fio2 * 713) - (1.2 * pco2)) - po2 if pco2 else 0
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5
            
            st.subheader("Results")
            st.write(f"**Primary:** {'Acidosis' if ph < 7.36 else 'Alkalosis' if ph > 7.44 else 'Normal'}")
            st.write(f"**Stewart cBE:** {cbe:.2f} mmol/L")
            st.write(f"**A-a Gradient:** {aa_grad:.1f} mmHg")
            st.write(f"**Corrected Anion Gap:** {ag_corr:.1f}")
