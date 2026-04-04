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
    'ph': [r'pH', r'p\.H', r'PH'],
    'pco2': [r'pCO2', r'PCO2', r'pCOz', r'pCO', r'PCO', r'pC02', r'pC0', r'pCO;'],
    'hco3': [r'cHCO3', r'HCO3', r'HCO', r'act\.HCO'],
    'po2': [r'pO2', r'PO2', r'pO', r'pOz', r'PO', r'p02'],
    'na': [r'Na\+', r'Sodium', r'Na', r'NA'],
    'cl': [r'Cl-', r'Chloride', r'Cl', r'CL'],
    'lactate': [r'Lactate', r'Lac', r'Laktat'],
    'po4': [r'PO4', r'Phosphate', r'Fosfat']
}

def clean_clinical_image(image):
    """The 'Naked' LSTM approach. Let Tesseract's neural net handle the screen grid."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Upscale using high-quality Lanczos interpolation (keeps text sharp)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    
    # 2. A very slight blur to soften the monitor's pixel grid
    blur = cv2.GaussianBlur(scaled, (3, 3), 0)
    
    # 3. Notice we return the grayscale directly. No thresholding/binarization!
    return blur

def analyze_acid_base(ph, pco2, hco3):
    """The Gobseck Diagnostic Core: Evaluates primary disorders and compensations."""
    report = []
    
    # Bleich Rule
    expected_h = 24 * (pco2 / hco3) if hco3 else 0
    expected_ph = 9.0 - np.log10(expected_h) if expected_h > 0 else 7.4
    
    if abs(ph - expected_ph) > 0.05:
        report.append("⚠️ **Bleich Rule Failed:** Suspect venous sample or lab error.")
    else:
        report.append("✅ **Bleich Rule Passed:** Data is internally consistent.")

    # Primary & Compensation Logic
    if ph < 7.36:
        if hco3 < 22:
            report.append("🩸 **Primary:** Metabolic Acidosis")
            exp_pco2 = (1.5 * hco3) + 8
            if pco2 < (exp_pco2 - 2):
                report.append(f"🔍 **Mixed:** Concomitant Respiratory Alkalosis (Exp. pCO2: {exp_pco2:.1f} ±2)")
            elif pco2 > (exp_pco2 + 2):
                report.append(f"🔍 **Mixed:** Concomitant Respiratory Acidosis (Exp. pCO2: {exp_pco2:.1f} ±2)")
            else:
                report.append("Adequate respiratory compensation.")
        elif pco2 > 44:
            report.append("🫁 **Primary:** Respiratory Acidosis")
            report.append(f"Expected HCO3 (Acute): {24 + ((pco2 - 40) / 10):.1f} | (Chronic): {24 + 4 * ((pco2 - 40) / 10):.1f}")
    elif ph > 7.44:
        if hco3 > 26:
            report.append("🩸 **Primary:** Metabolic Alkalosis")
            exp_pco2 = (0.7 * hco3) + 21
            if pco2 > (exp_pco2 + 2):
                report.append(f"🔍 **Mixed:** Concomitant Respiratory Acidosis (Exp. pCO2: {exp_pco2:.1f} ±2)")
            elif pco2 < (exp_pco2 - 2):
                report.append(f"🔍 **Mixed:** Concomitant Respiratory Alkalosis (Exp. pCO2: {exp_pco2:.1f} ±2)")
            else:
                report.append("Adequate respiratory compensation.")
        elif pco2 < 36:
            report.append("🫁 **Primary:** Respiratory Alkalosis")
            report.append(f"Expected HCO3 (Acute): {24 - 2 * ((40 - pco2) / 10):.1f} | (Chronic): {24 - 5 * ((40 - pco2) / 10):.1f}")
    else:
        report.append("Normal pH. Check Anion Gap and Stewart cBE for occult metabolic disorders.")

    return report

# --- SIDEBAR ---
with st.sidebar:
    st.header("Patient Context")
    age = st.number_input("Age", value=60)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0)

# --- NATIVE CAPTURE ---
st.info("Tap 'Browse files' -> 'Camera'. Keep the phone steady.")
uploaded_file = st.file_uploader("Upload or Take Photo of ABG", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with st.spinner("Decoding Screen Capture..."):
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img) 
        
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = clean_clinical_image(frame)
        
        # PSM 4 is highly resilient to columns and spacing
        text = pytesseract.image_to_string(processed, config='--oem 3 --psm 4')
        
        d = {k: None for k in patterns.keys()}
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=,\|\~_*\-]*([-+]?\d+[\.,]\d+|\d+)', text, re.IGNORECASE)
                if m: 
                    d[key] = float(m.group(1).replace(',', '.'))
                    break

        # --- MATH GATEKEEPER ---
        if d['ph'] is None or d['pco2'] is None:
            st.error("❌ OCR missed primary values (pH or pCO2).")
            with st.expander("Machine Vision Debugger"):
                st.image(processed, caption="Naked Grayscale View")
                st.text("Raw Text Found by OCR:\n" + text)
        else:
            ph = d['ph']
            pco2 = d['pco2']
            hco3 = d['hco3'] if d['hco3'] is not None else 24.0
            na = d['na'] if d['na'] is not None else 140.0
            cl = d['cl'] if d['cl'] is not None else 104.0
            lac = d['lactate'] if d['lactate'] is not None else 1.0
            po2 = d['po2'] if d['po2'] is not None else 90.0
            po4 = d['po4'] if d['po4'] is not None else 3.0
            
            # Math
            aa_grad = (((fio2_pct/100) * 713) - (1.2 * pco2)) - po2
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6
            ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb))

            # The Gobseck Audit
            diagnostic_report = analyze_acid_base(ph, pco2, hco3)

            # UI Dashboard
            st.success("✅ Analysis Complete")
            col1, col2, col3 = st.columns(3)
            col1.metric("pH", ph)
            col2.metric("pCO2", pco2)
            col3.metric("HCO3", hco3)
            
            st.divider()
            st.subheader("📋 Diagnostic Ledger")
            for line in diagnostic_report:
                st.write(line)
            
            st.divider()
            st.subheader("🧪 Stewart & Gradients")
            st.write(f"**Stewart cBE:** {cbe:.2f} mmol/L")
            st.write(f"**A-a Gradient:** {aa_grad:.1f} mmHg (Exp: <{(age+10)/4:.1f})")
            st.write(f"**Anion Gap (Corr):** {ag_corr:.1f} mmol/L")
