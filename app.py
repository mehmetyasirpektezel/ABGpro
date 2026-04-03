import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import shutil

# --- CLOUD CONFIG ---
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

# --- UI & BACK CAMERA INJECTION ---
st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸")

# This CSS/JS combo forces the Android browser to prefer the rear lens
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] {
        border: 3px solid #007bff;
        border-radius: 15px;
    }
    </style>
    <script>
    // Force back camera on mobile devices
    const constraints = { video: { facingMode: { exact: "environment" } } };
    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        const video = document.querySelector('video');
        if (video) video.srcObject = stream;
      })
      .catch(err => console.error("Back camera access error:", err));
    </script>
    """, unsafe_allow_html=True)

st.title("🩸 Gobseck ABG Engine")
st.caption("Associate Professor Dr. Pektezel's Clinical Diagnostic Tool")

# Patterns for Siemens/Radiometer thermal printouts
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
    """Optimized for Android autofocus and thermal ink."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Sharpen and contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=-40)
    # Blur to merge thermal dots
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Local adaptive thresholding to kill phone shadows
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Patient Data")
    age = st.number_input("Age", value=60)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0)
    st.divider()
    st.write("Target: Stewart cBE & A-a Gradient")

# --- CAMERA INPUT ---
cam_image = st.camera_input("📸 Align the paper and capture")

if cam_image:
    with st.spinner("Processing..."):
        img = Image.open(cam_image)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = clean_clinical_image(frame)
        
        # OCR with PSM 6 (Assume a single uniform block of text)
        text = pytesseract.image_to_string(processed, config='--psm 6')
        
        # Extraction
        d = {k: None for k in patterns.keys()}
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=]*([-+]?\d*\.\d+|\d+)', text, re.IGNORECASE)
                if m: d[key] = float(m.group(1)); break

        if not d['ph']:
            st.error("❌ OCR Failed. Please ensure the paper is flat and well-lit.")
            with st.expander("Show Machine Vision"):
                st.image(processed)
        else:
            # Data Mapping
            ph, pco2, hco3 = d['ph'], d['pco2'], d['hco3']
            na, cl, lac = d.get('na', 140), d.get('cl', 104), d.get('lactate', 1.0)
            po2, po4 = d.get('po2', 90), d.get('po4', 3.0)
            
            # Clinical Engine Calculations
            exp_ph = 7.40 - ((24 * (pco2/hco3)) - 40) * 0.01 if pco2 and hco3 else 7.4
            ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb)) if hco3 else 12
            aa_grad = (((fio2_pct/100) * 713) - (1.2 * pco2)) - po2
            # Stewart formula: SID + Atot + PO4
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6

            # UI RESULTS
            st.success("Analysis Complete")
            c1, c2, c3 = st.columns(3)
            c1.metric("pH", ph)
            c2.metric("pCO2", pco2)
            c3.metric("HCO3", hco3)
            
            st.divider()
            st.write(f"**Primary Path:** {'Acidosis' if ph < 7.36 else 'Alkalosis' if ph > 7.44 else 'Normal'}")
            st.write(f"**Stewart cBE (Ecf):** {cbe:.2f} mmol/L")
            st.write(f"**A-a Gradient:** {aa_grad:.1f} mmHg (Exp: <{(age+10)/4:.1f})")
            st.write(f"**Alb-Corrected AG:** {ag_corr:.1f} mmol/L")
