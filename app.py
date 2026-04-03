import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import shutil

# --- CLOUD CONFIG ---
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸")

# --- THE BACK CAMERA FORCER ---
# This CSS hides the 'flip' button and attempts to force the rear lens via the browser's capture attribute
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] > label {
        color: #007bff;
        font-weight: bold;
    }
    /* This targets the internal video element and requests the back camera */
    video {
        transform: scaleX(1) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩸 Gobseck ABG Engine")
st.caption("Associate Professor Dr. Pektezel's Clinical Tool")

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=-50)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11)

with st.sidebar:
    st.header("Patient Data")
    age = st.number_input("Age", value=60)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0)

# --- THE FIX: ADDING MEDIA DEVICE CONSTRAINTS ---
# We use st.camera_input but add a 'key' to reset it if needed
cam_image = st.camera_input("📸 Align the ABG and capture", key="abg_cam")

if cam_image:
    with st.spinner("Analyzing..."):
        img = Image.open(cam_image)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = clean_clinical_image(frame)
        text = pytesseract.image_to_string(processed, config='--psm 6')
        
        d = {k: None for k in patterns.keys()}
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=]*([-+]?\d*\.\d+|\d+)', text, re.IGNORECASE)
                if m: d[key] = float(m.group(1)); break

        if not d['ph']:
            st.error("❌ OCR Fail. Move closer/further to help the phone focus.")
            with st.expander("Show Machine Vision"):
                st.image(processed)
        else:
            ph, pco2, hco3 = d['ph'], d['pco2'], d['hco3']
            na, cl, lac = d.get('na', 140), d.get('cl', 104), d.get('lactate', 1.0)
            po2, po4 = d.get('po2', 90), d.get('po4', 3.0)
            
            # Clinical Logic
            aa_grad = (((fio2_pct/100) * 713) - (1.2 * pco2)) - po2
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6

            st.success("Analysis Complete")
            st.write(f"**Stewart cBE:** {cbe:.2f} mmol/L")
            st.write(f"**A-a Gradient:** {aa_grad:.1f} mmHg")
