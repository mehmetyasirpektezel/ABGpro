import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import shutil

# --- LINUX CLOUD CONFIG ---
# Automatically finds Tesseract on the Streamlit Linux server
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

# --- UI SETUP ---
st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸", layout="centered")

# CSS to make the camera interface look better on Android
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] {
        border: 2px solid #ff4b4b;
        border-radius: 15px;
        padding: 10px;
    }
    .main {
        background-color: #f5f7f9;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩸 Gobseck ABG Engine")
st.markdown("🔍 **Docent Dr. Pektezel's 9-Step Stewart Engine**")

# Regex Patterns for Siemens/Radiometer
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
    """Optimized for high-res mobile back cameras and thermal paper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast to handle faint thermal ink
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-30)
    
    # Resize only if the mobile capture is low resolution
    if gray.shape[1] < 1200:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Blend thermal dots into solid lines
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Local adaptive threshold to neutralize shadows from the phone/hand
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
    )
    return thresh

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("🩸 Patient Context")
    age = st.number_input("Age", value=60, min_value=0, max_value=120)
    fio2_pct = st.number_input("FiO2 (%)", value=21, min_value=21, max_value=100)
    alb = st.number_input("Albumin (g/dL)", value=4.0, step=0.1)
    fio2 = fio2_pct / 100.0
    st.divider()
    st.info("Ensure the printout is flat and in good light.")

# --- MAIN CAMERA INTERFACE ---
# Streamlit will usually show a 'Switch Camera' button on Android Chrome
cam_image = st.camera_input("Scan ABG Printout")

if cam_image is not None:
    with st.spinner("Processing Clinical Values..."):
        # Image conversion
        img = Image.open(cam_image)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # CV Engine
        processed = preprocess_mobile_thermal(frame)
        text = pytesseract.image_to_string(processed, config='--psm 6')
        
        # Extraction logic
        d = {}
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=]*([-+]?\d*\.\d+|\d+)', text, re.IGNORECASE)
                if m: 
                    d[key] = float(m.group(1))
                    break
        
        # Machine Vision Viewer
        with st.expander("🔬 Machine Vision (Debug)"):
            st.image(processed, caption="OCR Binarized View")
            st.text("Raw Text Found:\n" + text[:200])

        if 'ph' not in d:
            st.error("⚠️ Primary anchors (pH/pCO2) not found. Try a clearer photo.")
        else:
            # Clinical Variables
            ph, pco2, hco3 = d.get('ph'), d.get('pco2'), d.get('hco3')
            na, cl, lac = d.get('na', 140), d.get('cl', 104), d.get('lactate', 1.0)
            po2 = d.get('po2', 90)
            po4 = d.get('po4', 3.0)
            
            # Calculations
            exp_ph = 7.40 - ((24 * (pco2/hco3)) - 40) * 0.01 if pco2 and hco3 else 7.4
            ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb)) if hco3 else 12
            aa_grad = ((fio2 * 713) - (1.2 * pco2)) - po2 if pco2 else 0
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6
            
            # --- RESULTS DASHBOARD ---
            st.subheader("📊 Clinical Analysis")
            m1, m2, m3 = st.columns(3)
            m1.metric("pH", ph)
            m2.metric("pCO2", pco2)
            m3.metric("HCO3", hco3)
            
            st.markdown("---")
            
            # 9-Step Display
            st.write(f"**1. Bleich Rule:** {'✅ Consistent' if abs(ph-exp_ph)<0.05 else '⚠️ Inconsistent'}")
            st.write(f"**2. Primary:** {'Acidosis' if ph < 7.36 else 'Alkalosis' if ph > 7.44 else 'Normal'}")
            st.write(f"**5. Corrected AG:** {ag_corr:.1f} mmol/L")
            st.write(f"**8. A-a Gradient:** {aa_grad:.1f} mmHg (Exp: <{(age+10)/4:.1f})")
            
            st.success(f"**9. Stewart cBE (Ecf):** {cbe:.2f} mmol/L")
            
            if abs(cbe) > 2:
                st.warning("Metabolic component identified via Stewart strong ion difference.")
