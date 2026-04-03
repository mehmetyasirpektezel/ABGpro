{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import cv2\
import pytesseract\
import numpy as np\
from PIL import Image\
import re\
\
# --- UI SETUP ---\
st.set_page_config(page_title="Gobseck ABG Engine", page_icon="\uc0\u55358 \u56952 ", layout="centered")\
st.title("Gobseck ABG Engine")\
st.markdown("**9-Step Stewart Approach & Bleich Validation**")\
\
# --- CORE LOGIC ---\
patterns = \{\
    'ph': [r'pH', r'p\\.H'],\
    'pco2': [r'pCO2', r'PCO2'],\
    'hco3': [r'cHCO3', r'HCO3\\(act\\)', r'act\\.HCO3', r'HCO3a', r'HCO3-'],\
    'po2': [r'pO2', r'PO2'],\
    'na': [r'Na\\+', r'Sodium', r'Na'],\
    'cl': [r'Cl-', r'Chloride', r'Cl'],\
    'lactate': [r'Lactate', r'Lac', r'Laktat'],\
    'po4': [r'PO4', r'Phosphate', r'Fosfat']\
\}\
\
def preprocess_mobile_thermal(image):\
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)\
    blur = cv2.GaussianBlur(gray, (5, 5), 0)\
    thresh = cv2.adaptiveThreshold(\
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10\
    )\
    return thresh\
\
# --- SIDEBAR ---\
with st.sidebar:\
    st.header("Clinical Context")\
    age = st.number_input("Patient Age", value=60, step=1)\
    fio2_pct = st.number_input("FiO2 (%)", value=21, step=1)\
    alb = st.number_input("Albumin (g/dL)", value=4.0, step=0.1)\
    fio2 = fio2_pct / 100.0\
\
# --- CAMERA INPUT ---\
st.info("Ensure the ABG printout is flat and well-lit. Tap below to scan.")\
cam_image = st.camera_input("Scan Clinical Printout")\
\
if cam_image is not None:\
    with st.spinner("Processing High-Res Mobile Image..."):\
        img = Image.open(cam_image)\
        frame = np.array(img)\
        frame = cv2.cvtColor(frame, cv2.RGB2BGR)\
        \
        processed = preprocess_mobile_thermal(frame)\
        \
        custom_config = r'--oem 3 --psm 6'\
        text = pytesseract.image_to_string(processed, config=custom_config)\
        \
        d = \{\}\
        for key, regs in patterns.items():\
            for r in regs:\
                m = re.search(r + r'[\\s:=]*([-+]?\\d*\\.\\d+|\\d+)', text, re.IGNORECASE)\
                if m: \
                    d[key] = float(m.group(1))\
                    break\
        \
        with st.expander("View OCR Machine Vision"):\
            st.image(processed, caption="Machine Vision View", use_column_width=True)\
            st.text("Raw Extraction Data:\\n" + text[:300] + "...")\
\
        if 'ph' not in d:\
            st.error("\uc0\u9888 \u65039  Failed to lock onto primary anchors. Adjust lighting and try again.")\
        else:\
            st.success("Target Locked. Compiling clinical data...")\
            \
            ph, pco2, hco3 = d.get('ph'), d.get('pco2'), d.get('hco3')\
            na, cl, lac = d.get('na', 140), d.get('cl', 104), d.get('lactate', 1.0)\
            po2 = d.get('po2', 90)\
            po4 = d.get('po4', 3.0)\
            \
            exp_ph = 7.40 - ((24 * (pco2/hco3)) - 40) * 0.01 if pco2 and hco3 else None\
            ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb)) if hco3 else None\
            aa_grad = ((fio2 * 713) - (1.2 * pco2)) - po2 if pco2 else None\
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6\
            \
            st.subheader("\uc0\u55357 \u56561  9-Step ABG Analysis")\
            col1, col2 = st.columns(2)\
            col1.metric("pH", ph)\
            col2.metric("pCO2", pco2)\
            col1.metric("HCO3", hco3)\
            col2.metric("Na / Cl", f"\{na\} / \{cl\}")\
            \
            st.markdown("---")\
            if exp_ph:\
                status = "\uc0\u9989  Valid" if abs(ph-exp_ph)<0.05 else "\u9888 \u65039  Inconsistent"\
                st.write(f"**1. Bleich Consistency:** \{status\}")\
            \
            st.write(f"**2. Primary Disorder:** \{'Acidosis' if ph < 7.36 else 'Alkalosis' if ph > 7.44 else 'Normal'\}")\
            \
            if hco3: st.write(f"**5. Anion Gap (Alb-C):** \{ag_corr:.1f\} mmol/L")\
            if pco2: st.write(f"**8. A-a Gradient:** \{aa_grad:.1f\} mmHg *(Expected < \{(age+10)/4:.1f\})*")\
            st.write(f"**9. Stewart cBE Ecf:** \{cbe:.2f\} mmol/L")}