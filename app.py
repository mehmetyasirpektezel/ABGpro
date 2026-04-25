import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageOps
import re
import shutil

# --- BULUT & TESSERACT AYARLARI ---
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

# --- ARAYÜZ KURULUMU ---
st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸")

st.title("🩸 Gobseck ABG Engine")
st.caption("Doç. Dr. Pektezel'in Profesyonel Klinik Tanı Aracı")

# --- OCR ARAMA ŞABLONLARI (HALÜSİNASYONLAR EKLENDİ) ---
patterns = {
    'ph': [r'pH', r'p\.H', r'PH', r'DH'],
    'pco2': [r'pCO2', r'PCO2', r'pCOz', r'pCO', r'PCO', r'pC02', r'pC0', r'pCO;', r'HCQ'],
    'hco3': [r'cHCO3', r'HCO3', r'HCO', r'act\.HCO', r'TOO'], 
    'po2': [r'pO2', r'PO2', r'pO', r'pOz', r'PO', r'p02', r'PO»'],
    'na': [r'Na\+', r'Sodium', r'Na', r'NA'],
    'cl': [r'Cl-', r'Chloride', r'Cl', r'CL', r'C3'],
    'lactate': [r'Lactate', r'Lac', r'Laktat'],
    'po4': [r'PO4', r'Phosphate', r'Fosfat']
}

# --- GÖRÜNTÜ İŞLEME (EKRAN İÇİN OPTİMİZE) ---
def clean_clinical_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(scaled, (3, 3), 0)
    return blur

# --- GOBSECK KLİNİK TANI MOTORU (WINTERS, BLEICH, KOMPANZASYON) ---
def analyze_acid_base(ph, pco2, hco3):
    report = []
    
    # 1. Bleich Kuralı (Veri Tutarlılığı)
    expected_h = 24 * (pco2 / hco3) if hco3 else 0
    expected_ph = 9.0 - np.log10(expected_h) if expected_h > 0 else 7.4
    
    if abs(ph - expected_ph) > 0.05:
        report.append("⚠️ **Bleich Kuralı İhlali:** Değerler tutarsız. Okuma hatası, venöz kan veya laboratuvar hatası şüphesi.")
    else:
        report.append("✅ **Bleich Kuralı Geçerli:** Veriler kendi içinde fizyolojik olarak tutarlı.")

    # 2. Birincil Bozukluk ve Kompanzasyon Analizi
    if ph < 7.36:
        if hco3 < 22:
            report.append("🩸 **Birincil Bozukluk:** Metabolik Asidoz")
            exp_pco2 = (1.5 * hco3) + 8
            if pco2 < (exp_pco2 - 2):
                report.append(f"🔍 **Miks Bozukluk:** Eşzamanlı Solunumsal Alkaloz (Beklenen pCO2: {exp_pco2:.1f} ±2)")
            elif pco2 > (exp_pco2 + 2):
                report.append(f"🔍 **Miks Bozukluk:** Eşzamanlı Solunumsal Asidoz (Beklenen pCO2: {exp_pco2:.1f} ±2)")
            else:
                report.append("✅ Solunumsal kompanzasyon yeterli (Winter Formülü ile uyumlu).")
        elif pco2 > 44:
            report.append("🫁 **Birincil Bozukluk:** Solunumsal Asidoz")
            report.append(f"Beklenen HCO3 (Akut): {24 + ((pco2 - 40) / 10):.1f} | (Kronik): {24 + 4 * ((pco2 - 40) / 10):.1f}")
    elif ph > 7.44:
        if hco3 > 26:
            report.append("🩸 **Birincil Bozukluk:** Metabolik Alkaloz")
            exp_pco2 = (0.7 * hco3) + 21
            if pco2 > (exp_pco2 + 2):
                report.append(f"🔍 **Miks Bozukluk:** Eşzamanlı Solunumsal Asidoz (Beklenen pCO2: {exp_pco2:.1f} ±2)")
            elif pco2 < (exp_pco2 - 2):
                report.append(f"🔍 **Miks Bozukluk:** Eşzamanlı Solunumsal Alkaloz (Beklenen pCO2: {exp_pco2:.1f} ±2)")
            else:
                report.append("✅ Solunumsal kompanzasyon yeterli.")
        elif pco2 < 36:
            report.append("🫁 **Birincil Bozukluk:** Solunumsal Alkaloz")
            report.append(f"Beklenen HCO3 (Akut): {24 - 2 * ((40 - pco2) / 10):.1f} | (Kronik): {24 - 5 * ((40 - pco2) / 10):.1f}")
    else:
        report.append("pH Normal aralıkta. Gizli metabolik bozukluklar için Anyon Açığı ve Stewart cBE değerlerini kontrol edin.")

    return report

# --- YAN MENÜ: KLİNİK BAĞLAM ---
with st.sidebar:
    st.header("Hasta Verileri")
    age = st.number_input("Yaş", value=60)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0)

# --- MOBİL KAMERA & DOSYA YÜKLEME ---
st.info("Kamerayı açmak için 'Browse files' -> 'Kamera' seçeneğine dokunun. Mümkünse sadece pH, pCO2 ve HCO3 içeren bölüme odaklanın.")
uploaded_file = st.file_uploader("ABG Fişini Yükle veya Fotoğrafını Çek", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with st.spinner("Klinik Değerler Çözümleniyor..."):
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img) 
        
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = clean_clinical_image(frame)
        
        text = pytesseract.image_to_string(processed, config='--oem 3 --psm 4')
        
        # Değerleri Ayıklama
        d = {k: None for k in patterns.keys()}
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=,\|\~_*\-\.]*([-+]?\d+[\.,]\d+|\d+)', text, re.IGNORECASE)
                if m: 
                    clean_number = m.group(1).replace(',', '.')
                    val = float(clean_number)
                    
                    # KLİNİK GÜVENLİK: Eksik ondalık noktaları otomatik düzelt (Örn: pH 745 okunduysa 7.45 yap)
                    if key == 'ph':
                        if val > 600 and val < 800: val = val / 100
                        elif val > 6000 and val < 8000: val = val / 1000
                        
                    d[key] = val
                    break

        # --- GÜVENLİK KAPISI ---
        if d['ph'] is None or d['pco2'] is None:
            st.error("❌ OCR temel değerleri (pH veya pCO2) bulamadı. Lütfen daha net bir fotoğraf çekin.")
            with st.expander("Geliştirici Paneli - Makine Ne Gördü?"):
                st.image(processed, caption="İşlenmiş Görüntü")
                st.text("Makinenin Okuduğu Ham Metin:\n" + text)
        else:
            ph = d['ph']
            pco2 = d['pco2']
            hco3 = d['hco3'] if d['hco3'] is not None else 24.0
            na = d['na'] if d['na'] is not None else 140.0
            cl = d['cl'] if d['cl'] is not None else 104.0
            lac = d['lactate'] if d['lactate'] is not None else 1.0
            po2 = d['po2'] if d['po2'] is not None else 90.0
            po4 = d['po4'] if d['po4'] is not None else 3.0
            
            # İLERİ DÜZEY HESAPLAMALAR
            aa_grad = (((fio2_pct/100) * 713) - (1.2 * pco2)) - po2
            cbe = (na - cl - 38) + (1 - lac) + (4 - alb) * 2.5 + (3 - po4) * 0.6
            ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb))

            diagnostic_report = analyze_acid_base(ph, pco2, hco3)

            # --- SONUÇ EKRANI ---
            st.success("✅ Analiz ve Teşhis Tamamlandı")
            col1, col2, col3 = st.columns(3)
            col1.metric("pH", ph)
            col2.metric("pCO2", pco2)
            col3.metric("HCO3", hco3)
            
            st.divider()
            
            st.subheader("📋 Klinik Teşhis Raporu")
            for line in diagnostic_report:
                st.write(line)
            
            st.divider()
            
            st.subheader("🧪 Stewart & İleri Fizyoloji")
            st.write(f"**Stewart cBE (Ecf):** {cbe:.2f} mmol/L")
            st.write(f"**A-a Gradient:** {aa_grad:.1f} mmHg (Beklenen: <{(age+10)/4:.1f})")
            st.write(f"**Albümin Düzeltilmiş Anyon Açığı:** {ag_corr:.1f} mmol/L")
