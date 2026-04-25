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
st.set_page_config(page_title="Gobseck ABG Pro", page_icon="🩸", layout="centered")

st.title("🩸 Gobseck ABG Engine")
st.caption("Doç. Dr. Pektezel'in Profesyonel Klinik Tanı Aracı")

# --- OCR ARAMA ŞABLONLARI ---
patterns = {
    'ph': [r'pH', r'p\.H', r'PH', r'DH'],
    'pco2': [r'pCO2', r'PCO2', r'pCOz', r'pCO', r'PCO', r'pC02', r'pC0', r'pCO;', r'HCQ'],
    'hco3': [r'cHCO3', r'HCO3', r'HCO', r'act\.HCO', r'TOO'], 
    'po2': [r'pO2', r'PO2', r'pO', r'pOz', r'PO', r'p02', r'PO»'],
    'na': [r'Na\+', r'Sodium', r'Na', r'NA'],
    'cl': [r'Cl-', r'Chloride', r'Cl', r'CL', r'C3'],
    'lactate': [r'Lactate', r'Lac', r'Laktat'],
    'po4': [r'PO4', r'Phosphate', r'Fosfat'],
    'cbe': [r'cBase\s*\(Ecf\)', r'cBE', r'BE\(B\)', r'BE', r'Base\s*Excess', r'Baz\s*Fazlas']
}

# --- GÖRÜNTÜ İŞLEME ---
def clean_clinical_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    return cv2.GaussianBlur(scaled, (3, 3), 0)

# --- KLİNİK TANI MOTORU (WINTERS, BLEICH, KOMPANZASYON) ---
def analyze_acid_base(ph, pco2, hco3):
    report = []
    
    # 1. Bleich Kuralı
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

# --- YAN MENÜ: KLİNİK BAĞLAM VE TERMOREGÜLASYON ---
with st.sidebar:
    st.header("Hasta Verileri")
    age = st.number_input("Yaş", value=60)
    temp = st.number_input("Vücut Sıcaklığı (°C)", value=37.0, step=0.1)
    fio2_pct = st.number_input("FiO2 (%)", value=21)
    alb = st.number_input("Albumin (g/dL)", value=4.0, step=0.1)

# --- MOBİL KAMERA & DOSYA YÜKLEME ---
st.info("Kamerayı açmak için 'Browse files' -> 'Kamera' seçeneğine dokunun.")
uploaded_file = st.file_uploader("ABG Fişini Yükle veya Fotoğrafını Çek", type=['jpg', 'jpeg', 'png'])

d = {k: None for k in patterns.keys()}

if uploaded_file is not None:
    with st.spinner("Optik Asistan Değerleri Okuyor..."):
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img) 
        
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = clean_clinical_image(frame)
        
        text = pytesseract.image_to_string(processed, config='--oem 3 --psm 4')
        
        for key, regs in patterns.items():
            for r in regs:
                m = re.search(r + r'[\s:=,\|\~_*\-\.]*([-+]?\d+[\.,]\d+|\d+)', text, re.IGNORECASE)
                if m: 
                    clean_number = m.group(1).replace(',', '.')
                    val = float(clean_number)
                    
                    if key == 'ph':
                        if val > 600 and val < 800: val = val / 100
                        elif val > 6000 and val < 8000: val = val / 1000
                    
                    d[key] = val
                    break

# --- HİBRİT KLİNİK DOĞRULAMA FORMU ---
st.divider()
st.subheader("🔍 Klinik Doğrulama ve Teşhis")
st.write("Makinenin okuyabildiği değerler (cihazın 37°C standart ölçümleri) aşağıya dolduruldu. Eksik olanları tamamlayıp motoru çalıştırın.")

with st.form("klinik_dogrulama_formu"):
    c1, c2, c3, c4 = st.columns(4)
    safe_ph = d['ph'] if d['ph'] and 6.8 < d['ph'] < 7.8 else 7.40
    ph = c1.number_input("pH", value=float(safe_ph), format="%.3f", step=0.01)
    pco2 = c2.number_input("pCO2", value=float(d['pco2'] if d['pco2'] else 40.0), step=1.0)
    hco3 = c3.number_input("HCO3", value=float(d['hco3'] if d['hco3'] else 24.0), step=1.0)
    po2 = c4.number_input("pO2", value=float(d['po2'] if d['po2'] else 90.0), step=1.0)
    
    c5, c6, c7, c8 = st.columns(4)
    na = c5.number_input("Na", value=float(d['na'] if d['na'] else 140.0), step=1.0)
    cl = c6.number_input("Cl", value=float(d['cl'] if d['cl'] else 104.0), step=1.0)
    lac = c7.number_input("Laktat", value=float(d['lactate'] if d['lactate'] else 1.0), step=0.1)
    po4 = c8.number_input("Fosfat", value=float(d['po4'] if d['po4'] else 3.0), step=0.1)
    
    cbe_ocr = st.number_input("Cihazdan Okunan cBase(Ecf) - Opsiyonel", value=float(d['cbe'] if d['cbe'] else 0.0), step=0.1)
    
    submit_btn = st.form_submit_button("🩺 Gobseck Tanı Motorunu Çalıştır", type="primary", use_container_width=True)

# --- MOTORUN ÇALIŞMASI VE SONUÇLAR ---
if submit_btn:
    with st.spinner("Stewart ve Fizyolojik Denge Analiz Ediliyor..."):
        
        # 1. TEMEL HESAPLAMALAR
        aa_grad = (((fio2_pct/100) * 713) - (1.2 * pco2)) - po2
        ag_corr = (na - (cl + hco3)) + (2.5 * (4.0 - alb))
        
        # 2. STEWART HESAPLAMALARI
        cbe_hesaplanan = hco3 - 24.8 + (16.2 * (ph - 7.4))
        
        dsid_nacl = (na - cl) - 38
        dsid_lac = 1 - lac
        dsid_po4 = 2 - (po4 * 0.6)
        dsid_alb = (4 - alb) * 2.5
        dsid_total = dsid_nacl + dsid_lac + dsid_po4 + dsid_alb

        # 3. KLİNİK TEŞHİS RAPORU
        diagnostic_report = analyze_acid_base(ph, pco2, hco3)

        # --- SONUÇ EKRANI ---
        st.success("✅ Teşhis Raporu Hazır")
        
        st.subheader("📋 Klinik Karar Defteri")
        for line in diagnostic_report:
            st.info(line)
            
        # --- TERMOREGÜLASYON PANeli (SADECE 37°C DIŞINDA GÖRÜNÜR) ---
        if temp != 37.0:
            # Sıcaklık Düzeltme Formülleri
            ph_t = ph - (0.0146 + 0.065 * (ph - 7.40)) * (temp - 37.0)
            pco2_t = pco2 * (10 ** (0.021 * (temp - 37.0)))
            
            st.divider()
            st.subheader("🌡️ Termoregülasyon (pH-Stat Düzeltmesi)")
            st.warning(f"Kan gazı cihazı analizleri standart 37°C'de yapar. Hastanın gerçek vücut sıcaklığına ({temp}°C) göre in vivo değerler:")
            
            t1, t2 = st.columns(2)
            t1.metric(f"Düzeltilmiş pH ({temp}°C)", f"{ph_t:.3f}", delta=f"{ph_t - ph:.3f}", delta_color="inverse")
            t2.metric(f"Düzeltilmiş pCO2 ({temp}°C)", f"{pco2_t:.1f} mmHg", delta=f"{pco2_t - pco2:.1f} mmHg", delta_color="inverse")
        
        st.divider()
        
        # STEWART (ΔSID) DETAYLI ANALİZİ
        st.subheader("🔬 Stewart (ΔSID) Detaylı Analizi")
        
        col_cbe1, col_cbe2 = st.columns(2)
        col_cbe1.metric("Hesaplanan cBase(Ecf)", f"{cbe_hesaplanan:.2f} mmol/L")
        if cbe_ocr != 0.0:
            col_cbe2.metric("Cihazdan Okunan cBE", f"{cbe_ocr:.2f} mmol/L")
            
        st.markdown(f"""
        * **ΔSID (Na-Cl):** `{dsid_nacl:+.2f} mmol/L` — *{'Alkalozis' if dsid_nacl > 0 else 'Asidozis' if dsid_nacl < 0 else 'Normal'}*
        * **ΔSID (Laktat):** `{dsid_lac:+.2f} mmol/L` — *{'Laktik Asidoz' if dsid_lac < 0 else 'Normal'}*
        * **ΔSID (Fosfat):** `{dsid_po4:+.2f} mmol/L` — *{'Hiperfosfatemik Asidoz' if dsid_po4 < 0 else 'Normal'}*
        * **ΔSID (Albümin):** `{dsid_alb:+.2f} mmol/L` — *{'Hipoalbüminemik Alkaloz' if dsid_alb > 0 else 'Normal'}*
        """)
        
        st.info(f"**Toplam cBE_st (ΔSID_total):** {dsid_total:+.2f} mmol/L")

        fark = abs(cbe_hesaplanan - dsid_total)
        if fark <= 2.5:
            st.success(f"**Sağlama Başarılı:** Hesaplanan cBE ({cbe_hesaplanan:.2f}) ile Toplam ΔSID ({dsid_total:.2f}) formülleri birbiriyle uyumlu.")
        else:
            st.error(f"**Güçlü Katkı Uyarısı:** cBE ve Toplam ΔSID arasında {fark:.2f} mmol/L açıklanamayan fark (SIG) var. Ölçülmeyen diğer anyonları (Keton, Toksin, Üremi) değerlendirin.")

        st.divider()
        
        m1, m2 = st.columns(2)
        m1.metric("A-a Gradient", f"{aa_grad:.1f} mmHg", delta=f"Beklenen: <{(age+10)/4:.1f}", delta_color="off")
        m2.metric("Düzeltilmiş Anyon Açığı", f"{ag_corr:.1f} mmol/L")
