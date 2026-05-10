from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Hacı Global Risk API")

# Sitenin API'ye bağlanabilmesi için CORS izni
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Modeli yüklüyoruz
model = joblib.load('haci_risk_model.joblib')

# 2. Şablon hazırlığı 
df_raw = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin1')
gereksiz_sutunlar = ['Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname', 'Product Image', 'Customer Street', 'Order Zipcode']
hileli_sutunlar = ['Days for shipping (real)', 'Delivery Status']
hedef = 'Late_delivery_risk'

X_clean = df_raw.drop(columns=gereksiz_sutunlar + hileli_sutunlar + [hedef])

kategorik_sutunlar = X_clean.select_dtypes(include=['object']).columns.tolist()

# Boş verileri doldur
X_clean[kategorik_sutunlar] = X_clean[kategorik_sutunlar].fillna("Bilinmiyor")
X_clean = X_clean.fillna(0)

# Metinleri sayılara kodla
le = LabelEncoder()
for col in kategorik_sutunlar:
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))

# Şablonu güvenle oluştur
X_sablon = X_clean.iloc[[0]].copy()
for col in X_clean.columns:
    X_sablon[col] = X_clean[col].median()

@app.get("/")
def home():
    return {"mesaj": "Hacı Risk Analitik Platformu Backend'i Çalışıyor!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    test_verisi = X_sablon.copy()
    
    # EN KRİTİK DÜZELTME: Siteden gelen alt çizgili isimleri, CSV'nin boşluklu isimleriyle eşleştiriyoruz!
    sutun_eslesmeleri = {
        "Shipping_Mode": "Shipping Mode",
        "Order_City": "Order City", 
        "Category_Id": "Category Id"
    }
    
    for frontend_key, dataset_key in sutun_eslesmeleri.items():
        if frontend_key in veri and dataset_key in test_verisi.columns:
            test_verisi[dataset_key] = veri[frontend_key]
            
    # Model Hesaplaması
    try:
        olasiliklar = model.predict_proba(test_verisi)[0]
        gecikme_riski = float(olasiliklar[1] * 100)
    except AttributeError:
        tahmin = model.predict(test_verisi)
        gecikme_riski = 85.0 if tahmin[0] == 1 else 15.0

    # GELİŞMİŞ HAVA DURUMU BEYNİ (Bulutlu/Kapalı havalar da eklendi)
    weather_desc = veri.get("weather_desc", "").lower()
    if any(x in weather_desc for x in ["yağmur", "fırtına", "kar", "rain", "storm"]):
        gecikme_riski = min(gecikme_riski + 25.0, 98.5) # Kötü havada risk çok artar
    elif any(x in weather_desc for x in ["kapalı", "bulutlu", "clouds", "overcast"]):
        gecikme_riski = min(gecikme_riski + 8.0, 98.5) # Kapalı/Bulutlu havada risk azıcık artar
    elif any(x in weather_desc for x in ["açık", "güneş", "clear", "sun"]):
        gecikme_riski = max(gecikme_riski - 10.0, 2.0) # İyi havada risk düşer

    risk_metni = "🚨 YÜKSEK RİSK" if gecikme_riski > 65 else "⚠️ ORTA RİSK" if gecikme_riski > 35 else "✅ DÜŞÜK RİSK"
    
    return {
        "Sistem Mesajı": "Başarılı",
        "risk_skoru": round(gecikme_riski, 2),
        "Tahmin Sonucu": risk_metni
    }