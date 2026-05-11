from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

# Gereksiz uyarıları kapat
warnings.filterwarnings('ignore')

app = FastAPI(title="Haci Global Visibility Hub API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. MODEL VE VERİYİ YÜKLE
model = joblib.load('haci_risk_model.joblib')

df_raw = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin1')
gereksiz_sutunlar = ['Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname', 'Product Image', 'Customer Street', 'Order Zipcode']
hileli_sutunlar = ['Days for shipping (real)', 'Delivery Status']
hedef = 'Late_delivery_risk'

X_clean = df_raw.drop(columns=gereksiz_sutunlar + hileli_sutunlar + [hedef])
kategorik_sutunlar = X_clean.select_dtypes(include=['object']).columns.tolist()

X_clean[kategorik_sutunlar] = X_clean[kategorik_sutunlar].fillna("Unknown")
X_clean = X_clean.fillna(0)

# Tüm veriyi KESİN OLARAK sayıya (float) çevirip eğitiyoruz
for col in kategorik_sutunlar:
    le = LabelEncoder()
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))

# İlk satırı şablon olarak al ve tüm tipleri zorla Float yap
X_sablon = X_clean.iloc[[0]].copy().astype(float)

@app.get("/")
def home():
    return {"message": "Haci Global API is running!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    # Şablonumuzu al
    test_verisi = X_sablon.copy()
    
    # 2. SEÇİMLERİ SAYI OLARAK ZORLA YAZ (Tercüman yok, direkt eşleştirme)
    mode_map = {
        "First Class": 0.0,
        "Same Day": 1.0,
        "Second Class": 2.0,
        "Standard Class": 3.0
    }
    
    gelen_mod = str(veri.get("Shipping_Mode", "Standard Class"))
    mode_val = float(gelen_mod) if gelen_mod.isdigit() else float(mode_map.get(gelen_mod, 3.0))

    # Gelen 3 veriyi şablona yerleştir
    test_verisi["Shipping Mode"] = mode_val
    test_verisi["Order City"] = float(veri.get("Order_City", test_verisi["Order City"].values[0]))
    test_verisi["Category Id"] = float(veri.get("Category_Id", test_verisi["Category Id"].values[0]))

    w_desc = veri.get("weather_desc", "").lower()
    
    # 3. DÜZ TAHMİN (Hiçbir atraksiyon yok)
    try:
        olasiliklar = model.predict_proba(test_verisi)[0]
        risk = float(olasiliklar[1] * 100)
    except Exception:
        tahmin = model.predict(test_verisi)
        risk = 85.0 if tahmin[0] == 1 else 15.0

    # Hava durumu etkisi
    if any(x in w_desc for x in ["yağmur", "fırtına", "kar", "rain", "storm", "snow"]):
        risk = min(risk + 25.0, 98.5)
    elif any(x in w_desc for x in ["kapalı", "bulutlu", "clouds", "overcast"]):
        risk = min(risk + 8.0, 98.5)
    elif any(x in w_desc for x in ["açık", "güneş", "clear", "sun"]):
        risk = max(risk - 10.0, 2.0)

    # Sonucu gönder
    risk_metni = "🚨 HIGH RISK" if risk > 65 else "⚠️ MEDIUM RISK" if risk > 35 else "✅ LOW RISK"
    
    return {
        "risk_skoru": round(risk, 2),
        "Tahmin Sonucu": risk_metni,
        "Tavsiye": ""  # Tavsiye kutusunu boş bıraktık, hata yaratmasın
    }