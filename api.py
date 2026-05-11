from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Haci Global Visibility Hub API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('haci_risk_model.joblib')

df_raw = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin1')
gereksiz_sutunlar = ['Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname', 'Product Image', 'Customer Street', 'Order Zipcode']
hileli_sutunlar = ['Days for shipping (real)', 'Delivery Status']
hedef = 'Late_delivery_risk'

X_clean = df_raw.drop(columns=gereksiz_sutunlar + hileli_sutunlar + [hedef])
kategorik_sutunlar = X_clean.select_dtypes(include=['object']).columns.tolist()

X_clean[kategorik_sutunlar] = X_clean[kategorik_sutunlar].fillna("Unknown")
X_clean = X_clean.fillna(0)

# Veriyi modelin anladığı şekle çevir (Orijinal çalışan yöntem)
for col in kategorik_sutunlar:
    le = LabelEncoder()
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))

# Şablonu çıkar (Sadece orijinal sütunlar var, dışarıdan Shipping_Mode gelmeyecek)
X_sablon = X_clean.iloc[[0]].copy()
for col in X_clean.columns:
    X_sablon[col] = X_clean[col].median()

@app.get("/")
def home():
    return {"message": "Haci Global API is running!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    test_verisi = X_sablon.copy()
    
    # Sadece Şehir ve Kategoriyi al, Shipping_Mode'u sildik!
    sutun_eslesmeleri = {
        "Order_City": "Order City", 
        "Category_Id": "Category Id"
    }
    
    for frontend_key, dataset_key in sutun_eslesmeleri.items():
        if frontend_key in veri and dataset_key in test_verisi.columns:
            test_verisi[dataset_key] = float(veri[frontend_key])

    try:
        olasiliklar = model.predict_proba(test_verisi)[0]
        risk = float(olasiliklar[1] * 100)
    except AttributeError:
        tahmin = model.predict(test_verisi)
        risk = 85.0 if tahmin[0] == 1 else 15.0

    w_desc = veri.get("weather_desc", "").lower()
    
    # Hava durumu düzeltmesi
    if any(x in w_desc for x in ["yağmur", "fırtına", "kar", "rain", "storm", "snow"]):
        risk = min(risk + 25.0, 98.5)
    elif any(x in w_desc for x in ["kapalı", "bulutlu", "clouds", "overcast"]):
        risk = min(risk + 8.0, 98.5)
    elif any(x in w_desc for x in ["açık", "güneş", "clear", "sun"]):
        risk = max(risk - 10.0, 2.0)

    risk_metni = "🚨 HIGH RISK" if risk > 65 else "⚠️ MEDIUM RISK" if risk > 35 else "✅ LOW RISK"
    
    return {
        "risk_skoru": round(risk, 2),
        "Tahmin Sonucu": risk_metni
    }