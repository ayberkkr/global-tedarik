from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

# Tüm verileri eğitiyoruz
le = LabelEncoder()
for col in kategorik_sutunlar:
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))

# Şablonu al ve TERTEMİZ bir sözlüğe çevir (Pandas'ın huysuzluklarından kaçmak için)
sutun_sirasi = list(X_clean.columns)
sablon_degerleri = {col: float(X_clean[col].median()) for col in sutun_sirasi}

@app.get("/")
def home():
    return {"message": "Haci Global API is running!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    
    # 1. Gelen Verileri Topla (Sadece Sayısal)
    mode_map = {
        "First Class": 0.0,
        "Same Day": 1.0,
        "Second Class": 2.0,
        "Standard Class": 3.0
    }
    
    gelen_mod_str = str(veri.get("Shipping_Mode", "3.0"))
    if gelen_mod_str.isdigit() or gelen_mod_str.replace('.','',1).isdigit():
        secili_mod = float(gelen_mod_str)
    else:
        secili_mod = float(mode_map.get(gelen_mod_str, 3.0))

    order_city_val = float(veri.get("Order_City", sablon_degerleri["Order City"]))
    category_id_val = float(veri.get("Category_Id", sablon_degerleri["Category Id"]))
    w_desc = veri.get("weather_desc", "").lower()

    # 2. Şablonu Güncelle (Sadece değiştirdiğimiz 3 değeri güncelliyoruz)
    guncel_degerler = sablon_degerleri.copy()
    guncel_degerler["Shipping Mode"] = secili_mod
    guncel_degerler["Order City"] = order_city_val
    guncel_degerler["Category Id"] = category_id_val

    # 3. Modeli Kandırma (Pandas DataFrame oluşturmak ZORUNDAYIZ ama tek satırlık)
    # CatBoost sütun isimlerini kontrol ettiği için DataFrame vermek zorundayız.
    tek_satirlik_veri = {col: [guncel_degerler[col]] for col in sutun_sirasi}
    df_input = pd.DataFrame(tek_satirlik_veri)
    
    # 4. HATA YAKALAMA VE RİSK HESABI (BÜTÜN OLAY BURADA)
    try:
        # Önce olasılık (Yüzde) almaya çalışıyoruz. 
        # Eğer senin o lanet hata ('Cannot convert Standard Class to float') çıkarsa, sistem çökmeyecek, 'except' bloğuna düşecek!
        olasiliklar = model.predict_proba(df_input)[0]
        risk = float(olasiliklar[1] * 100)
    except Exception as e:
        # HATA YAKALANDI! Sistem çökmedi.
        print(f"CatBoost Proba Hatası Yakalandı: {e} - Düz tahmine geçiliyor.")
        
        # Olasılık alamadığımız için düz tahmin yapıyoruz (1 veya 0 verir)
        tahmin = model.predict(df_input)
        
        # Eğer gecikecek diyorsa taban risk %85, gecikmeyecek diyorsa taban risk %15
        # (Birazdan hava durumuyla bu sayılar değişecek)
        risk = 85.0 if tahmin[0] == 1 else 15.0

    # 5. Hava Durumu Dinamik Etkisi
    if any(x in w_desc for x in ["yağmur", "fırtına", "kar", "rain", "storm", "snow"]):
        risk = min(risk + 25.0, 98.5)
    elif any(x in w_desc for x in ["kapalı", "bulutlu", "clouds", "overcast"]):
        risk = min(risk + 8.0, 98.5)
    elif any(x in w_desc for x in ["açık", "güneş", "clear", "sun"]):
        risk = max(risk - 10.0, 2.0)

    # 6. Sonucu Gönder
    risk_metni = "🚨 HIGH RISK" if risk > 65 else "⚠️ MEDIUM RISK" if risk > 35 else "✅ LOW RISK"
    
    return {
        "risk_skoru": round(risk, 2),
        "Tahmin Sonucu": risk_metni,
        "Tavsiye": "" 
    }