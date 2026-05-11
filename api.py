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

le = LabelEncoder()
for col in kategorik_sutunlar:
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))

# Şablonu sadece değerleri (median) tutmak için bir sözlük olarak kaydediyoruz
sablon_degerleri = {}
for col in X_clean.columns:
    sablon_degerleri[col] = float(X_clean[col].median())

@app.get("/")
def home():
    return {"message": "Haci Global API is running!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    
    # 1. TERCÜMAN (İngilizce kelimeleri kesin rakamlara çevirir)
    mode_map = {
        "First Class": 0.0,
        "Same Day": 1.0,
        "Second Class": 2.0,
        "Standard Class": 3.0
    }
    
    gelen_mod_str = str(veri.get("Shipping_Mode", "Standard Class"))
    
    if gelen_mod_str.isdigit():
        secili_mod_degeri = float(gelen_mod_str)
    else:
        secili_mod_degeri = float(mode_map.get(gelen_mod_str, 3.0))

    order_city_val = float(veri.get("Order_City", sablon_degerleri["Order City"]))
    category_id_val = float(veri.get("Category_Id", sablon_degerleri["Category Id"]))

    # 2. RİSK HESAPLAMA MOTORU (Sıfırdan DataFrame yaratır)
    def calculate_base_risk(mode_val, city_val, cat_val, w_desc):
        
        # Orijinal sütun sırasına göre yepyeni, tertemiz bir sözlük oluşturuyoruz
        yeni_veri = {}
        for col in X_clean.columns:
            yeni_veri[col] = [sablon_degerleri[col]] # Median değerleri varsayılan olarak koy
            
        # Kullanıcının seçtiği özellikleri üzerine yaz
        yeni_veri["Shipping Mode"] = [float(mode_val)]
        yeni_veri["Order City"] = [float(city_val)]
        yeni_veri["Category Id"] = [float(cat_val)]
        
        # Sıfırdan, saf matematik olan bir DataFrame oluştur
        df_input = pd.DataFrame(yeni_veri)

        try:
            olasiliklar = model.predict_proba(df_input)[0]
            risk = float(olasiliklar[1] * 100)
        except AttributeError:
            tahmin = model.predict(df_input)
            risk = 85.0 if tahmin[0] == 1 else 15.0

        w_desc_low = w_desc.lower()
        if any(x in w_desc_low for x in ["yağmur", "fırtına", "kar", "rain", "storm", "snow"]):
            risk = min(risk + 25.0, 98.5)
        elif any(x in w_desc_low for x in ["kapalı", "bulutlu", "clouds", "overcast"]):
            risk = min(risk + 8.0, 98.5)
        elif any(x in w_desc_low for x in ["açık", "güneş", "clear", "sun"]):
            risk = max(risk - 10.0, 2.0)
        return risk

    # Şu anki riski hesapla
    current_risk = calculate_base_risk(secili_mod_degeri, order_city_val, category_id_val, veri.get("weather_desc", ""))

    # 3. Lojistik Danışmanı (Tavsiye Motoru)
    all_modes = {
        "Ocean Freight (Ship)": "Standard Class",
        "Road Freight (Truck)": "Second Class",
        "Air Freight (Plane)": "Same Day"
    }

    best_alt_mode = None
    best_alt_risk = current_risk

    for display_name, mode_str in all_modes.items():
        if mode_str != gelen_mod_str:
            alt_mode_val = float(mode_map.get(mode_str, 3.0))
            alt_risk = calculate_base_risk(alt_mode_val, order_city_val, category_id_val, veri.get("weather_desc", ""))
            
            if alt_risk < best_alt_risk:
                best_alt_risk = alt_risk
                best_alt_mode = display_name

    risk_metni = "🚨 HIGH RISK" if current_risk > 65 else "⚠️ MEDIUM RISK" if current_risk > 35 else "✅ LOW RISK"
    
    response_data = {
        "status": "Success",
        "risk_skoru": round(current_risk, 2),
        "Tahmin Sonucu": risk_metni,
        "Tavsiye": ""
    }

    if best_alt_mode and (current_risk - best_alt_risk > 5.0):
        response_data["Tavsiye"] = f"💡 AI Tip: Using {best_alt_mode} reduces the risk to {round(best_alt_risk, 2)}%."

    return response_data