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

# Şablon Değerlerini ve En Önemlisi SÜTUN SIRASINI Kaydediyoruz
sutun_sirasi = list(X_clean.columns)
sablon_degerleri = {col: float(X_clean[col].median()) for col in sutun_sirasi}

@app.get("/")
def home():
    return {"message": "Haci Global API is running!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    
    # 1. TERCÜMAN
    mode_map = {
        "First Class": 0.0,
        "Same Day": 1.0,
        "Second Class": 2.0,
        "Standard Class": 3.0
    }
    
    gelen_mod_str = str(veri.get("Shipping_Mode", "Standard Class"))
    secili_mod_degeri = float(gelen_mod_str) if gelen_mod_str.isdigit() else float(mode_map.get(gelen_mod_str, 3.0))

    order_city_val = float(veri.get("Order_City", sablon_degerleri["Order City"]))
    category_id_val = float(veri.get("Category_Id", sablon_degerleri["Category Id"]))

    # 2. RİSK HESAPLAMA MOTORU (Numpy Balyozu)
    def calculate_base_risk(mode_val, city_val, cat_val, w_desc):
        # Önce değerleri sözlüğe yaz
        anlik_degerler = sablon_degerleri.copy()
        anlik_degerler["Shipping Mode"] = mode_val
        anlik_degerler["Order City"] = city_val
        anlik_degerler["Category Id"] = cat_val
        
        # Sütun sırasını BOZMADAN sadece değerleri bir listeye (array) diziyoruz!
        sadece_sayilar = []
        for col in sutun_sirasi:
            sadece_sayilar.append(anlik_degerler[col])
            
        # CatBoost'a Pandas DataFrame yerine, 2 boyutlu çıplak bir SAYI DİZİSİ yolluyoruz.
        # Böylece "Hani nerede Shipping Mode?" diye soramayacak!
        saf_array = np.array([sadece_sayilar], dtype=float)

        try:
            olasiliklar = model.predict_proba(saf_array)[0]
            risk = float(olasiliklar[1] * 100)
        except AttributeError:
            tahmin = model.predict(saf_array)
            risk = 85.0 if tahmin[0] == 1 else 15.0

        w_desc_low = w_desc.lower()
        if any(x in w_desc_low for x in ["yağmur", "fırtına", "kar", "rain", "storm", "snow"]):
            risk = min(risk + 25.0, 98.5)
        elif any(x in w_desc_low for x in ["kapalı", "bulutlu", "clouds", "overcast"]):
            risk = min(risk + 8.0, 98.5)
        elif any(x in w_desc_low for x in ["açık", "güneş", "clear", "sun"]):
            risk = max(risk - 10.0, 2.0)
        return risk

    current_risk = calculate_base_risk(secili_mod_degeri, order_city_val, category_id_val, veri.get("weather_desc", ""))

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