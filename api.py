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

# Her kategorik sütun için ayrı bir LabelEncoder tutuyoruz ki veriyi doğru dönüştürebilelim
encoders = {}
for col in kategorik_sutunlar:
    le = LabelEncoder()
    # Model eğitilirken veri nasıl string yapıldıysa, burada da aynı şekilde fit ediyoruz.
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))
    encoders[col] = le

# Şablonu, her sütunun median değeri ile oluşturuyoruz (sayısal)
X_sablon = X_clean.iloc[[0]].copy()
for col in X_clean.columns:
    X_sablon[col] = X_clean[col].median()

@app.get("/")
def home():
    return {"message": "Haci Global API is running!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    
    # Şablonu kopyalayarak başlıyoruz. Bu şablon tamamen dönüştürülmüş sayısal verilerden oluşur.
    test_verisi = X_sablon.copy()
    
    # 1. TERCÜMAN (İngilizce kelimeleri veri setindeki orijinal metinlerine çevirir)
    mode_map = {
        "First Class": "First Class",
        "Same Day": "Same Day",
        "Second Class": "Second Class",
        "Standard Class": "Standard Class"
    }
    
    gelen_mod_str = str(veri.get("Shipping_Mode", "Standard Class"))
    orijinal_mod_metni = mode_map.get(gelen_mod_str, "Standard Class")

    # 2. VERİLERİ ŞABLONA YERLEŞTİRME VE DÖNÜŞTÜRME
    # Order City ve Category Id zaten sayısal, direkt şablona yazıyoruz
    test_verisi["Order City"] = float(veri.get("Order_City", X_sablon["Order City"].values[0]))
    test_verisi["Category Id"] = float(veri.get("Category_Id", X_sablon["Category Id"].values[0]))
    
    # 💡 KRİTİK NOKTA: Shipping Mode'u, modeli eğitirken kullandığımız encoder ile sayıya çeviriyoruz!
    # Eğer "Shipping Mode" sütunu kategorik ise (ki öyle):
    if "Shipping Mode" in encoders:
        try:
            # Sadece bu kelimeyi dönüştür
            sayisal_deger = encoders["Shipping Mode"].transform([orijinal_mod_metni])[0]
            test_verisi["Shipping Mode"] = float(sayisal_deger)
        except ValueError:
            # Bilinmeyen bir değer gelirse median kullan
            test_verisi["Shipping Mode"] = X_sablon["Shipping Mode"].values[0]
    else:
        # Değilse (olası değil ama güvenlik için), direkt yazmayı dene
         test_verisi["Shipping Mode"] = float(veri.get("Shipping_Mode", 3.0))


    # 3. RİSK HESAPLAMA MOTORU
    def calculate_base_risk(df_input, w_desc):
        # Dataframe'in tüm verilerini float'a çevirdiğimizden emin oluyoruz
        df_numeric = df_input.astype(float)
        
        try:
            olasiliklar = model.predict_proba(df_numeric)[0]
            risk = float(olasiliklar[1] * 100)
        except AttributeError:
            tahmin = model.predict(df_numeric)
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
    current_risk = calculate_base_risk(test_verisi, veri.get("weather_desc", ""))

    # 4. Lojistik Danışmanı (Tavsiye Motoru)
    all_modes = {
        "Ocean Freight (Ship)": "Standard Class",
        "Road Freight (Truck)": "Second Class",
        "Air Freight (Plane)": "Same Day"
    }

    best_alt_mode = None
    best_alt_risk = current_risk

    for display_name, mode_str in all_modes.items():
        if mode_str != gelen_mod_str:
            alt_test = test_verisi.copy()
            orijinal_alt_mod_metni = mode_map.get(mode_str, "Standard Class")
            
            # Alternatif modu da encoder ile dönüştür
            if "Shipping Mode" in encoders:
                try:
                    sayisal_deger = encoders["Shipping Mode"].transform([orijinal_alt_mod_metni])[0]
                    alt_test["Shipping Mode"] = float(sayisal_deger)
                except ValueError:
                    alt_test["Shipping Mode"] = X_sablon["Shipping Mode"].values[0]
            else:
                 alt_test["Shipping Mode"] = float(mode_map.get(mode_str, 3.0))

            alt_risk = calculate_base_risk(alt_test, veri.get("weather_desc", ""))
            
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