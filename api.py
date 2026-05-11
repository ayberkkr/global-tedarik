from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

X_sablon = X_clean.iloc[[0]].copy()
for col in X_clean.columns:
    X_sablon[col] = X_clean[col].median()

@app.get("/")
def home():
    return {"message": "Haci Global API is running!"}

@app.post("/tahmin_et")
def risk_tahmin_et(veri: dict):
    test_verisi = X_sablon.copy()
    
    sutun_eslesmeleri = {
        "Shipping_Mode": "Shipping Mode",
        "Order_City": "Order City", 
        "Category_Id": "Category Id"
    }
    
    for frontend_key, dataset_key in sutun_eslesmeleri.items():
        if frontend_key in veri and dataset_key in test_verisi.columns:
            test_verisi[dataset_key] = veri[frontend_key]

    # Riski hesaplayan iç fonksiyon (Tekrar tekrar kullanabilmek için)
    def calculate_base_risk(df_input, w_desc):
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

    # Kullanıcının seçtiği mod için risk hesapla
    current_mode = veri.get("Shipping_Mode", "Standard Class")
    current_risk = calculate_base_risk(test_verisi, veri.get("weather_desc", ""))

    # Lojistik Danışmanı: Diğer taşıma modlarını gizlice test et
    all_modes = {
        "Ocean Freight (Ship)": "Standard Class",
        "Road Freight (Truck)": "Second Class",
        "Air Freight (Plane)": "Same Day"
    }

    best_alt_mode = None
    best_alt_risk = current_risk

    for display_name, dataset_val in all_modes.items():
        if dataset_val != current_mode:
            alt_test = test_verisi.copy()
            alt_test["Shipping Mode"] = dataset_val
            alt_risk = calculate_base_risk(alt_test, veri.get("weather_desc", ""))
            
            # Eğer diğer mod daha düşük riskliyse, onu en iyi alternatif yap
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

    # Eğer bulduğumuz alternatif mod, şu anki riskten %5 daha iyiyse tavsiye ver
    if best_alt_mode and (current_risk - best_alt_risk > 5.0):
        response_data["Tavsiye"] = f"💡 AI Tip: Using {best_alt_mode} reduces the risk to {round(best_alt_risk, 2)}%."

    return response_data