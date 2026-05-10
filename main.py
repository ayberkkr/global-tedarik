import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

df = pd.read_csv(r'C:\Users\ayber\OneDrive\Masaüstü\hepsi\global-tedarik-set\DataCoSupplyChainDataset.csv', encoding='latin1')

gereksiz_sutunlar = [
    'Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname', 
    'Product Image', 'Customer Street', 'Order Zipcode'
]
df = df.drop(columns=gereksiz_sutunlar)

hileli_sutunlar = ['Days for shipping (real)', 'Delivery Status']
df = df.drop(columns=hileli_sutunlar)

hedef = 'Late_delivery_risk'
Y = df[hedef]
X = df.drop(columns=[hedef])

kategorik_sutunlar = X.select_dtypes(include=['object']).columns.tolist()

X[kategorik_sutunlar] = X[kategorik_sutunlar].fillna("Bilinmiyor")
X = X.fillna(0)

le = LabelEncoder()
for col in kategorik_sutunlar:
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='logloss')
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)

ensemble_model = VotingClassifier(
    estimators=[('cat', cat_model), ('xgb', xgb_model), ('lgb', lgb_model)],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Modelimizi hiç görmediği test verisiyle (X_test) sınıyoruz
y_tahmin = ensemble_model.predict(X_test)

# 2. Başarı oranını hesaplıyoruz
basari_orani = accuracy_score(y_test, y_tahmin)
print(f"\nModelin Doğruluk Oranı (Accuracy): %{basari_orani * 100:.2f}")

# 3. Detaylı karne (Hangi sınıfta ne kadar başarılı?)
print("\nDetaylı Karne:")
print(classification_report(y_test, y_tahmin))

# 4. Modeli Frontend'e (arayüze) bağlamak için kaydediyoruz!
joblib.dump(ensemble_model, 'haci_risk_model.joblib')
print("\nModel başarıyla 'haci_risk_model.joblib' olarak kaydedildi!")