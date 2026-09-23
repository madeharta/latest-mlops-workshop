"""
Sesi 4 -> Sesi 6 -- Menjembatani Model Registry ke container.

app.py (Sesi 5) bisa memuat model LANGSUNG dari MLflow Registry saat
development, karena tracking store (mlflow.db + folder artifacts) ada di
mesin yang sama. Tapi container Docker (Sesi 6) TIDAK seharusnya bergantung
pada tracking store developer -- image harus mandiri (self-contained).

Solusinya: export model dari registry menjadi satu file statis models/model.pkl
sebelum build image. app.py akan otomatis memakai file ini jika tersedia
(lihat fallback logic di api/app.py).

Menarik lewat alias "production" (bukan nomor versi tetap) -- alias inilah
yang dipindahkan mlflow/retrain_and_compare.py (Sesi 8) setiap kali model
baru terbukti lebih baik, sehingga skrip ini SELALU mengekspor model
production yang terbaru tanpa perlu diedit manual.

Jalankan (setelah mlflow/train_with_mlflow.py selesai dan model teregistrasi):
    python3 mlflow/export_model.py
"""
import os
import joblib
import mlflow.sklearn

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://127.0.0.1:9001")

MODEL_NAME = "insurance-approval-model"
ALIAS = "champion"

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

os.makedirs("api/models", exist_ok=True)
joblib.dump(model, "api/models/model.pkl")
print(f"Model '{MODEL_NAME}' (alias '{ALIAS}') berhasil diekspor ke api/models/model.pkl")
