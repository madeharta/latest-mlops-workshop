"""
Continuous Training (CT) dengan gerbang perbandingan.

Ini BUKAN sekadar "retrain lalu deploy otomatis" -- itu justru anti-pola
yang bisa membuat model produksi memburuk tanpa disadari kalau data baru
kebetulan menghasilkan model yang lebih jelek (mis. akibat data quality
issue, bukan perbaikan sesungguhnya).

Alur yang benar, dan yang diimplementasikan di sini:
  1. Latih kandidat baru dari data terkini (data/train.csv + data/test.csv --
     ganti dengan data baru Anda di sini kalau ini dijalankan sungguhan)
  2. Baca metrik model yang SEDANG dipakai produksi (alias "production")
  3. Bandingkan -- kandidat baru HANYA dipromosikan kalau accuracy-nya
     lebih baik dari production saat ini
  4. Kalau tidak lebih baik: tetap diregistrasi (untuk audit trail / Sesi 4),
     tapi alias "production" TIDAK dipindah -- model lama tetap yang dipakai

Dipanggil oleh orchestration/pipeline.py sebagai task retrain_and_compare(),
yang membaca baris "PROMOTED" / "NOT_PROMOTED" di stdout untuk memutuskan
apakah build_and_deploy_local() perlu dijalankan.

Jalankan manual:
    python3 mlflow/retrain_and_compare.py
"""
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

MODEL_NAME = "insurance-approval-model"
ALIAS = "champion"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Insurance Approval Model")
client = mlflow.MlflowClient()

# --- 1. Latih kandidat baru ---------------------------------------------
# Di kelas: ganti dua baris ini dengan path data baru untuk mensimulasikan
# "data baru datang". Untuk demo, memakai dataset yang sama.
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns=["approved"]), train["approved"]
X_test, y_test = test.drop(columns=["approved"]), test["approved"]

CANDIDATES = {
    "RandomForest-retrained": (RandomForestClassifier, {"n_estimators": 200, "max_depth": 8}),
    "GradientBoosting-retrained": (GradientBoostingClassifier, {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 3}),
    "DecisionTree-retrained": (DecisionTreeClassifier, {"max_depth": 6}),
}

best_acc, best_name, best_run_id = 0, None, None
with mlflow.start_run(run_name="retrain-challenger-batch") as parent:
    for name, (ModelClass, params) in CANDIDATES.items():
        with mlflow.start_run(run_name=name, nested=True) as run:
            model = ModelClass(**params, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            #mlflow.sklearn.log_model(model, name="model")
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="skops",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ],
            )

            print(f"Kandidat {name}: accuracy={acc:.4f}")

            if acc > best_acc:
                best_acc, best_name, best_run_id = acc, name, run.info.run_id

print(f"\nKandidat terbaik: {best_name} (accuracy={best_acc:.4f})")

# --- 2. Baca metrik model production saat ini ----------------------------
try:
    current_prod = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    prod_run = client.get_run(current_prod.run_id)
    prod_acc = prod_run.data.metrics.get("accuracy", 0.0)
    print(f"Model production saat ini: versi {current_prod.version} (accuracy={prod_acc:.4f})")
except mlflow.exceptions.MlflowException:
    # Belum ada model production sama sekali -- jalankan train_with_mlflow.py dulu (Sesi 4)
    print("Belum ada model dengan alias 'production'. Jalankan mlflow/train_with_mlflow.py terlebih dahulu.")
    sys.exit(1)

# --- 3. Registrasi kandidat (SELALU, untuk audit trail) ------------------
result = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
print(f"Kandidat teregistrasi sebagai versi {result.version}")

# --- 4. Bandingkan & putuskan promosi -------------------------------------
if best_acc > prod_acc:
    client.set_registered_model_alias(MODEL_NAME, ALIAS, result.version)
    print(f"\nPROMOTED: versi {result.version} ({best_acc:.4f}) > production lama ({prod_acc:.4f})")
    print(f"Alias 'production' dipindah ke versi {result.version}")
else:
    print(f"\nNOT_PROMOTED: kandidat ({best_acc:.4f}) tidak lebih baik dari production ({prod_acc:.4f})")
    print(f"Alias 'production' TETAP di versi {current_prod.version}")
