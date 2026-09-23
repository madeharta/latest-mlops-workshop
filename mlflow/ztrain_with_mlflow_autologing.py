"""
Instrumentasi training dengan MLflow: tracking, perbandingan
eksperimen, dan registrasi model terbaik ke Model Registry.

PENTING (lihat README / slide gotcha Sesi 4): backend tracking berbasis file
sederhana (file:///...) sudah masuk "maintenance mode" pada MLflow versi
terbaru dan bisa ditolak langsung. Kita pakai backend SQLite, yang tetap
sepenuhnya lokal (satu file mlflow.db) tapi didukung penuh.

Jalankan:
    python3 mlflow/train_with_mlflow.py

Lalu buka dashboard (dari root proyek):
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 9001
    # buka http://127.0.0.1:9001

Skrip ini juga menandai model terbaik dengan alias registry "production" --
inilah baseline yang nanti ditantang oleh mlflow/retrain_and_compare.py
(Sesi 8) setiap kali data baru datang.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Insurance Approval Model")

# Enable autologging
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns=["approved"]), train["approved"]
X_test, y_test = test.drop(columns=["approved"]), test["approved"]

MODELS = {
    "DecisionTree": (DecisionTreeClassifier, {"max_depth": 6}),
    "RandomForest": (RandomForestClassifier, {"n_estimators": 200, "max_depth": 8}),
    "GradientBoosting": (GradientBoostingClassifier, {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 3}),
}

best_acc, best_name, best_run_id = 0, None, None

for name, (ModelClass, params) in MODELS.items():
    with mlflow.start_run(run_name=name) as run:
        model = ModelClass(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, preds)

        #for k, v in params.items():
        #    mlflow.log_param(k, v)
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)

        #mlflow.sklearn.log_model(model, name="model")
        # Log model
        #mlflow.sklearn.log_model(
        #    model,
        #    name="model",
        #    serialization_format="skops",
        #    skops_trusted_types=[
        #        "sklearn.tree._tree.Tree"
        #    ],
        #)

        print(f"{name}: acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}")

        if acc > best_acc:
            best_acc, best_name, best_run_id = acc, name, run.info.run_id

print(f"\nBEST: {best_name} {best_run_id}")

# Registrasi model terbaik ke Model Registry
result = mlflow.register_model(f"runs:/{best_run_id}/model", "insurance-approval-model")
print(f"Registered version: {result.version}")

# Tandai sebagai model yang sedang "dilayani" -- alias ini yang akan dibaca
# api/app.py (Sesi 5), mlflow/export_model.py (Sesi 6), dan ditantang oleh
# mlflow/retrain_and_compare.py (Sesi 8) setiap kali data baru datang.
client = mlflow.MlflowClient()
client.set_registered_model_alias("insurance-approval-model", "champion", result.version)
print(f"Alias 'champion' -> versi {result.version}")
