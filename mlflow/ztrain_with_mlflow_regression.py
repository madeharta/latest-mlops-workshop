"""
Sesi 4 -- Instrumentasi training dengan MLflow: tracking, perbandingan
eksperimen, dan registrasi model terbaik ke Model Registry.

PENTING (lihat README / slide gotcha Sesi 4): backend tracking berbasis file
sederhana (file:///...) sudah masuk "maintenance mode" pada MLflow versi
terbaru dan bisa ditolak langsung. Kita pakai backend SQLite, yang tetap
sepenuhnya lokal (satu file mlflow.db) tapi didukung penuh.

Jalankan:
    python3 mlflow/train_with_mlflow.py

Lalu buka dashboard (dari root proyek):
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    # buka http://127.0.0.1:5001
    # (macOS: JANGAN pakai port 5000 -- dipakai AirPlay Receiver sejak Monterey)

Skrip ini juga menandai model terbaik dengan alias registry "production" --
inilah baseline yang nanti ditantang oleh mlflow/retrain_and_compare.py
(Sesi 8) setiap kali data baru datang.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://127.0.0.1:9001")
print("MLFlow URI is configured.")
mlflow.set_experiment("Risk Score Regression")
print("MLFlow experiment is set.")
# Enable autologging
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns=["approved", "risk_score"]), train["risk_score"]
X_test, y_test = test.drop(columns=["approved", "risk_score"]), test["risk_score"]

MODELS = {
    "LinearRegression": (LinearRegression,{}),
    "DecisionTree": (DecisionTreeRegressor,{"max_depth": 6,"random_state": 42}),
    "RandomForest": (RandomForestRegressor,{"n_estimators": 200, "max_depth": 8,"random_state": 42}),
}

best_r2, best_name, best_run_id = 0, None, None

for name, (ModelClass, params) in MODELS.items():
    with mlflow.start_run(run_name=name) as run:
        model = ModelClass(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        print(f"{name}: mae={mae:.4f} mse={mse:.4f} r2={r2:.4f}")

        if r2 > best_r2:
            best_r2, best_name, best_run_id = r2, name, run.info.run_id

print(f"\nBEST: {best_name} {best_run_id}")

client = mlflow.MlflowClient()

try:
    champion = client.get_model_version_by_alias("risk-score-regression","champion")
    champion_version = champion.version

    champion_run = client.get_run(champion.run_id)
    # If you have new data, it is better usually to run inference to the champion model with the new data first
    # Then, you can evaluate based on this new inference result. For this implementation, we will just refer old data
    champion_r2 = champion_run.data.metrics["R2"]
    champion_mae = champion_run.data.metrics["MAE"]
    champion_mse = champion_run.data.metrics["MSE"]

except Exception:
    champion = None
    print("\nNo existing champion model found.")

if champion is None:
    # No champion yet → automatically promote candidate
    promote = True
else:
    if best_r2 > champion_r2:
        promote = True
    else:
        promote = False

if promote:

    # Registrasi model terbaik ke Model Registry
    result = mlflow.register_model(f"runs:/{best_run_id}/model", "risk-score-regression")

    print("\n======================================")
    print("Registering Candidate")
    print("======================================")

    print(f"Registered version: {result.version}")

    # Move champion alias to the new version
    client.set_registered_model_alias("risk-score-regression", "champion", result.version)
    print(f"Alias 'champion' -> versi {result.version}")

    print(
        f"Alias 'champion' -> version {result.version}"
    )

else:

    print("\n======================================")
    print("Promotion Rejected")
    print("======================================")

    if champion is not None:
        print(
            f"Champion remains version {champion.version}"
        )




