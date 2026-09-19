"""
Sesi 2 -- Contoh unit test sederhana, dijalankan via `make test` / `pytest tests/ -v`.
Memverifikasi bahwa pipeline training menghasilkan model dengan performa
di atas baseline minimum -- bukan menguji angka akurasi yang presisi (yang
akan berubah setiap kali data atau parameter berubah), melainkan ambang
kewarasan (sanity threshold).
"""
import subprocess
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def test_train_script_runs_and_produces_model():
    result = subprocess.run(
        [sys.executable, "train.py"], capture_output=True, text=True, cwd="."
    )
    assert result.returncode == 0, f"train.py gagal dijalankan:\n{result.stderr}"
    assert "Model saved to models/model.pkl" in result.stdout


def test_model_accuracy_above_baseline():
    model = joblib.load("models/model.pkl")
    test = pd.read_csv("data/test.csv")
    X_test, y_test = test.drop(columns=["approved"]), test["approved"]
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # Ambang kewarasan -- bukan angka final, hanya memastikan model tidak "rusak"
    assert acc > 0.7, f"Accuracy {acc:.4f} di bawah ambang minimum 0.70"
