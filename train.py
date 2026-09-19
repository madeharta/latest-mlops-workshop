"""
Sesi 2 -- Pipeline training sederhana yang dikendalikan oleh config.yml.
Mengganti model cukup mengubah config.yml, tanpa menyentuh file ini.

Jalankan:
    python3 train.py
    # atau:
    make run
"""
import yaml
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

MODELS = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "DecisionTree": DecisionTreeClassifier,
}


def main():
    cfg = yaml.safe_load(open("config.yml"))

    train = pd.read_csv(cfg["data"]["train_path"])
    test = pd.read_csv(cfg["data"]["test_path"])
    target = cfg["data"]["target"]

    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    ModelClass = MODELS[cfg["model"]["name"]]
    model = ModelClass(**cfg["model"]["params"], random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"Model      : {cfg['model']['name']}")
    print(f"Accuracy   : {accuracy_score(y_test, preds):.4f}")
    print(f"ROC-AUC    : {roc_auc_score(y_test, proba):.4f}")

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
