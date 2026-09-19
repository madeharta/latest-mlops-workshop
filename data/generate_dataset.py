"""
Menghasilkan dataset sintetis 'Insurance Approval' yang dipakai di seluruh
materi Sesi 1-7 (train.csv, test.csv, production.csv).

File CSV hasilnya sudah disertakan langsung di folder data/, jadi menjalankan
script ini TIDAK WAJIB. Script ini disediakan untuk transparansi -- agar
mahasiswa bisa melihat persis bagaimana data dibuat, dan agar instruktur bisa
menghasilkan variasi baru (mis. ganti random_state) jika diperlukan.

production.csv sengaja diberi pergeseran distribusi pada tiga kolom
(income, credit_score, num_claims) untuk mensimulasikan data drift yang
akan dideteksi di Sesi 7 dengan Evidently AI.

Jalankan dari root proyek:
    python3 data/generate_dataset.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

X, y = make_classification(
    n_samples=2000, n_features=8, n_informative=6, n_redundant=1, random_state=42
)
cols = [
    "age", "income", "bmi", "tenure_months",
    "num_claims", "credit_score", "policy_value", "risk_score",
]
df = pd.DataFrame(X, columns=cols)

# Skalakan ke rentang yang realistis untuk domain asuransi
df["age"] = (df["age"] * 8 + 45).clip(18, 85).round(0)
df["income"] = (df["income"] * 15000 + 65000).clip(15000, 250000).round(0)
df["bmi"] = (df["bmi"] * 4 + 26).clip(15, 45).round(1)
df["tenure_months"] = (df["tenure_months"] * 20 + 36).clip(1, 240).round(0)
df["num_claims"] = (df["num_claims"] * 1.5 + 2).clip(0, 15).round(0)
df["credit_score"] = (df["credit_score"] * 80 + 680).clip(300, 850).round(0)
df["policy_value"] = (df["policy_value"] * 40000 + 120000).clip(10000, 500000).round(0)
df["risk_score"] = (df["risk_score"] * 15 + 50).clip(0, 100).round(1)
df["approved"] = y

train = df.iloc[:1500].reset_index(drop=True)
test = df.iloc[1500:].reset_index(drop=True)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

# Data "produksi" dengan drift yang disengaja pada 3 kolom -- dipakai di Sesi 7
prod = test.copy()
prod["income"] = prod["income"] * 1.35 + 8000
prod["credit_score"] = (prod["credit_score"] - 40).clip(300, 850)
prod["num_claims"] = prod["num_claims"] + 2
prod.to_csv("data/production.csv", index=False)

print(f"train.csv      : {train.shape}")
print(f"test.csv       : {test.shape}")
print(f"production.csv : {prod.shape} (drift disengaja: income, credit_score, num_claims)")
