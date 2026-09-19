"""
Sesi 7 -- Monitoring model di produksi: tiga sinyal, bukan satu.

Banyak tutorial MLOps berhenti di "data drift" saja. Materi ini sengaja
menunjukkan tiga sinyal yang berbeda sifatnya, karena satu sinyal saja bisa
menyesatkan (lihat temuan nyata di bagian bawah skrip ini):

  1. DATA DRIFT       -- distribusi fitur INPUT berubah. Tidak butuh label
                          ground truth, jadi bisa dihitung real-time begitu
                          request masuk.
  2. PREDICTION DRIFT  -- distribusi OUTPUT model berubah. Juga tidak butuh
                          ground truth -- hanya butuh model untuk menghasilkan
                          prediksi pada data reference & current.
  3. MODEL/CONCEPT DRIFT (performance) -- akurasi, precision, recall, ROC-AUC
                          model SESUNGGUHNYA dibandingkan reference vs current.
                          Ini SATU-SATUNYA sinyal yang butuh ground truth
                          (kolom 'approved') -- di produksi nyata, label ini
                          biasanya baru tersedia belakangan (label latency:
                          klaim asuransi baru diputuskan berbulan-bulan
                          setelah polis diajukan), sehingga sinyal ini paling
                          akurat tapi paling lambat datangnya.

TEMUAN NYATA dari data insurance-approval yang dipakai di materi ini:
data drift TERDETEKSI (3/8 kolom), tapi prediction drift TIDAK terdeteksi
secara statistik -- padahal akurasi model SUNGGUHAN turun dari 0.938 ke
0.888 begitu diukur dengan ground truth. Prediction drift saja tidak cukup
sensitif menangkap penurunan performa yang nyata terjadi di sini -- inilah
kenapa monitoring produksi butuh lebih dari satu sinyal.

⚠ CATATAN VERSI (lihat README Sesi 7 untuk riwayat lengkap): materi ini
sempat memakai evidently==0.4.40, rilis lama yang TIDAK mendeklarasikan
batas atas plotly -- menyebabkan pip menarik plotly 7.x yang menghapus
figure_factory.create_distplot (dipanggil ClassificationPreset), dan juga
riskan bentrok numpy di Python 3.14+. Materi ini sekarang memakai
evidently==0.7.21 (rilis aktif dipelihara, plotly<6 & numpy tanpa batas
atas dideklarasikan dengan benar) -- TAPI tetap memakai API lama yang sama
persis (Report/DataDriftPreset/ColumnMapping) lewat compatibility shim
`evidently.legacy`, sehingga kode di bawah ini TIDAK BERUBAH sama sekali
selain empat baris import.

Jalankan:
    python3 monitoring/monitor.py

Output: monitoring/drift_report.html -- buka di browser atau lewat
VS Code Live Preview.
"""
import os
import joblib
import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.legacy.metrics import ColumnDriftMetric

FEATURES = ["age", "income", "bmi", "tenure_months", "num_claims", "credit_score", "policy_value", "risk_score"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "models", "model.pkl")

# --- Muat model -- pola fallback yang sama dengan api/app.py (Sesi 5) -----
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    import mlflow.sklearn
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model("models:/insurance-approval-model@production")

# Reference: representasi data yang dipakai saat training/evaluasi (Sesi 2-4)
reference = pd.read_csv("data/test.csv").copy()

# Current: data produksi -- idealnya log request dari endpoint lokal (Sesi 6),
# bukan file statis seperti contoh ini.
current = pd.read_csv("data/production.csv").copy()

# Hasilkan prediksi pada KEDUA dataset -- inilah yang membedakan prediction
# drift & performance monitoring dari data drift biasa.
for df in (reference, current):
    df["prediction"] = model.predict(df[FEATURES])
    df["prediction_proba"] = model.predict_proba(df[FEATURES])[:, 1]

column_mapping = ColumnMapping(target="approved", prediction="prediction_proba", pos_label=1)

report = Report(metrics=[
    DataDriftPreset(columns=FEATURES),           # sinyal 1 -- HANYA 8 fitur input
    ColumnDriftMetric(column_name="prediction"),  # sinyal 2 -- output model
    ClassificationPreset(),                       # sinyal 3 -- butuh ground truth
])
report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

os.makedirs("monitoring", exist_ok=True)
report.save_html("monitoring/drift_report.html")
print("Laporan tersimpan di monitoring/drift_report.html\n")

result = report.as_dict()
data_drift_share, n_drifted, n_total = None, None, None
prediction_drift_detected = None
ref_acc = cur_acc = ref_auc = cur_auc = None

for m in result["metrics"]:
    if m["metric"] == "DataDriftTable":
        n_drifted = m["result"]["number_of_drifted_columns"]
        n_total = m["result"]["number_of_columns"]
        data_drift_share = m["result"]["share_of_drifted_columns"]
    elif m["metric"] == "ColumnDriftMetric":
        prediction_drift_detected = m["result"]["drift_detected"]
    elif m["metric"] == "ClassificationQualityMetric":
        ref_acc = m["result"]["reference"]["accuracy"]
        cur_acc = m["result"]["current"]["accuracy"]
        ref_auc = m["result"]["reference"]["roc_auc"]
        cur_auc = m["result"]["current"]["roc_auc"]

print(f"1. DATA DRIFT        -- {n_drifted}/{n_total} kolom input drift (share: {data_drift_share:.3f})")
print(f"2. PREDICTION DRIFT  -- {'terdeteksi' if prediction_drift_detected else 'tidak terdeteksi'} pada output model")
print(f"3. MODEL/CONCEPT DRIFT (performance) -- accuracy: {ref_acc:.4f} -> {cur_acc:.4f}  |  ROC-AUC: {ref_auc:.4f} -> {cur_auc:.4f}")

# --- Kebijakan retraining: OR dari dua ambang, bukan cuma satu sinyal -----
DRIFT_THRESHOLD = 0.30       # >=30% kolom input drift
ACCURACY_DROP_THRESHOLD = 0.03  # penurunan accuracy >=3 poin persentase

accuracy_drop = ref_acc - cur_acc
drift_trigger = data_drift_share >= DRIFT_THRESHOLD
performance_trigger = accuracy_drop >= ACCURACY_DROP_THRESHOLD

print()
if drift_trigger or performance_trigger:
    alasan = []
    if drift_trigger:
        alasan.append(f"data drift {data_drift_share:.2f} >= ambang {DRIFT_THRESHOLD}")
    if performance_trigger:
        alasan.append(f"accuracy turun {accuracy_drop:.4f} >= ambang {ACCURACY_DROP_THRESHOLD}")
    print(f"⚠ Kebijakan retraining melewati ambang ({'; '.join(alasan)}) -- pertimbangkan retraining.")
else:
    print("Kedua sinyal masih di bawah ambang -- belum perlu retraining.")
