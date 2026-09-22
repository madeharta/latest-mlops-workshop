"""
Monitoring model di produksi

Terdapat tiga sinyal yang digunakan untuk mengecek apakah proses 
retrainning model diperlukan atau tidak, yaitu:

  1. DATA DRIFT       -- distribusi fitur INPUT berubah. Tidak butuh label
                          ground truth, jadi bisa dihitung real-time begitu
                          request masuk.
  2. PREDICTION DRIFT  -- distribusi OUTPUT model berubah. Juga tidak butuh
                          ground truth -- hanya butuh model untuk menghasilkan
                          prediksi pada data reference & current.
  3. MODEL PERFORMANCE    (performance) -- akurasi, precision, recall, ROC-AUC
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

Jalankan:
    python3 monitoring/monitor.py

Output:
  - monitoring/drift_report.html   -- laporan mandiri, buka di browser langsung
  - monitoring/workspace/          -- data untuk Evidently UI (lihat di bawah)

Untuk melihat lewat Evidently UI (setelah workspace terisi):
    sh run_evidently_service.sh
    # buka http://localhost:9002
"""
import os
import joblib
import pandas as pd
from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.metrics import ValueDrift
from evidently.ui.workspace import Workspace

FEATURES = ["age", "income", "bmi", "tenure_months", "num_claims", "credit_score", "policy_value", "risk_score"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "models", "model.pkl")

# Workspace lokal yang dibaca kontainer evidently-ui (lihat docker-compose.yml,
# volume ./monitoring/workspace:/workspace) -- override via env var jika perlu.
WORKSPACE_PATH = os.environ.get("EVIDENTLY_WORKSPACE", os.path.join(os.path.dirname(__file__), ".workspace"))
PROJECT_NAME = "Insurance Approval Monitoring"

# --- Muat model -- pola fallback yang sama dengan api/app.py (Sesi 5) -----
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    import mlflow.sklearn
    #mlflow.set_tracking_uri("sqlite:///mlflow.db") 
    mlflow.set_tracking_uri("http://127.0.0.1:9001")
    model = mlflow.sklearn.load_model("models:/insurance-approval-model@production")

# Reference: representasi data yang dipakai saat training/evaluasi (Sesi 2-4)
reference = pd.read_csv("data/test.csv").copy()

# Current: data produksi -- idealnya log request dari endpoint lokal (Sesi 6),
# bukan file statis seperti contoh ini.
current = pd.read_csv("data/production.csv").copy()

# Hasilkan prediksi pada KEDUA dataset -- inilah yang membedakan prediction
# drift & performance monitoring dari data drift biasa. Target & prediction
# label di-cast ke string -- BinaryClassification native mensyaratkan tipe
# yang konsisten dan eksplisit untuk pos_label, bukan int/bool campur.
for df in (reference, current):
    df["prediction"] = model.predict(df[FEATURES]).astype(str)
    df["prediction_proba"] = model.predict_proba(df[FEATURES])[:, 1]
    df["approved"] = df["approved"].astype(str)

# DataDefinition: representasi data eksplisit ala API native (menggantikan
# ColumnMapping lama) -- mendeklarasikan kolom numerik + definisi klasifikasi
# (target, label prediksi, probabilitas) dalam satu objek yang dipakai ulang.
data_definition = DataDefinition(
    numerical_columns=FEATURES,
    classification=[BinaryClassification(
        target="approved",
        prediction_labels="prediction",
        prediction_probas="prediction_proba",
        pos_label="1",
    )],
)
reference_ds = Dataset.from_pandas(reference, data_definition=data_definition)
current_ds = Dataset.from_pandas(current, data_definition=data_definition)

# --- Laporan utama: sinyal 1 (data drift) + sinyal 2 (prediction drift) ---
# + metrik performa CURRENT (bagian dari sinyal 3). columns=FEATURES membatasi
# DataDriftPreset hanya ke 8 fitur asli -- tanpa ini, kolom prediction yang
# baru ditambahkan ikut terhitung sebagai fitur data drift.
report = Report([
    DataDriftPreset(columns=FEATURES),      # sinyal 1 -- HANYA 8 fitur input
    ValueDrift(column="prediction"),         # sinyal 2 -- output model
    ClassificationPreset(),                  # sinyal 3 -- performa CURRENT
])
snapshot = report.run(current_data=current_ds, reference_data=reference_ds)

os.makedirs(os.path.dirname(os.path.join("monitoring", "drift_report.html")), exist_ok=True)
snapshot.save_html("monitoring/drift_report.html")

# --- Laporan kedua (ringan): performa REFERENCE saja -- untuk perbandingan
# tekstual "accuracy: X -> Y" di terminal. report.run() dengan HANYA
# current_data (tanpa reference_data) menghitung metrik dataset itu sendiri.
ref_report = Report([ClassificationPreset()])
ref_snapshot = ref_report.run(current_data=reference_ds)


def _metric_value(snap, metric_type: str, column: str = None):
    """Ambil value metric dari snapshot. Jika `column` diisi, filter juga
    berdasarkan config['column'] -- DataDriftPreset menghasilkan SATU
    ValueDrift per fitur (age, income, dst.), jadi mencari type saja tidak
    cukup untuk mengambil ValueDrift milik kolom 'prediction' secara spesifik."""
    for m in snap.dict()["metrics"]:
        if m["config"]["type"] != metric_type:
            continue
        if column is not None and m["config"].get("column") != column:
            continue
        return m["value"]
    return None


n_drifted_share = _metric_value(snapshot, "evidently:metric_v2:DriftedColumnsCount")
prediction_drift_pvalue = _metric_value(snapshot, "evidently:metric_v2:ValueDrift", column="prediction")
cur_acc = _metric_value(snapshot, "evidently:metric_v2:Accuracy")
cur_auc = _metric_value(snapshot, "evidently:metric_v2:RocAuc")
ref_acc = _metric_value(ref_snapshot, "evidently:metric_v2:Accuracy")
ref_auc = _metric_value(ref_snapshot, "evidently:metric_v2:RocAuc")

n_drifted, n_total = int(n_drifted_share["count"]), len(FEATURES)
data_drift_share = n_drifted_share["share"]
prediction_drift_detected = prediction_drift_pvalue < 0.05

#print(f"\n1. DATA DRIFT        -- {n_drifted}/{n_total} kolom input drift (share: {data_drift_share:.3f})")
#print(f"2. PREDICTION DRIFT  -- {'terdeteksi' if prediction_drift_detected else 'tidak terdeteksi'} pada output model (p-value: {prediction_drift_pvalue:.4f})")
#print(f"3. MODEL/CONCEPT DRIFT (performance) -- accuracy: {ref_acc:.4f} -> {cur_acc:.4f}  |  ROC-AUC: {ref_auc:.4f} -> {cur_auc:.4f}")

# --- Simpan ke Evidently Workspace agar terlihat di Evidently UI ----------
workspace = Workspace.create(WORKSPACE_PATH)
existing = workspace.search_project(PROJECT_NAME)
project = existing[0] if existing else workspace.create_project(
    PROJECT_NAME, description="Monitoring ML model using evidently."
)
workspace.add_run(project.id, snapshot, include_data=False)
#print(f"\nSnapshot tersimpan ke workspace '{WORKSPACE_PATH}' (project: {PROJECT_NAME})")
#print("Buka via Evidently UI: http://localhost:9002 setelah service di docker dijalan")

# --- Kebijakan retraining: OR dari dua ambang, bukan cuma satu sinyal -----
DRIFT_THRESHOLD = 0.30          # >=30% kolom input drift
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
    print(f"Kebijakan retraining melewati ambang ({'; '.join(alasan)}) -- pertimbangkan retraining.")
else:
    print("Kedua sinyal masih di bawah ambang -- belum perlu retraining.")
