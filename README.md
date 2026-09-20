# MLOps Workshop: Dari Eksperimen ke Produksi

Source code dan resource lengkap untuk Sesi 1-8, mendampingi:
- `Materi MLOps Day-3.pptx`
- `Panduan Pelaksanaan Workshop.md` (langkah menjalankan seluruh workshop)

Environment: **VS Code + Docker (lokal)** untuk development **dan** deployment.

> **Catatan pendekatan workshop:** deployment awalnya dirancang ke Google Cloud Run (lihat riwayat keputusan arsitektur di `Silabus_MLOps.md`). Untuk pelaksanaan workshop ini, mekanisme deployment dipindah sepenuhnya ke Docker Compose lokal karena keterbatasan akses/kuota environment GCP bersama saat kelas berlangsung — konsep CI/CD/CT tetap identik, hanya eksekusinya di laptop sendiri, bukan cloud. GCS (Google Cloud Storage) tetap dipakai untuk DVC remote di Sesi 3, karena itu bagian terpisah dari mekanisme deployment.

Semua kode di repo ini sudah **dijalankan dan diuji nyata** menghasilkan output yang ditampilkan di slide materi (angka akurasi, screenshot, dsb.) — bukan cuplikan yang belum pernah dieksekusi.

---

## Struktur Kode Program

```
mlops-workshop/
├── requirements.txt                  # dependency development lokal (root)
├── config.yml                        # Sesi 2 -- konfigurasi model
├── train.py                          # Sesi 2 -- pipeline training baseline
├── Makefile                          # Sesi 2 -- automation layer
├── data/
│   ├── generate_dataset.py           # cara dataset dibuat (opsional dijalankan ulang)
│   ├── train.csv / test.csv          # data training & evaluasi
│   └── production.csv                # data "produksi" dengan drift disengaja (Sesi 7)
├── tests/test_pipeline.py            # Sesi 2 -- unit test contoh
├── mlflow/
│   ├── train_with_mlflow.py          # Sesi 4 -- tracking + registry, bootstrap alias "production"
│   ├── export_model.py               # jembatan registry (alias production) -> file statis untuk Docker
│   └── retrain_and_compare.py        # Sesi 8 -- CT: retrain + bandingkan + promosikan jika lebih baik
├── api/
│   ├── app.py                        # Sesi 5 -- FastAPI serving
│   └── requirements.txt              # dependency ramping khusus image Docker
├── docker/
│   ├── Dockerfile                    # Sesi 6
│   ├── docker-compose.yml            # Sesi 6 -- deployment lokal (pengganti Cloud Run)
│   ├── deploy.sh                     # Sesi 6 -- CI/CD lokal: export + build + redeploy + smoke test
├── monitoring/
│   ├── monitor.py                    # Sesi 7 -- data drift + prediction drift + performance
│   └── requirements.txt
├── scripts/test_deployment.py        # Sesi 6 -- uji endpoint lokal yang sudah live
└── orchestration/
    ├── pipeline.py                   # Sesi 8 -- CI/CD/CT dengan Prefect
    └── requirements.txt
```

---

## Sesi 0 — Setup Environment 

### 0.1 Checklist Environment Python + Docker

```bash
# 0. Pastikan Docker Desktop dan VSCode sudah terinstall di komputer
open -a Docker

# 1. Buka folder ini di VS Code
code mlops-workshop

# 2. Buat & aktifkan virtual environment -- WAJIB pakai python3.11 eksplisit,
#    bukan python3 generik (lihat peringatan batas Python di bawah)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install semua dependency sekaligus dari satu file -- versi numpy/evidently
#    sudah dipin saling kompatibel, tidak perlu instalasi bertahap:
python3 -m pip install -r requirements.txt

# 4. Pilih interpreter di VS Code: Command Palette -> "Python: Select Interpreter" -> ./venv

# 5. Verifikasi
python --version        # Verifikasi versi python > 3.10
python -c "import numpy; print(numpy.__version__)"   # harus >= 1.26 (tanpa batas atas)
docker run hello-world  # daemon Docker hidup

```

---

## Sesi 1 — Motivasi & Lanskap MLOps

Pengantar untuk workshop MLOPs (CI/CD/CT).

---

## Sesi 2 — Reproducibility & Struktur Proyek

```bash
make setup    # membuat venv + install dependency (redundan jika sudah Sesi 0)
make run      # menjalankan train.py, hasil: Accuracy 0.9380, ROC-AUC 0.9817
make test     # menjalankan tests/test_pipeline.py

# Jika Makefile tidak didukung, lakukan secara manual untuk pengujian
python3 train.py    # menjalankan trainning
pytest tests/ -v    # menjalankan testing model

```

Ganti model tanpa menyentuh kode — edit `config.yml`:
```yaml
model:
  name: GradientBoosting   # dari: RandomForest
  params:
    n_estimators: 150
    learning_rate: 0.08
    max_depth: 3
```

---

## Sesi 3 — Data Version Control (DVC) 
### Catatan: materi ini akan diskip pada pertemua DAY-3 karena sudah dibahas di DAY-2. Pada DAY-3 kita hanya akan menggunakan data yang sudah tersedia pada sumber yang diberikan.

```bash
# Inisialisasi Git (jika repo ini belum jadi repo Git)
git init

# Inisialisasi DVC & lacak data
dvc init
dvc add data/train.csv

# Setup remote GCS -- ganti nama bucket dengan milik Anda
gcloud auth application-default login
dvc remote add -d gcsremote gs://NAMA-BUCKET-ANDA/dvcstore

git add data/train.csv.dvc data/.gitignore .dvc
git commit -m "track train.csv with dvc"
dvc push
```

Simulasi kolaborasi tim: dari mesin/folder lain,
```bash
git clone <url-repo-anda>
cd mlops-workshop
dvc pull   # menarik data yang identik (verifikasi hash MD5 di data/train.csv.dvc)
```

---

## Sesi 4 — Experiment Tracking & Model Registry (MLflow)

```bash
python3 mlflow/train_with_mlflow.py
```

Output yang diharapkan (persis seperti di slide):
```
DecisionTree: acc=0.8740 auc=0.9075 f1=0.8706
RandomForest: acc=0.9380 auc=0.9817 f1=0.9350
GradientBoosting: acc=0.9380 auc=0.9804 f1=0.9347

BEST: RandomForest <run_id>
Registered version: 1
```

Buka dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 9001
# buka http://127.0.0.1:5001
```

> ⚠️ **warning**: `mlflow ui` bisa gagal berjalan jika port 9001 sudah dipakai oleh aplikasi lain. Pastikan port 9001 belum digunakan, jika sudah pilih port lain rekomendasi > 9000.

> ⚠️ **Kenapa `sqlite:///mlflow.db`, bukan `file:///...`?** MLflow versi terbaru menandai filesystem tracking backend sebagai *maintenance mode* dan bisa menolaknya langsung dengan error saat `set_experiment()` dipanggil. Backend SQLite tetap sepenuhnya lokal tapi didukung penuh.

---

## Sesi 5 — Serving Model dengan FastAPI

```bash
# Dari root proyek (bukan dari dalam folder api/)
uvicorn api.app:app --reload
# buka http://127.0.0.1:8000/docs
```

Uji endpoint:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":34,"income":78000,"bmi":24.5,"tenure_months":18,"num_claims":1,"credit_score":720,"policy_value":150000,"risk_score":42.5}'

# Response: {"approved":false,"probability":0.4112,"model_version":"RandomForest-v1"}
```

**Debugging** — tekan `F5` di VS Code (memakai `.vscode/launch.json` yang sudah disiapkan), pasang breakpoint di baris `proba = model.predict_proba(df)[0][1]` dalam `api/app.py`, lalu kirim request `/predict` dari terminal lain.

`api/app.py` otomatis memuat model dari **MLflow Registry** di tahap ini (karena `api/models/model.pkl` belum ada) — inilah jalur development. Lanjut ke Sesi 6 untuk mengekspornya menjadi file statis.

---

## Sesi 6 — Kontainerisasi & Deployment (Docker, Lokal)

> **Catatan pendekatan workshop:** pada kondisi nyata umumnya deployment model dilakukan ke sistem produksi yang bisa berupa layanan di gloud public seperti Google/AWS/ ataupun on-premise server produksi. Namun, mempertimbangkan keterbatasan resource dan juga waktu pelaksanaan workshop, mekanisme deployment dilakukan ke **Docker Compose lokal**. Urutan operasinya — build image → jalankan → health check → ganti versi semuanya di Docker di laptop sendiri. Endpoint model serving dapat diakses melalui `localhost`.

### Langkah 0 — Ekspor model untuk kontainerisasi
```bash
python3 mlflow/export_model.py
# menarik model via alias "production" dari MLflow Registry -> api/models/model.pkl
```

### Bagian A — Belajar Docker (siklus manual)
```bash
docker build -f docker/Dockerfile -t insurance-api .
docker run -d -p 8080:8080 --name api insurance-api
docker ps
docker logs api
docker exec -it api /bin/bash   # verifikasi: ls models/ -> model.pkl ada
docker stop api && docker rm api
```

Demonstrasi layer caching: ubah satu baris di `api/app.py`, jalankan `docker build` lagi, amati baris `CACHED` pada layer `pip install`.

### Bagian B — Deploy Lokal dengan Docker Compose (CI/CD)
```bash
chmod +x docker/local_deploy.sh
./docker/deploy.sh
```

Script ini menjalankan alur CI/CD lokal secara berurutan: **[CI]** `mlflow/export_model.py` (tarik model production terbaru) → `docker compose build` (build image) → **[CD]** `docker compose up -d --force-recreate` (redeploy container, mengganti yang lama jika masih berjalan) → smoke test otomatis ke `http://localhost:8080/`.

Uji endpoint yang sudah live:
```bash
python3 scripts/test_deployment.py
# default menguji http://localhost:8080 -- tidak perlu argumen
```

**Membersihkan / menghentikan:**
```bash
docker compose -f docker/docker-compose.yml down
```

**Redeploy setelah model production berganti** (mis. setelah Sesi 8 mempromosikan model baru) — cukup jalankan ulang `./docker/local_deploy.sh`; tidak perlu urutan build→tag→push manual.

---

## Sesi 7 — Monitoring Model di Produksi (Evidently AI)

Tiga sinyal, bukan satu — banyak tutorial berhenti di data drift saja. Materi ini menambahkan **prediction drift** dan **model/concept drift (performance)**:

```bash
python3 monitoring/monitor.py
```

Output nyata (dari data yang sengaja diberi drift):
```
Laporan tersimpan di monitoring/drift_report.html

1. DATA DRIFT        -- 3/8 kolom input drift (share: 0.375)
2. PREDICTION DRIFT  -- tidak terdeteksi pada output model
3. MODEL/CONCEPT DRIFT (performance) -- accuracy: 0.9380 -> 0.8880  |  ROC-AUC: 0.9817 -> 0.9575

⚠ Kebijakan retraining melewati ambang (data drift 0.38 >= ambang 0.3; accuracy turun 0.0500 >= ambang 0.03) -- pertimbangkan retraining.
```

Buka `monitoring/drift_report.html` di browser — laporan ini sekarang berisi tiga bagian: tabel data drift (histogram reference vs current per kolom fitur), skor drift kolom `prediction`, dan panel *Classification Performance* lengkap (accuracy/precision/recall/ROC-AUC/confusion matrix, reference vs current berdampingan).

**Temuan yang jadi bahan diskusi wajib:** pada data yang dipakai materi ini, *prediction drift* **tidak terdeteksi** secara statistik — padahal akurasi model yang sesungguhnya turun 5 poin persentase (0.938 → 0.888) begitu diukur dengan ground truth. Artinya kalau tim hanya memantau prediction drift (karena lebih murah — tidak perlu menunggu label), penurunan performa ini **akan terlewat**. Inilah alasan kebijakan retraining di bawah memakai OR dari dua ambang, bukan satu sinyal saja.

**Kenapa tiga sinyal ini berbeda sifatnya:**

| Sinyal | Butuh ground truth? | Kecepatan | Yang diukur |
|---|---|---|---|
| Data drift | Tidak | Real-time | Distribusi fitur **input** berubah |
| Prediction drift | Tidak (butuh model) | Real-time | Distribusi **output** model berubah |
| Model/concept drift (performance) | **Ya** | Tertunda (label latency) | Akurasi **sesungguhnya** dibanding reference |

Di dunia nyata, label (`approved` — klaim disetujui atau tidak) sering baru diketahui berbulan-bulan setelah prediksi dibuat. Data/prediction drift adalah **early warning** yang tersedia instan; performance monitoring adalah **kebenaran akhir** yang datang belakangan.


**Untuk data produksi yang lebih nyata**: tambahkan logging sederhana di `api/app.py` yang menulis setiap payload `/predict` (plus hasil aktualnya begitu tersedia) ke file/tabel, lalu pakai itu sebagai `current` dataset. Dimana pada workshop kali ini data tersebut dianggap sudah tersedia di `data/production.csv`. 

---

## Sesi 8 — Orkestrasi CI/CD/CT dengan Prefect (Lokal)

Sesi 4-7 dijalankan sebagai perintah terpisah yang diketik manusia satu per satu. `orchestration/pipeline.py` membungkus skrip-skrip yang **sama persis** (tidak ada logika ML yang ditulis ulang) menjadi satu graf CI/CD/CT yang mengambil keputusan sendiri — dan, penting, **tidak asal deploy**:

```
check_drift()  →  jika drift melewati ambang...
retrain_and_compare()  →  latih kandidat baru, bandingkan dgn production saat ini
                           ├─ TIDAK lebih baik → STOP (model lama tetap dipakai)
                           └─ lebih baik → lanjut...
build_and_deploy_local()  →  export model → docker compose build → up --force-recreate
```

**Kenapa perlu perbandingan?** "Retrain lalu deploy otomatis" tanpa membandingkan dulu adalah praktik yang tidak baik, model produksi bisa memburuk tanpa disadari kalau data baru kebetulan menghasilkan model yang lebih jelek (data quality issue, bukan perbaikan). Perbandingan ini yang menjawab pertanyaan "bagaimana update model dilakukan ketika data baru menghasilkan model yang lebih baik" — jawabannya: **hanya kalau benar-benar lebih baik**, diverifikasi otomatis, bukan diasumsikan.

```bash
pip install -r orchestration/requirements.txt   # In case prefect belum terinstall
python -m prefect server start                  # Jalankan server prefect terlebih dahulu, server prefect dapat diakses pada http://127.0.0.1:4200`
python3 orchestration/pipeline.py               # Jalankan pipeline orkestrasi, diuji tanpa dan dengan schedule/intervals
```

Output nyata (kasus model baru **tidak** lebih baik — data retraining identik dengan sebelumnya):
```
Flow run 'omniscient-ermine' - Beginning flow run for flow 'insurance-approval-ci-cd-ct'
Task run 'check_drift-b85' - Drift terdeteksi pada 3/8 kolom (share: 0.375)
Task run 'check_drift-b85' - Finished in state Completed()
Flow run 'omniscient-ermine' - >> Drift melewati ambang -- memicu retraining (CT)...
Task run 'retrain_and_compare-081' - Kandidat terbaik: RandomForest-retrained (accuracy=0.9380)
Task run 'retrain_and_compare-081' - Model production saat ini: versi 1 (accuracy=0.9380)
Task run 'retrain_and_compare-081' - NOT_PROMOTED: kandidat (0.9380) tidak lebih baik dari production (0.9380)
Flow run 'omniscient-ermine' - >> Model baru TIDAK lebih baik dari production -- deploy DIBATALKAN.
Flow run 'omniscient-ermine' - Finished in state Completed()
```

Kalau kandidat **memang** lebih baik, baris terakhir berubah jadi `PROMOTED: versi N (...) > production lama (...)`, dan flow lanjut memanggil `build_and_deploy_local()` — image baru ter-build dan container ter-redeploy otomatis, tanpa campur tangan manual.

**Yang berubah dibanding menjalankan skrip manual:**
- `check_drift()` **memicu** `retrain_and_compare()` secara kondisional — bukan cuma dicetak untuk dibaca manusia
- `retrain_and_compare()` **memblokir** deploy kalau model tidak membaik — gerbang kualitas yang tidak ada di versi manual
- `@task(retries=3)` pada `build_and_deploy_local()` menjawab kegagalan build/startup sesaat pada Docker lokal
- Riwayat setiap run (kapan, task mana gagal/berhasil, PROMOTED atau tidak) tersimpan dan bisa dilihat lewat `prefect server start` + buka `http://127.0.0.1:4200`

> **Catatan pendekatan workshop:** `build_and_deploy_local()` memanggil `docker/local_deploy.sh` yang men-deploy ke Docker lokal. Struktur flow-nya — urutan, kondisional, retry, identik dengan yang umumnya dipakai untuk deploye ke server produksi nyata.

> Kenapa Prefect, bukan Airflow/Kubeflow/ZenML? Airflow butuh Postgres + scheduler process, Kubeflow butuh cluster Kubernetes — keduanya bertentangan dengan prinsip "zero local setup berat" yang dipegang sepanjang kuliah ini. Prefect adalah `pip install` biasa, `@task`/`@flow` di atas fungsi Python yang sudah ada.

---

## Ringkasan Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `exec format error` saat build/run | Jarang terjadi untuk deployment lokal (build & run di mesin yang sama) — hanya relevan kalau image dipindah ke mesin arsitektur lain | Pastikan `docker build` dan `docker run`/`docker compose up` di mesin yang sama |
| `mlflow ui` gagal/aneh di macOS | Port 5000 dipakai AirPlay Receiver | Gunakan `--port 5001` (sudah default di semua contoh) |
| `MlflowException: ...maintenance mode` | Backend file:// dipakai | Gunakan `sqlite:///mlflow.db` |
| `AttributeError: np.float_ was removed` | Memakai `evidently==0.4.23` (patch sangat lama) | Materi sudah pakai `evidently==0.7.21` via `evidently.legacy` — cek `pip show evidently` |
| `AttributeError: module 'plotly.figure_factory' has no attribute 'create_distplot'` | Memakai `evidently==0.4.40` tanpa pin plotly (rilis lama tanpa batas atas plotly) | Materi sudah pakai `evidently==0.7.21`, yang mendeklarasikan `plotly<6` sendiri — bug ini tidak lagi bisa terjadi |
| `ModuleNotFoundError: No module named 'evidently.report'` | Import lama (`from evidently.report import Report`) dipakai langsung, bukan lewat `evidently.legacy` | `evidently==0.7.21` merombak total struktur modul top-level; gunakan `from evidently.legacy.report import Report` (lihat `monitor.py`) |
| `ResolutionImpossible` saat `pip install -r requirements.txt` | venv dibuat dengan Python 3.15+ (di luar rentang `prefect<3.15`) | Buat ulang venv dengan `python3.11`–`python3.14` |
| `make: command not found` (Windows) | `make` tidak ada secara default | Git Bash, WSL2, atau Dev Container |
| `docker build` lambat / context besar | `data/`, `venv/`, `mlruns/` ikut terkirim | Sudah ditangani `.dockerignore` — jangan hapus file ini |
| Container gagal listen di port 8080 | Lupa set `$PORT` | Dockerfile di repo ini sudah benar (`ENV PORT=8080`) |
| `docker compose up` gagal — port 8080 dipakai | Container lama dari sesi sebelumnya masih jalan | `docker compose -f docker/docker-compose.yml down` dulu, atau `docker ps` + `docker stop <id>` |
| `mlflow.exceptions.MlflowException` saat `retrain_and_compare.py` | Belum ada model dengan alias `production` | Jalankan `mlflow/train_with_mlflow.py` dulu (Sesi 4) untuk bootstrap alias |
| `curl: (7) Failed to connect` saat smoke test `deploy.sh` | Container belum siap / gagal start | `docker compose -f docker/docker-compose.yml logs` untuk lihat error startup |

---

## Urutan Menjalankan dari Nol (Ringkasan Cepat)

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

make run                              # Sesi 2
dvc init && dvc add data/train.csv    # Sesi 3 (sesuaikan remote GCS Anda)
python3 mlflow/train_with_mlflow.py   # Sesi 4 -- bootstrap alias "production"
uvicorn api.app:app --reload          # Sesi 5 (Ctrl+C untuk lanjut)
python3 mlflow/export_model.py        # jembatan ke Sesi 6
docker build -f docker/Dockerfile -t insurance-api .   # Sesi 6 Bagian A
./docker/deploy.sh                    # Sesi 6 Bagian B -- CI/CD lokal, tanpa GCP
python3 monitoring/monitor.py         # Sesi 7
python3 orchestration/pipeline.py     # Sesi 8 -- CI/CD/CT: retrain, bandingkan, deploy hanya jika lebih baik
```
