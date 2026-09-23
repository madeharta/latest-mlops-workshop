# Panduan Pelaksanaan Workshop MLOps
### Panduan Fasilitator/Instruktur

Dokumen ini menjabarkan urutan aktivitas untuk menjalankan seluruh kegiatan workshop dari persiapan hingga teknik orkestrasi MLOps. Dokumen lain yang menjadi rujukan:

| Berkas | Fungsi dalam panduan ini |
|---|---|
| `Materi MLOps Day-3.pptx`  | Deck **utama** — sumber materi mengajar tiap sesi |
| `mlops-workshop-source-code.zip` | Kode yang benar-benar dijalankan live di kelas |

### Ringkasan Alokasi Waktu Total

| Sesi | Topik | Estimasi (menit) |
|---|---|---|
| 1 | Reproducibility & Struktur | 15 |
| 2 | MLflow | 105 |
| 3 | FastAPI | 60 |
| 4 | Docker + Deploy Lokal | 30  |
| 5 | Evidently AI (3 sinyal: data/prediction/performance) | 90 |
| 6 | Prefect + Incident Handling + Penutup | 60 |
---

**Notasi yang dipakai:**
- `[Detail #N]` = nomor slide di `Materi MLOps Day-3.pptx`
- Blok kode = perintah persis yang diketik/dijalankan di terminal

---

## 1. Sesi 1 — Pengantar Workshop MLOPs DAY-3

**Tidak ada kode.** Sesi ini murni konseptual.

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Slide #4]` — Pengantar peta pembelajaran workshop |
| B (5’) | `[Slide #5]`: Pengenalan tools MLOps dari OSS hingga Enterprise grade |
| E (5’) | `[Slide #6]` Penjelasan tools yang akan dipakai pada workshop |

## 2. Persiapan Awal

### 2.1 Tugas Instruktur

Mendistribusikan kode resource yang diperlukan dan memantau/mendampingi proses konfigurasi environment untuk persiapan workshop. Memastikan semua mahasiswa sudah melakukan proses konfigurasi dengan benar, sesuai requirements.

> **Catatan pendekatan workshop:** deployment (Sesi 6) dan orkestrasi (Sesi 8) berjalan sepenuhnya lokal via Docker 

### 2.2 Tugas Mahasiswa (Sesi 0, pada awal sesi - 15 Menit)

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Slide #7]` Penjelasan terkait data yang akan dipakai pada workshop |
| B (5’) | `[Slide #8]` Penjelasan struktur kode program selama workshop (anatomi struktur proyek) |
| E (15’) | `[Slide #9]` Persiapan dan pengecekan lingkungan pengembangan pada komputer masing-masing peserta |

Checklist ini ada di `README.md` bagian "Sesi 0", mahasiswa menjalankannya mandiri dan melapor jika ada kegagalan pada awal sesi terkait konfigurasi environment. Hal-hal yang perlu dicek:

#### Mahasiswa memastikan sudah menginstall Docker Desktop dan VSCode serta melakukan pre-setup

```bash
code mlops-workshop                                   # Membuka kode sumber dengan VSCode
open -a Docker                                        # Menjalankan Docker  Desktop via terminal in macOS 
docker desktop start                                  # Menjalankan Docker Desktop via terminal in Windows

python3 --version                                     # harus >3.10, rekomendasi 3.11-3.14
python3 -m venv .venv                                 # Membuat virtual environment. 
source .venv/bin/activate                             # Aktivasi virtual environment di mac OS 
.venv/scripts/activate                                # Aktivasi virtual environemtn di Windows 

python3 -m pip install -r requirements.txt            # Install semua tools dan library yang diperlukan untuk menjalankan seluruh aktivitas
python3 -c "import numpy; print(numpy.__version__)"   # harus >= 1.26 (tanpa batas atas)
docker run hello-world                                # daemon Docker hidup, jalankan perintah dari terminal VScode
```

**Warning** Pastikan semua proses berjalan dengan baik tanpa error. Jika terjadi error silahkan laporkan ke Dosen atau TA, sebelum melanjutkan ke tahapan berikutnya. Kegagalan pada proses konfigurasi awal akan berdampak pada gagalnya tahapan workshop selanjutnya.

---

## 3. Sesi 1 — Reproducibility & Struktur Proyek

**Skrip demo (jalankan persis, proyeksikan terminal):**
```bash
cd mlops-workshop
cat config.yml              # tunjukkan model+params
cat train.py                # tunjukkan pembacaan config, tanpa hardcode
```
Output yang harus muncul (rujuk `[Detail #13]`):
```
Model      : RandomForest
Accuracy   : 0.9380
ROC-AUC    : 0.9817
Model saved to models/model.pkl
```
Lalu demonstrasikan **mengganti model tanpa menyentuh kode** — ubah `config.yml`:
```yaml
model:
  name: GradientBoosting
  params: {n_estimators: 150, learning_rate: 0.08, max_depth: 3}
```
```bash
make run     # jalankan train.py
make test    # 2 test pytest lulus

# Jika Makefile tidak didukung, lakukan secara manual untuk pengujian
python3 train.py    # menjalankan trainning
pytest tests/ -v    # menjalankan testing model
```

---

## 4. Sesi 2 — Experiment Tracking & Model Registry (MLflow)

Sesi ini yang paling padat tutorialnya — pertimbangkan mengurangi Fase B jika waktu ketat.

| Fase | Slide/Aktivitas |
|---|---|
| A (10’) | `[Detail #11–16]` Pengantar tentang MLflow dan diskusi singkat |
| B (10’) - demo langsung | `[Detail #17-18]` Menjalankan eksperimen pada MLFlow dan melihat hasilnya pada MLflow UI|
| **Gotcha wajib disampaikan** | `[Detail #18]` — backend `file:///` deprecated, HARUS pakai `sqlite:///` |
| C (35’) - demo langsung | `[Detail #19–26]` | Menguji fitur-fitur MLflow seperti autologgin dan advanced tracking |
| D (20’) - demo langsung | `[Detail #27–29]` | Melakukan model registrasi dan explorasi fitur model registry MLFlow |
| D (15’) - latihan langsung | `[Detail #27–29]` | Melakukan model registrasi dan explorasi fitur model registry MLFlow |

**Skrip demo:**
```bash
python3 mlflow/train_with_mlflow.py
```
Output (rujuk `[Detail #31]`):
```
DecisionTree: acc=0.8740 auc=0.9075 f1=0.8706
RandomForest: acc=0.9380 auc=0.9817 f1=0.9350
GradientBoosting: acc=0.9380 auc=0.9804 f1=0.9347

BEST: RandomForest <run_id akan berbeda>
Registered version: 1
```
Buka dashboard di proyektor:
```bash
# Buka jedela terminal baru di VScode kemudian jalankan perintah berikut (pastikan .venv sudah aktif)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 9001
```
Buka `http://127.0.0.1:9001` di browser, tunjukkan tabel perbandingan (bandingkan dengan `[Detail #33]` screenshot), klik ke Model Registry, tunjukkan versi 1 ter-registrasi.

> ⚠️ **warning**: `mlflow ui` bisa gagal berjalan jika port 9001 sudah dipakai oleh aplikasi lain. Pastikan port 9001 belum digunakan, jika sudah pilih port lain rekomendasi > 9000.

Lakukan explorasi fitur-fitur MLFlow dengan menjalankan kode-kode berikut secarfa bergantian. Kemudian pantau MLFlow UI untuk melihat perbedaan yang diberikan oleh setiap kode program.

```bash
# Untuk menguji fitur autologgin dari MLFlow
python3 mlflow/ztrain_with_mlflow_autologgin.py

# Untuk menguji fitur autologgin dengan model regression
python3 mlflow/ztrain_with_mlflow_regression.py

# Untuk menguji fitur advanced tracking pada MLflow
python3 mlflow/zmlflow_explore_tracking.py
```

Untuk setiap eksekusi program diatas, buka MLFlfow UI dan perhatikan informasi apa saja yang diberikan. Kemudian coba hubungkan baris kode program mana yang menghasilkan output tersebut.

---

## 5. Sesi 3 — Serving Model dengan FastAPI

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #32]` Pengenalan model serving dengan FastAPI|
| B (15’) | `[Detail #33-34]` Penjelasan kode program dan API dari FastAPI untuk model serving |
| C (40’) — demo langsung | `[Detail #35]` Demo deployment model dengan FastAPI diikuti handson oleh mahasiswa |

**Skrip demo:**
```bash
# Buka jedela terminal baru di VScode kemudian jalankan perintah berikut (pastikan .venv sudah aktif)
cat api/app.py    # tunjukkan Pydantic ApplicantData + endpoint /predict
uvicorn api.app:app --reload
```
Buka `http://127.0.0.1:8000/docs` di browser (bandingkan dengan `[Detail #45]`), klik **Try it out** pada `/predict`, kirim payload contoh, tunjukkan response:
```json
{"approved": false, "probability": 0.4112, "model_version": "RandomForest-v1"}
```

**Demo debugging (`[Detail #47–48]`)** — tekan `F5` di VS Code, pasang breakpoint di baris `proba = model.predict_proba(df)[0][1]` pada `api/app.py`, kirim request dari terminal lain:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
-d '{"age":34,"income":78000,"bmi":24.5,"tenure_months":18,"num_claims":1,"credit_score":720,"policy_value":150000,"risk_score":42.5}'

# Sometime curl syntax can be messy, try the following one in Windows from VScode terminal:
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{\"age\":34,\"income\":78000,\"bmi\":24.5,\"tenure_months\":18,\"num_claims\":1,\"credit_score\":720,\"policy_value\":150000,\"risk_score\":42.5}'

# If it does not work, just try the API from the browser by editing the request body and observe the result.

```
Tunjukkan eksekusi berhenti di breakpoint, inspeksi variabel `df`.

**Demo validasi gagal (`[Detail #49]`)** — kirim payload dengan field hilang, tunjukkan HTTP 422 otomatis tanpa kode tambahan.

---

## 6. Sesi 4 — Kontainerisasi & Deployment (Docker, Lokal)

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #39]` Penjelasan model deployment pada environment lokal dengan docker|
| B (20’) - demo langsung | `[Detail #40]` Melakukan deployment model lokal dengan docker diikuti handson peserta |
| D (20’) — lab | `[Detail #40]` Peserta melakukan mengujian terhadap API yang sudah dideploy baik melalui browser atau terminal |

### Bagian A — Belajar Docker (demo langsung, 30’)

```bash
python3 mlflow/export_model.py     # jembatan: registry -> api/models/model.pkl

docker build -f docker/Dockerfile -t insurance-api .
docker run -d -p 8080:8080 --name api insurance-api
docker ps
docker logs api
docker exec -it api /bin/bash      # ls models/ -> tunjukkan model.pkl ada
exit
docker stop api && docker rm api
```
**Demonstrasi layer caching (`[Detail #56]`)** — ubah satu baris komentar di `api/app.py`, jalankan `docker build` lagi, tunjukkan baris `CACHED` pada layer `pip install`.

### Bagian B — Deploy Lokal dengan Docker Compose (demo langsung, 25’)

> **Catatan pendekatan workshop:** pada kondisi nyata umumnya deployment model dilakukan ke sistem produksi yang bisa berupa layanan di gloud public seperti Google/AWS/ ataupun on-premise server produksi. Namun, mempertimbangkan keterbatasan resource dan juga waktu pelaksanaan workshop, mekanisme deployment dilakukan ke **Docker Compose lokal**. Urutan operasinya — build image → jalankan → health check → ganti versi semuanya di Docker di laptop sendiri. Endpoint model serving dapat diakses melalui `localhost`.

```bash
# [Detail #59] docker-compose.yml, [Detail #60] menjalankan deploy.sh
# Untuk Windows jalankan perintah berikut dari Git Bash, power shell tidak mendukung .sh 
# Jika menggunakan power shell jalankan perintah docker compose build/up secara manual
chmod +x docker/deploy.sh
./docker/deploy.sh
```
Script ini menjalankan CI (export model + build image) dan CD (redeploy container + smoke test) berurutan — jelaskan tiap tahap saat log muncul di terminal.

```bash
# [Detail #61] kode pengujian
python3 scripts/test_deployment.py
# default menguji http://localhost:8080 -- tidak perlu argumen URL
```

`[Detail #62]` (restart policy & health check) — jelaskan `restart: unless-stopped` dan `healthcheck` sebagai pengganti auto-healing pada layanan server produksi yang umum dipakai.

---

## 7. Sesi 5 — Monitoring Model di Produksi (Evidently AI)

| Fase | Slide/Aktivitas |
|---|---|
| A (15’) | `[Ringkas #42–46]` Penjelasan terkait model monitoring dan tools evidently AI|
| B (35’) - demo langsung| `[Detail #47–48]` Melakukan model monitoring dengan evidently AI diikuti handson peserta |
| C (25’) - lab | `[Detail #49-53]` Membaca eveidently report secara lokal maupun melalui server UI|
| D (20’) | `[Detail #54-56]` Penetapan kriteria re-trainning model dan dampaknya |

**Skrip demo:**
Jalankan code monitoring dengan perintah berikut:

```bash
python3 monitoring/monitor.py
```
Jalankan server evidently-ui dengan perintah berikut:

```bash
# Untuk mac OS gunakan perintah berikut untuk menjalankan 
sh monitoring/run_evidently_service.sh 

# Jika shell tidak didukung pada environment anda atau workspace tidak ditampilkan, 
# Matikan evidently-service dan jalankan dengan perintah berikut 
docker run -p 9002:8000 -v "${PWD}/monitoring/.workspace:/app/workspace" --name evidently-service --detach evidently/evidently-service:latest
```
Server ini akan menampilkan report monitoring yang disimpan oleh evidently pada workspace yang dibuat yang dapat diakses melalui http://localhost:9002. 

Output nyata:
```
1. DATA DRIFT        -- 3/8 kolom input drift (share: 0.375)
2. PREDICTION DRIFT  -- tidak terdeteksi pada output model
3. MODEL/CONCEPT DRIFT (performance) -- accuracy: 0.9380 -> 0.8880  |  ROC-AUC: 0.9817 -> 0.9575

⚠ Kebijakan retraining melewati ambang (data drift 0.38 >= ambang 0.3; accuracy turun 0.0500 >= ambang 0.03) -- pertimbangkan retraining.
```
Buka `monitoring/drift_report.html` di browser — scroll ke tiga bagian: tabel data drift (histogram per fitur, `[Detail #73]` screenshot), skor drift kolom `prediction`, dan panel *Classification Performance* (`[Detail #74]` screenshot — accuracy/ROC-AUC reference vs current).

**Poin pedagogis kunci — tekankan ini secara eksplisit (`[Detail #75]`)**: prediction drift **tidak terdeteksi** secara statistik pada output di atas, padahal accuracy sungguhan turun 5 poin persentase. Ini bukti langsung kenapa memantau satu sinyal saja (prediction drift lebih murah karena tak perlu menunggu label) bisa **melewatkan** penurunan performa nyata. Jadikan ini diskusi: "kalau tim kalian hanya pasang alert di prediction drift, insiden ini tidak akan pernah terdeteksi."

`[Detail #56]` (kebijakan operasional) — jelaskan bahwa trigger retraining memakai **OR dari dua ambang** (data drift ≥0.30 ATAU accuracy turun ≥0.03), bukan cuma satu sinyal — langsung tersambung ke temuan di atas.

---

## 8. Sesi 6 — MLOps Lanjut, Tata Kelola & Proyek Akhir

| Fase | Slide/Aktivitas |
|---|---|
| A (10’) | `[Ringkas #58–61]` Pengantar terkait MLOps orkestrasi dengan prefect|
| **Keputusan tool** | `[Detail #61]` — kenapa Prefect dipilih untuk workshop ini |
| B (30’) - demo langsung | `[Detail #62]` Penjelasan kode program workflow otomisasi diikuti handson peserta |

**Skrip demo:**
```bash
pip install -r orchestration/requirements.txt   # In case prefect belum terinstall
python3 -m prefect server start                  # Jalankan server prefect terlebih dahulu, server prefect dapat diakses pada http://127.0.0.1:4200`
python3 orchestration/pipeline.py               # Jalankan pipeline orkestrasi, diuji tanpa dan dengan schedule/intervals
```
Riwayat setiap run (kapan, task mana gagal/berhasil, PROMOTED atau tidak) tersimpan dan bisa dilihat lewat prefect server start + buka http://127.0.0.1:4200

Tunjukkan log Prefect live di terminal — perhatikan tiga hal: (1) nama task (`check_drift-xxx`, `retrain_and_compare-xxx`), (2) baris `PROMOTED`/`NOT_PROMOTED` yang menentukan apakah deploy dijalankan, dan (3) bahwa `build_and_deploy_local()` **hanya terpanggil kalau** kandidat baru benar-benar lebih baik dari `production` saat ini. Karena deployment sepenuhnya lokal (Docker, bukan GCP), demo ini bisa dibiarkan berjalan sampai selesai tanpa bergantung kredensial cloud apa pun.

> **Poin pedagogis kunci**: kemungkinan besar hasil pertama adalah `NOT_PROMOTED` (data retraining identik dengan data yang sudah dipakai) — ini **bukan kegagalan demo**, justru bukti gerbang kualitas bekerja. Siapkan skenario `PROMOTED` sebagai cadangan: jalankan `mlflow/retrain_and_compare.py` secara terpisah setelah mengubah `data/train.csv` (mis. gabungkan dengan data tambahan), atau gunakan skrip verifikasi mekanisme di `README.md` bagian Sesi 8 yang mendemonstrasikan alias berpindah secara terisolasi.

---
## 9. Sesi 7 — Incident Handling

| Fase | Slide/Aktivitas |
|---|---|
| A (10’) | `[Ringkas #64–67]` Penjelasan dan diskusi terkait incident handling|
| B (5’)  | `[Detail #68]` Rangkuman dan penutup workshop DAY-3 |

## 9. Lampiran — Troubleshooting Cepat Selama Mengajar

Tabel ini adalah gabungan seluruh gotcha yang muncul di slide `[Detail]` dan `README.md` — cetak/simpan terpisah sebagai lembar contekan saat mengajar:

| Gejala saat live demo | Sesi | Solusi cepat |
|---|---|---|
| `docker compose up` gagal — port 8080 dipakai | 6 | Container lama masih jalan: `docker compose -f docker/docker-compose.yml down` dulu |
| `mlflow ui` blank/error di macOS | 4 | Ganti ke `--port 5001`, bukan 5000 |
| `MlflowException ... maintenance mode` | 4 | Tracking URI harus `sqlite:///mlflow.db` |
| `AttributeError: np.float_ was removed` | 7 | Cek `pip show evidently` — repo sudah pakai `0.7.21` via `evidently.legacy`; jika masih versi 0.4.x lama di mesin mahasiswa, install ulang dari `requirements.txt` |
| `AttributeError: 'plotly.figure_factory' has no attribute 'create_distplot'` | 7 | Materi sudah pakai `evidently==0.7.21`, yang deklarasikan `plotly<6` sendiri — bug ini seharusnya tidak muncul lagi; kalau muncul, cek versi evidently ter-install |
| `ModuleNotFoundError: No module named 'evidently.report'` | 7 | Import langsung `evidently.xxx` dipakai, bukan `evidently.legacy.xxx` — cek `monitoring/monitor.py` memakai jalur import yang benar |
| `ResolutionImpossible` saat `pip install` | 0,7 | venv dibuat dengan Python 3.15+ (di luar rentang `prefect<3.15`). Buat ulang venv dengan Python 3.11–3.14 |
| `make: command not found` | 2 | Jalankan `python3 train.py` langsung sebagai fallback |
| Container gagal listen di port 8080 | 6 | Cek `ENV PORT=8080` ada di Dockerfile (sudah benar di repo) |
| Port 5001/8000/8080 bentrok (jaringan sama, VM Remote-SSH bersama) | 4,5,6 | Instruksikan tiap mahasiswa pakai port unik, mis. `8000+<3 digit NIM terakhir>` — tidak relevan kalau tiap mahasiswa pakai laptop sendiri |
| `mlflow.exceptions.MlflowException` saat `retrain_and_compare.py` | 8 | Belum ada alias `production` — jalankan `mlflow/train_with_mlflow.py` dulu (Sesi 4) |
| Pipeline Sesi 8 selalu `NOT_PROMOTED`, mahasiswa bingung dikira error | 8 | Ini **ekspektasi normal** kalau data retraining identik — tekankan ini bukti gerbang kualitas bekerja, bukan kegagalan |
| Prefect flow tidak menampilkan progres | 8 | Pastikan `log_prints=True` ada di setiap `@task`/`@flow` |

---


