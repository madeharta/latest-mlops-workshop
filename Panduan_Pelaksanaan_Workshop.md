# Panduan Pelaksanaan Workshop MLOps
### Sesi 1 s.d. Sesi 6 — Panduan Fasilitator/Instruktur

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
| 5 | Evidently AI (3 sinyal: data/prediction/performance) | 120 |
| 6 | Prefect + Penutup | 60 |
---

**Notasi yang dipakai:**
- `[Detail #N]` = nomor slide di `Materi MLOps Day-3.pptx`
- Blok kode = perintah persis yang diketik/dijalankan di terminal

---

## 1. Sesi 1 — Pengantar Workshop MLOPs DAY-3

**Tidak ada kode.** Sesi ini murni konseptual.

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #1–2]` — judul & peta 8 sesi, orientasikan mahasiswa ke keseluruhan arc kuliah |
| B (60’) | `[Detail #3–7]`: divider → proses masalah klasik → definisi MLOps → maturity model → peta arsitektur |
| Diskusi (30’) | Slide `[Detail #4]` memuat instruksi diskusi kelas: minta mahasiswa membawa/menceritakan satu studi kasus kegagalan deployment ML, identifikasi tahap mana yang absen |
| C/D | *(tidak berlaku — tidak ada lab hands-on)* |
| E (5’) | `[Detail #8]` Rangkuman Sesi 1 |

**Persiapan sebelum kelas:** minta mahasiswa memikirkan satu contoh (media/berita/pengalaman) sebagai bahan diskusi 30 menit di atas — sampaikan ini di akhir sesi sebelumnya atau lewat pengumuman H-3.

## 2. Persiapan Awal

### 2.1 Tugas Instruktur

Mendistribusikan kode resource yang diperlukan dan memantau/mendampingi proses konfigurasi environment untuk persiapan workshop. Memastikan semua mahasiswa sudah melakukan proses konfigurasi dengan benar, sesuai requirements.
\
> **Catatan pendekatan workshop:** deployment (Sesi 6) dan orkestrasi (Sesi 8) berjalan sepenuhnya lokal via Docker 

### 2.2 Tugas Mahasiswa (Sesi 0, pada awal sesi - 15 Menit)

Checklist ini ada di `README.md` bagian "Sesi 0", mahasiswa menjalankannya mandiri dan melapor jika ada kegagalan pada awal sesi terkait konfigurasi environment. Hal-hal yang perlu dicek:

#### Mahasiswa memastikan sudah menginstall Docker Desktop dan VSCode serta melakukan pre-setup

```bash
open -a Docker                                        # Menjalankan Docker Desktop
code mlops-workshop                                   # Membuka kode sumber dengan VSCode

python3 --version                                     # harus >3.10 
python3 -m venv .venv && source .venv/bin/activate    # 3.11-3.14 valid; 3.11 default teraman. Untuk Windows aktivatsi virtual envirnomen dilukan dengan menjalankan perintah .venv/scripts/activate

python3 -m pip install -r requirements.txt            # Install semua tools dan library yang diperlukan untuk menjalankan seluruh aktivitas
python3 -c "import numpy; print(numpy.__version__)"   # harus >= 1.26 (tanpa batas atas)
docker run hello-world                                # daemon Docker hidup
```

**Warning** Pastikan semua proses berjalan dengan baik tanpa error. Jika terjadi error silahkan laporkan ke Dosen atau TA, sebelum melanjutkan ke tahapan berikutnya. Kegagalan pada proses konfigurasi awal akan berdampak pada gagalnya tahapan workshop selanjutnya.

---

## 3. Sesi 1 — Reproducibility & Struktur Proyek

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #7–8]` |
| B (20’) | `[Detail #9]` divider, `[Detail #15]` anatomi struktur proyek |
| C (40’) — demo langsung | `[Detail #10]` langkah setup → `[Detail #11–14]` config.yml/train.py/Makefile |

**Skrip demo (jalankan persis, proyeksikan terminal):**
```bash
cd mlops-workshop
cat config.yml              # tunjukkan model+params
cat train.py                # tunjukkan pembacaan config, tanpa hardcode
make run
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
make run     # accuracy berubah, train.py tidak disentuh sama sekali
make test    # 2 test pytest lulus

# Jika Makefile tidak didukung, lakukan secara manual untuk pengujian
python3 train.py    # menjalankan trainning
pytest tests/ -v    # menjalankan testing model
```

| Fase | Slide/Aktivitas |
|---|---|
| D (35’) — lab mandiri | Mahasiswa mengulang langkah di atas di laptop sendiri; tantangan tambahan: ganti ke `DecisionTree` dan bandingkan akurasi tiga model secara manual |
| E (10’) | `[Detail #16]` Rangkuman Sesi 2 |


---

## 4. Sesi 2 — Experiment Tracking & Model Registry (MLflow)

Sesi ini yang paling padat tutorialnya — pertimbangkan mengurangi Fase B jika waktu ketat.

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #11–12]` |
| B (25’) | `[Detail #25–27]` divider → masalah tanpa tracking → tiga komponen MLflow |
| **Gotcha wajib disampaikan** | `[Detail #28]` — backend `file:///` deprecated, HARUS pakai `sqlite:///` |
| C (45’) — demo langsung | `[Detail #29–37]` |

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
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 9001
```
Buka `http://127.0.0.1:5001` di browser, tunjukkan tabel perbandingan (bandingkan dengan `[Detail #33]` screenshot), klik ke Model Registry, tunjukkan versi 1 ter-registrasi.

> ⚠️ **warning**: `mlflow ui` bisa gagal berjalan jika port 9001 sudah dipakai oleh aplikasi lain. Pastikan port 9001 belum digunakan, jika sudah pilih port lain rekomendasi > 9000.

| Fase | Slide/Aktivitas |
|---|---|
| D (40’) — lab mandiri | Mahasiswa menjalankan `train_with_mlflow.py`, membuka UI di port unik per orang (`--port 500X`) agar tidak bentrok jika berbagi jaringan yang sama, lalu registrasi model sendiri |
| Diskusi (10’) | `[Detail #38]` MLflow vs W&B vs Neptune.ai |
| E (5’) | `[Detail #39]` Rangkuman Sesi 4 |

---

## 5. Sesi 3 — Serving Model dengan FastAPI

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #13–14]` |
| B (20’) | `[Detail #40–41]` divider → pola-pola serving |
| C (45’) — demo langsung | `[Detail #42–49]` |

**Skrip demo:**
```bash
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
```
Tunjukkan eksekusi berhenti di breakpoint, inspeksi variabel `df`.

**Demo validasi gagal (`[Detail #49]`)** — kirim payload dengan field hilang, tunjukkan HTTP 422 otomatis tanpa kode tambahan.

| Fase | Slide/Aktivitas |
|---|---|
| D (40’) — lab mandiri | Mahasiswa menjalankan sendiri, mencoba breakpoint sendiri, lalu **mengubah satu field jadi tipe salah** untuk melihat pesan error 422 |
| Diskusi (10’) | `[Detail #50]` REST vs gRPC vs TorchServe/Triton |
| E (5’) | `[Detail #51]` Rangkuman Sesi 5 |

---

## 6. Sesi 4 — Kontainerisasi & Deployment (Docker, Lokal)

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #15–16]` |
| B (20’) | `[Detail #52–55]` divider → image vs container → Dockerfile → kenapa urutan instruksi penting |

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
chmod +x docker/deploy.sh
./docker/local_deploy.sh
```
Script ini menjalankan CI (export model + build image) dan CD (redeploy container + smoke test) berurutan — jelaskan tiap tahap saat log muncul di terminal.

```bash
# [Detail #61] kode pengujian
python3 scripts/test_deployment.py
# default menguji http://localhost:8080 -- tidak perlu argumen URL
```

`[Detail #62]` (restart policy & health check) — jelaskan `restart: unless-stopped` dan `healthcheck` sebagai pengganti auto-healing pada layanan server produksi yang umum dipakai.

| Fase | Slide/Aktivitas |
|---|---|
| D (45’) — lab mandiri | Mahasiswa mengulang Bagian A, lalu `./docker/deploy.sh` sendiri di laptop masing-masing (tiap mahasiswa punya `localhost:8080` sendiri — tidak ada konflik nama seperti di Cloud Run) |
| **Sebelum sesi selesai** | `[Detail #63]` — `docker compose -f docker/docker-compose.yml down` untuk membebaskan port; tidak ada kontrol biaya cloud yang perlu dikhawatirkan |
| Diskusi (10’) | `[Detail #64]` Deployment Lokal vs Cloud Run vs ECS/Fargate vs Kubernetes — tabel sudah menyertakan baris "Docker Compose (lokal)" sebagai baseline pembanding |
| E (10’) | `[Detail #65]` Rangkuman Sesi 6 |

---

## 7. Sesi 5 — Monitoring Model di Produksi (Evidently AI)

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #17–18]` |
| B (25’) | `[Detail #66–68]` divider → taksonomi tiga sinyal → tabel perbandingan (butuh ground truth? kecepatan? yang diukur?) |
| **Gotcha wajib disampaikan** | `[Detail #69]` — dua lapis pinning versi: evidently/numpy DAN plotly (lihat detail di bawah) |
| C (45’) — demo langsung | `[Detail #70–74]` |

**Skrip demo:**
```bash
python3 monitoring/monitor.py
```
Output nyata:
```
Laporan tersimpan di monitoring/drift_report.html

1. DATA DRIFT        -- 3/8 kolom input drift (share: 0.375)
2. PREDICTION DRIFT  -- tidak terdeteksi pada output model
3. MODEL/CONCEPT DRIFT (performance) -- accuracy: 0.9380 -> 0.8880  |  ROC-AUC: 0.9817 -> 0.9575

⚠ Kebijakan retraining melewati ambang (data drift 0.38 >= ambang 0.3; accuracy turun 0.0500 >= ambang 0.03) -- pertimbangkan retraining.
```
Buka `monitoring/drift_report.html` di browser — scroll ke tiga bagian: tabel data drift (histogram per fitur, `[Detail #73]` screenshot), skor drift kolom `prediction`, dan panel *Classification Performance* (`[Detail #74]` screenshot — accuracy/ROC-AUC reference vs current).

**Poin pedagogis kunci — tekankan ini secara eksplisit (`[Detail #75]`)**: prediction drift **tidak terdeteksi** secara statistik pada output di atas, padahal accuracy sungguhan turun 5 poin persentase. Ini bukti langsung kenapa memantau satu sinyal saja (prediction drift lebih murah karena tak perlu menunggu label) bisa **melewatkan** penurunan performa nyata. Jadikan ini diskusi: "kalau tim kalian hanya pasang alert di prediction drift, insiden ini tidak akan pernah terdeteksi."

`[Detail #76]` (kebijakan operasional) — jelaskan bahwa trigger retraining memakai **OR dari dua ambang** (data drift ≥0.30 ATAU accuracy turun ≥0.03), bukan cuma satu sinyal — langsung tersambung ke temuan di atas.

| Fase | Slide/Aktivitas |
|---|---|
| D (35’) — lab mandiri | Mahasiswa menjalankan sendiri, lalu memodifikasi `data/production.csv` (geser kolom lain, atau kembalikan salah satu dari tiga kolom yang drift ke distribusi asli) untuk mengamati bagaimana ketiga angka bereaksi berbeda |
| Diskusi (15’) | `[Detail #77–79]` sumber data produksi bermakna → shadow/canary/A-B testing (S2) → keterbatasan statistical drift detection & label latency |
| E (5’) | `[Detail #80]` Rangkuman Sesi 7 |

---

## 8. Sesi 6 — MLOps Lanjut, Tata Kelola & Proyek Akhir

| Fase | Slide/Aktivitas |
|---|---|
| A (5’) | `[Ringkas #19–20]` |
| B (30’) | `[Detail #81–84]` divider → masalah orkestrasi manual → apa yang ditambahkan orkestrasi → tabel 4 tools |
| **Keputusan tool** | `[Detail #85]` — kenapa Prefect dipilih untuk workshop ini |
| C (35’) — demo langsung | `[Detail #86–87]` |

**Skrip demo:**
```bash
pip install -r orchestration/requirements.txt
python -m prefect server start
python3 orchestration/pipeline.py
```
Tunjukkan log Prefect live di terminal — perhatikan tiga hal: (1) nama task (`check_drift-xxx`, `retrain_and_compare-xxx`), (2) baris `PROMOTED`/`NOT_PROMOTED` yang menentukan apakah deploy dijalankan, dan (3) bahwa `build_and_deploy_local()` **hanya terpanggil kalau** kandidat baru benar-benar lebih baik dari `production` saat ini. Karena deployment sepenuhnya lokal (Docker, bukan GCP), demo ini bisa dibiarkan berjalan sampai selesai tanpa bergantung kredensial cloud apa pun.

> **Poin pedagogis kunci**: kemungkinan besar hasil pertama adalah `NOT_PROMOTED` (data retraining identik dengan data yang sudah dipakai) — ini **bukan kegagalan demo**, justru bukti gerbang kualitas bekerja. Siapkan skenario `PROMOTED` sebagai cadangan: jalankan `mlflow/retrain_and_compare.py` secara terpisah setelah mengubah `data/train.csv` (mis. gabungkan dengan data tambahan), atau gunakan skrip verifikasi mekanisme di `README.md` bagian Sesi 8 yang mendemonstrasikan alias berpindah secara terisolasi.

| Fase | Slide/Aktivitas |
|---|---|
| D (30’) — lab mandiri | Mahasiswa menjalankan `orchestration/pipeline.py` sendiri; tantangan: ubah `THRESHOLD` di `monitoring/monitor.py` agar drift **tidak** melewati ambang, jalankan ulang, verifikasi `retrain_and_compare()` tidak terpanggil sama sekali |
| B lanjutan (25’) | `[Detail #88–90]` sintesis arsitektur → tata kelola & responsible AI → arah riset LLMOps |
| E (10’) | `[Detail #91–92]` Rangkuman Sesi 8 + penutup "Satu Pipeline, Delapan Sesi" |
| Penutup kelas | `[Ringkas #22]` Proyek Akhir — sampaikan rubrik & linimasa (§11 di bawah) |

---

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


