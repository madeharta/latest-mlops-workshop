"""
Sesi 8 -- Orkestrasi CI/CD/CT dengan Prefect, sepenuhnya lokal.

Membungkus skrip-skrip yang SUDAH ADA dari Sesi 4, 6, dan 7 -- tidak ada
logika ML yang ditulis ulang di sini. Yang ditambahkan Prefect:
  1. Urutan eksekusi yang tidak lagi ada "di kepala manusia"
  2. Retry otomatis saat satu langkah gagal (build Docker, dsb.)
  3. Percabangan kondisional GANDA:
       a. retrain HANYA JIKA drift melewati ambang (CT dipicu oleh monitoring)
       b. deploy HANYA JIKA model baru TERBUKTI lebih baik dari production
          saat ini -- lihat mlflow/retrain_and_compare.py. Tanpa gerbang ini,
          "retrain otomatis" adalah anti-pola: model produksi bisa memburuk
          tanpa disadari kalau data baru kebetulan menghasilkan model jelek.
  4. Riwayat run yang bisa diaudit (Prefect UI: `prefect server start`)

Deployment (CI build image + CD redeploy) berjalan seluruhnya di Docker lokal
lewat docker/deploy.sh -- BUKAN Cloud Build/Cloud Run. Ini pendekatan yang
dipakai untuk pelaksanaan workshop ini (keterbatasan environment GCP saat
kelas berlangsung); di lingkungan produksi sesungguhnya, task build_and_deploy()
di bawah ini yang akan diganti memanggil pipeline Cloud Build/Cloud Run --
struktur flow (urutan, kondisional, retry) tetap identik.

Jalankan (dari root proyek, venv sudah aktif):
    python3 orchestration/pipeline.py
"""
import subprocess
from prefect import flow, task


@task(retries=2, retry_delay_seconds=5, log_prints=True)
def check_drift() -> bool:
    """Menjalankan monitor.py Sesi 7 dan membaca hasilnya -- bukan sekadar
    dicetak, tapi dijadikan keputusan yang memicu langkah berikutnya."""
    result = subprocess.run(
        ["python3", "monitoring/monitor.py"], capture_output=True, text=True, check=True
    )
    print(result.stdout)
    return "melewati ambang" in result.stdout


@task(retries=1, log_prints=True)
def retrain_and_compare() -> bool:
    """CT -- Continuous Training. Melatih kandidat baru dari data terkini,
    lalu membandingkannya dengan model 'production' saat ini di MLflow
    Registry. Kandidat HANYA dipromosikan (alias production dipindah) kalau
    accuracy-nya lebih baik -- lihat mlflow/retrain_and_compare.py untuk
    logika lengkapnya. Return True hanya jika model baru dipromosikan."""
    result = subprocess.run(
        ["python3", "mlflow/retrain_and_compare.py"], capture_output=True, text=True, check=True
    )
    print(result.stdout)
    return "PROMOTED" in result.stdout and "NOT_PROMOTED" not in result.stdout


@task(retries=3, retry_delay_seconds=10, log_prints=True)
def build_and_deploy_local() -> None:
    """CI (build image dari model production terbaru) + CD (redeploy
    container lokal) dalam satu langkah -- memanggil docker/deploy.sh, yang
    di dalamnya: export_model.py -> docker compose build -> up --force-recreate
    -> smoke test. Retry di sini menjawab kegagalan build/startup sesaat,
    bukan lagi kegagalan jaringan gcloud seperti versi Cloud Run sebelumnya."""
    result = subprocess.run(
        #["bash", "docker/deploy.sh"], capture_output=True, text=True, check=True
        ["bash", "docker/local_deploy.sh"], capture_output=True, text=True, check=True
    )
    print(result.stdout)


@flow(name="insurance-approval-ci-cd-ct", log_prints=True)
def mlops_pipeline():
    """Satu graf yang menghubungkan Sesi 4, 6, dan 7 -- closing the loop
    yang di Sesi 7 masih berupa kalimat kebijakan di README, sekarang
    benar-benar dieksekusi, lengkap dengan gerbang kualitas sebelum deploy."""
    drift_exceeds_threshold = check_drift()

    if not drift_exceeds_threshold:
        print(">> Drift masih di bawah ambang -- tidak ada aksi.")
        return

    print(">> Drift melewati ambang -- memicu retraining (CT)...")
    promoted = retrain_and_compare()

    if not promoted:
        print(">> Model baru TIDAK lebih baik dari production -- deploy DIBATALKAN.")
        print(">> Model production saat ini tetap yang dipakai (tidak ada perubahan).")
        return

    print(">> Model baru lebih baik -- menjalankan build & deploy lokal (CI/CD)...")
    build_and_deploy_local()
    print(">> Pipeline selesai: model baru sudah live di http://localhost:8080")


if __name__ == "__main__":
    mlops_pipeline()
    #mlops_pipeline.serve(
    #    name="insurance-approval-test",
    #    interval=60,
    #)
    #mlops_pipeline.serve(
    #    name="insurance-approval-daily",
    #    schedule=Cron(
    #        "0 2 * * *",
    #        timezone="Asia/Jakarta"
    #    )
    #)
