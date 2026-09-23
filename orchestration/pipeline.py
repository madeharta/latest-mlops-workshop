"""
Orkestrasi CI/CD/CT dengan Prefect, sepenuhnya lokal.

Deployment (CI build image + CD redeploy) berjalan seluruhnya di Docker lokal
lewat docker/deploy.sh

Jalankan (dari root proyek, venv sudah aktif):
    python3 orchestration/pipeline.py

⚠ CATATAN WINDOWS: subprocess di bawah memanggil sys.executable (interpreter
Python yang sedang menjalankan pipeline.py ini), BUKAN string "python3" yang
di-hardcode. Di Windows, venv hanya membuat python.exe -- tidak ada
python3.exe sama sekali. Kalau "python3" dipanggil langsung, Windows bisa
saja menemukan Python LAIN di PATH (mis. dari Microsoft Store) yang tidak
punya pandas/evidently/mlflow ter-install -- subprocess itu lalu gagal
dengan exit code bukan-nol, muncul di Prefect sebagai
CalledProcessError(1, ['python3', ...]). sys.executable menjamin subprocess
memakai environment yang SAMA PERSIS dengan yang menjalankan pipeline.py,
di OS mana pun.
"""
import subprocess
import sys
from prefect import flow, task


@task(retries=2, retry_delay_seconds=5, log_prints=True)
def check_drift() -> bool:
    """Menjalankan monitor.py dan membaca hasilnya untuk 
    dijadikan keputusan yang memicu langkah berikutnya."""
    result = subprocess.run(
        [sys.executable, "monitoring/monitor.py"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
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
        [sys.executable, "mlflow/retrain_and_compare.py"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    print(result.stdout)
    return "PROMOTED" in result.stdout and "NOT_PROMOTED" not in result.stdout


@task(retries=3, retry_delay_seconds=10, log_prints=True)
def build_and_deploy_local() -> None:
    """CI (build image dari model production terbaru) + CD (redeploy
    container lokal) dalam satu langkah -- memanggil docker/deploy.sh, yang
    di dalamnya: export_model.py -> docker compose build -> up --force-recreate
    -> smoke test.

    ⚠ WINDOWS: "bash" di bawah ini butuh Git Bash atau WSL2 di PATH -- Command
    Prompt/PowerShell murni tidak punya bash. Ini konsisten dengan kebutuhan
    Windows untuk `make` di Sesi 2 (lihat README). Jalankan pipeline ini dari
    dalam Git Bash/WSL2 kalau ingin build_and_deploy_local() ikut teruji."""
    result = subprocess.run(
        ["bash", "docker/deploy.sh"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    print(result.stdout)


@flow(name="insurance-approval-ci-cd-ct", log_prints=True)
def mlops_pipeline():
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
    #    interval=1,
    #)
    #mlops_pipeline.serve(
    #    name="insurance-approval-daily",
    #    schedule=Cron(
    #        "0 2 * * *",
    #        timezone="Asia/Jakarta"
    #    )
    #)
