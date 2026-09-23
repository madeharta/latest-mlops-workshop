#!/usr/bin/env bash
# Sesi 6 -- CI/CD lokal: export model production terbaru -> build image ->
# redeploy container -> smoke test. Pengganti deployment ke Google Cloud Run
# (lihat catatan riwayat keputusan di docker/docker-compose.yml).
#
# Dipanggil manual:
#   chmod +x docker/deploy.sh
#   ./docker/deploy.sh
#
# Atau otomatis oleh orchestration/pipeline.py (Sesi 8) setiap kali model
# baru terbukti lebih baik dari production saat ini.
#
# Dijalankan dari ROOT proyek (mlops-workshop/), bukan dari dalam docker/.

set -euo pipefail
cd "$(dirname "$0")/.."   # pastikan selalu jalan dari root proyek

# Resolusi interpreter Python yang benar-benar aktif -- BUKAN asumsi "python3"
# ada di PATH. Ini bug lintas-platform yang nyata: di Windows, venv hanya
# membuat Scripts/python.exe, TIDAK ADA python3.exe sama sekali. Kalau baris
# ini diganti "python3" polos, Git Bash bisa saja menemukan Python LAIN di
# PATH (dari Program Files, WSL alias, dsb.) yang tidak punya
# pandas/joblib/mlflow ter-install -- persis gejala ModuleNotFoundError yang
# terlihat aneh padahal (.venv) sudah tampak aktif di prompt.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/Scripts/python.exe"      # venv aktif, layout Windows
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"              # venv aktif, layout Unix
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"                              # fallback -- tidak ada venv aktif
else
    PYTHON_BIN="python"
fi
echo ">> Memakai interpreter: $PYTHON_BIN"

echo ">> [CI 1/2] Menarik model 'production' terbaru dari MLflow Registry..."
"$PYTHON_BIN" mlflow/export_model.py

echo ">> [CI 2/2] Build image dari kode + model terbaru..."
docker compose -f docker/docker-compose.yml build

echo ">> [CD] Redeploy container (mengganti yang lama jika masih berjalan)..."
docker compose -f docker/docker-compose.yml up -d --force-recreate

echo ">> Menunggu container siap..."
sleep 4

echo ">> Smoke test endpoint..."
if curl -sf http://localhost:8080/ > /dev/null; then
    echo ">> Deploy lokal BERHASIL. Endpoint: http://localhost:8080"
    echo "   Uji manual: curl http://localhost:8080/"
    echo "   Docs interaktif: http://localhost:8080/docs"
else
    echo ">> Smoke test GAGAL -- cek log dengan:"
    echo "   docker compose -f docker/docker-compose.yml logs"
    exit 1
fi

echo ""
echo ">> Untuk menghentikan & membersihkan:"
echo "   docker compose -f docker/docker-compose.yml down"