# Sesi 6 -- Dockerfile untuk aplikasi FastAPI insurance-approval-api.
#
# Build & jalankan manual (untuk belajar siklus Docker -- README Sesi 6 Bagian A):
#   cd mlops-workshop
#   docker build -f docker/Dockerfile -t insurance-api .
#   docker run -d -p 8080:8080 --name api insurance-api
#
# Build & deploy otomatis via Docker Compose (Bagian B -- deployment lokal,
# menggantikan Cloud Run untuk pelaksanaan workshop ini -- lihat catatan
# riwayat keputusan di docker/docker-compose.yml):
#   ./docker/deploy.sh
#
# PENTING: context build ini adalah ROOT proyek (mlops-workshop/), bukan
# folder docker/ -- karena perlu mengakses api/ dan api/models/.

FROM python:3.11-slim
WORKDIR /app

# requirements di-copy dan di-install SEBELUM kode aplikasi --
# agar layer pip install ter-cache saat kode berubah (lihat README Sesi 6)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/app.py .
COPY api/models/ ./models/

ENV PORT=8080
EXPOSE 8080

# 'exec' penting: uvicorn menjadi PID 1, sehingga menerima SIGTERM
# dengan benar saat container dihentikan/diganti (docker compose down/up).
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
