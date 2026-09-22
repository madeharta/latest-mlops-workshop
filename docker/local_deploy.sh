#!/usr/bin/env bash
# Sesi 6 -- Deploy to local container via docker.
#
# EDIT dua variabel di bawah sebelum menjalankan, lalu:
#   chmod +x docker/deploy.sh
#   ./docker/deploy.sh
#
# Dijalankan dari ROOT proyek (mlops-workshop/), bukan dari dalam docker/.

#!/bin/bash

#!/bin/bash

set -e

IMAGE_NAME="insurance-api"
CONTAINER_NAME="api"

HOST_PORT=8080
CONTAINER_PORT=8080

MODEL_SOURCE="api/models/model.pkl"
MODEL_DEST="/app/models/model.pkl"


echo "======================================"
echo " Insurance Approval API Deployment"
echo "======================================"


# ============================================================
# Check whether the container already exists
# ============================================================

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then

    # ========================================================
    # EXISTING CONTAINER
    # ========================================================

    echo ""
    echo "Existing container '${CONTAINER_NAME}' found."
    echo "Mode: MODEL UPDATE"

    echo ""
    echo "Checking model file..."

    if [ ! -f "${MODEL_SOURCE}" ]; then
        echo "ERROR: ${MODEL_SOURCE} not found."
        exit 1
    fi

    echo "Found: ${MODEL_SOURCE}"

    echo ""
    echo "Copying new model into container..."

    docker cp \
        "${MODEL_SOURCE}" \
        "${CONTAINER_NAME}:${MODEL_DEST}"

    echo ""
    echo "Restarting container..."

    # Need to restart the container so that the new model gets reloaded
    docker restart "${CONTAINER_NAME}"

    echo ""
    echo "======================================"
    echo " Model update completed"
    echo "======================================"

else

    # ========================================================
    # FIRST DEPLOYMENT
    # ========================================================

    echo ""
    echo "Container '${CONTAINER_NAME}' does not exist."
    echo "Mode: FIRST DEPLOYMENT"

    echo ">> [CI] Menarik model 'production' terbaru dari MLflow Registry..."
    python3 mlflow/export_model.py

    echo ">> [CI] Build image dari kode + model terbaru..."
    docker compose -f docker/docker-compose.yml build

    echo ">> [CD] Redeploy container (mengganti yang lama jika masih berjalan)..."
    docker compose -f docker/docker-compose.yml up -d --force-recreate

    echo ""
    echo "======================================"
    echo " First deployment completed"
    echo "======================================"

fi


# ============================================================
# Show container status
# ============================================================

echo ">> Menunggu container siap..."
sleep 4

echo ""
echo "Container status:"
docker ps --filter "name=${CONTAINER_NAME}"

echo ""
echo "API:"
echo "http://127.0.0.1:${HOST_PORT}"

echo ""
echo "Swagger:"
echo "http://127.0.0.1:${HOST_PORT}/docs"

echo ""
echo "======================================"

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


