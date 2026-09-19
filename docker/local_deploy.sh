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
    echo "[1/2] Checking model file..."

    if [ ! -f "${MODEL_SOURCE}" ]; then
        echo "ERROR: ${MODEL_SOURCE} not found."
        exit 1
    fi

    echo "Found: ${MODEL_SOURCE}"

    echo ""
    echo "[2/2] Copying new model into container..."

    docker cp \
        "${MODEL_SOURCE}" \
        "${CONTAINER_NAME}:${MODEL_DEST}"

    echo ""
    echo "Restarting container..."

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

    echo ""
    echo "[1/3] Checking model file..."

    if [ ! -f "${MODEL_SOURCE}" ]; then
        echo "ERROR: ${MODEL_SOURCE} not found."
        exit 1
    fi

    echo "Found: ${MODEL_SOURCE}"

    echo ""
    echo "[2/3] Building Docker image..."

    docker build \
        -f docker/Dockerfile \
        -t "${IMAGE_NAME}:latest" \
        .

    echo ""
    echo "[3/3] Starting container..."

    docker run -d \
        -p "${HOST_PORT}:${CONTAINER_PORT}" \
        --name "${CONTAINER_NAME}" \
        "${IMAGE_NAME}:latest"

    echo ""
    echo "======================================"
    echo " First deployment completed"
    echo "======================================"

fi


# ============================================================
# Show container status
# ============================================================

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


