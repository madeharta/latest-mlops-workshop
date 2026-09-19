"""
Sesi 6 -- Menguji endpoint yang sudah di-deploy secara lokal (Docker Compose).

Jalankan setelah docker/deploy.sh selesai:
    python3 scripts/test_deployment.py
    # atau, kalau menguji endpoint lain (mis. tunnel ngrok opsional):
    python3 scripts/test_deployment.py http://localhost:8080
"""
import sys
import time
import requests

SERVICE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")

payload = {
    "age": 34, "income": 78000, "bmi": 24.5, "tenure_months": 18,
    "num_claims": 1, "credit_score": 720, "policy_value": 150000, "risk_score": 42.5,
}

print(f">> Menguji {SERVICE_URL}")
print(">> Health check...")
r = requests.get(f"{SERVICE_URL}/")
print(r.status_code, r.json())

print("\n>> Predict...")
start = time.time()
r2 = requests.post(f"{SERVICE_URL}/predict", json=payload)
elapsed = time.time() - start
print(r2.status_code, r2.json())
print(f"Latency: {elapsed*1000:.0f} ms")
