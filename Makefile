# Sesi 2 -- Automation layer. Windows: gunakan Git Bash, WSL2, atau Dev Container.

setup:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

run:
	python3 train.py

test:
	pytest tests/ -v
