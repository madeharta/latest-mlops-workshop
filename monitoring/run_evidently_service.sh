docker run -p 9002:8000 \
  -v $(pwd)/.workspace:/app/workspace \
  --name evidently-service \
  --detach \
  evidently/evidently-service:latest