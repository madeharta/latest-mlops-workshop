docker run -p 9002:8000 \
  -v $(pwd)/monitoring/.workspace:/app/workspace \
  --name evidently-service \
  --detach \
  evidently/evidently-service:latest

# Script antisipasi shell tidak bisa dijalankan
# docker run -p 9002:8000 -v "${PWD}/monitoring/.workspace:/app/workspace" --name evidently-service --detach evidently/evidently-service:latest