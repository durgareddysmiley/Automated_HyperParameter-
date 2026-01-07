FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY notebooks/ notebooks/

# 🔥 CREATE OUTPUTS WITH PERMISSION
RUN mkdir -p /app/outputs && chmod -R 777 /app/outputs

ENV MLFLOW_TRACKING_URI=file:///app/outputs/mlruns

CMD ["python", "src/optimize.py"]
