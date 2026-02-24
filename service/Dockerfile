FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/bank_service.py ./
COPY service/app/ ./app/
COPY models/model.bin ./models/model.bin

ENV MODEL_PATH=/app/models/model.bin

EXPOSE 8000

CMD ["python", "bank_service.py"]
