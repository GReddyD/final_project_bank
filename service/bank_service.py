"""
Точка входа для запуска FastAPI сервиса рекомендации банковских продуктов.

Сервис поддерживает:
- Предсказание новых банковских продуктов для клиента (multi-label классификация)
- Информацию о загруженной модели (22 LightGBM-классификатора)
- Проверку состояния сервиса

Запуск:
    python bank_service.py

Или с помощью uvicorn:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os

import uvicorn

from app import app


if __name__ == "__main__":
    # Параметры запуска
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
