"""
FastAPI приложение для сервиса рекомендации банковских продуктов.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.models import (
    ClientFeatures,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.core import ServiceConfig, ModelStore, BankPredictor
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные объекты
_store: Optional[ModelStore] = None
_predictor: Optional[BankPredictor] = None


def get_store() -> Optional[ModelStore]:
    """Получить экземпляр хранилища модели."""
    return _store


def get_predictor() -> Optional[BankPredictor]:
    """Получить экземпляр предиктора."""
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle события приложения."""
    global _store, _predictor

    # Startup: загружаем модель
    model_path = os.getenv("MODEL_PATH", "/app/models/model.bin")

    # Конфигурация из переменных окружения с валидацией через pydantic
    try:
        config = ServiceConfig(
            model_path=model_path,
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "7")),
        )
    except ValueError as e:
        logger.error(f"Ошибка валидации конфигурации: {e}")
        raise

    try:
        _store = ModelStore()
        _store.load_model(config.model_path)
        _predictor = BankPredictor(store=_store, config=config)
        logger.info(
            f"Предиктор инициализирован: "
            f"{_store.n_models} классификаторов, "
            f"{_store.n_features} признаков, "
            f"default_top_k={config.default_top_k}"
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise

    yield
    # Shutdown
    logger.info("Завершение работы сервиса")


app = FastAPI(
    title="Bank Product Recommender",
    description="Микросервис для рекомендации банковских продуктов на основе ML-модели",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Middleware: сбор метрик по HTTP-запросам
# ============================================================================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware для сбора метрик latency и throughput по каждому запросу."""
    # Не считаем метрики для самого эндпоинта /metrics
    if request.url.path == "/metrics":
        return await call_next(request)

    method = request.method
    endpoint = request.url.path
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start
    status_code = str(response.status_code)

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    return response


# ============================================================================
# Эндпоинты API
# ============================================================================

@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus-метрики сервиса."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Проверка состояния сервиса."""
    store = get_store()
    if store is None or not store.is_loaded():
        raise HTTPException(status_code=503, detail="Сервис не готов — модель не загружена")

    stats = store.get_stats()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        n_models=stats["n_models"],
        n_features=stats["n_features"],
        n_products=stats["n_products"],
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """
    Информация о загруженной модели.

    Возвращает:
    - Количество классификаторов
    - Список признаков модели
    - Список продуктов с названиями
    """
    store = get_store()
    if store is None or not store.is_loaded():
        raise HTTPException(status_code=503, detail="Сервис не готов — модель не загружена")

    return ModelInfoResponse(
        n_models=store.n_models,
        feature_cols=store.feature_cols,
        product_cols=store.product_cols,
        product_names=store.product_names,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    """
    Получение рекомендаций продуктов для клиента.

    - **age**: Возраст клиента (обязательный)
    - **prev_products**: Текущие продукты клиента (обязательный)
    - **top_k**: Количество рекомендаций (по умолчанию 7)

    Возвращает список рекомендованных продуктов, отсортированных по вероятности,
    исключая уже имеющиеся у клиента продукты.
    """
    predictor = get_predictor()
    store = get_store()

    if store is None or not store.is_loaded():
        raise HTTPException(status_code=503, detail="Сервис не готов — модель не загружена")

    try:
        result = predictor.predict(client)
    except Exception as e:
        logger.error(f"Ошибка инференса: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Ошибка предсказания: {e}")

    return PredictionResponse(**result)
