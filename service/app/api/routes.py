"""
Эндпоинты API сервиса рекомендации банковских продуктов.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models import (
    ClientFeatures,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.main import get_store, get_predictor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
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


@router.get("/model/info", response_model=ModelInfoResponse)
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


@router.post("/predict", response_model=PredictionResponse)
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
