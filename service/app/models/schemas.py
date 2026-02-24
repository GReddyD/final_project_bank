"""
Pydantic модели для API сервиса рекомендации банковских продуктов.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    """Признаки клиента для получения рекомендаций."""

    age: int = Field(..., ge=0, le=150, description="Возраст клиента")
    renta: Optional[float] = Field(None, description="Доход домохозяйства")
    antiguedad: Optional[int] = Field(None, description="Стаж клиента в банке (месяцы)")
    sexo: Optional[str] = Field(None, description="Пол (H=мужской, V=женский)")
    segmento: Optional[str] = Field(
        None, description="Сегмент: 01 - TOP, 02 - PARTICULARES, 03 - UNIVERSITARIO"
    )
    canal_entrada: Optional[str] = Field(None, description="Канал привлечения клиента")
    pais_residencia: Optional[str] = Field(None, description="Страна резидентства")
    nomprov: Optional[str] = Field(None, description="Название провинции")
    ind_empleado: Optional[str] = Field(
        None, description="Статус занятости: A, B, F, N, S"
    )
    tiprel_1mes: Optional[str] = Field(
        None, description="Тип отношений: A=активный, I=неактивный, P=бывший, R=потенциальный"
    )
    indresi: Optional[str] = Field(None, description="Резидент страны банка (S/N)")
    indext: Optional[str] = Field(
        None, description="Страна рождения != страна банка (S/N)"
    )
    indfall: Optional[str] = Field(None, description="Актуальность счёта (N/S)")
    indrel_1mes: Optional[str] = Field(
        None, description="Тип клиента в начале месяца: 1, 2, P, 3, 4"
    )
    ind_nuevo: Optional[int] = Field(
        None, description="1 = зарегистрирован за последние 6 мес."
    )
    indrel: Optional[int] = Field(None, description="1=первичный клиент, 99=иначе")
    ind_actividad_cliente: Optional[int] = Field(
        None, description="Активность клиента (1=активный, 0=нет)"
    )
    cod_prov: Optional[int] = Field(None, description="Код провинции")

    prev_products: dict[str, int] = Field(
        ..., description="Текущие продукты клиента {product_col: 0/1}"
    )
    product_changes: Optional[int] = Field(
        0, description="Количество изменений продуктов с прошлого месяца"
    )
    fecha_dato: Optional[str] = Field(
        None, description="Дата наблюдения (YYYY-MM-DD), по умолчанию сегодня"
    )
    top_k: int = Field(7, ge=1, le=22, description="Количество рекомендаций")


class ProductRecommendation(BaseModel):
    """Одна рекомендация продукта."""

    product_col: str = Field(..., description="Имя колонки продукта")
    product_name: str = Field(..., min_length=1, description="Название продукта на русском")
    probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность подключения")


class PredictionResponse(BaseModel):
    """Ответ с рекомендациями продуктов."""

    recommendations: list[ProductRecommendation] = Field(
        ...,
        min_length=0,
        max_length=22,
        description="Список рекомендованных продуктов",
    )
    n_current_products: int = Field(
        ..., ge=0, description="Количество текущих продуктов клиента"
    )


class HealthResponse(BaseModel):
    """Ответ проверки здоровья сервиса."""

    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    n_models: int = Field(..., ge=0, description="Количество классификаторов")
    n_features: int = Field(..., ge=0, description="Количество признаков")
    n_products: int = Field(..., ge=0, description="Количество продуктов")


class ModelInfoResponse(BaseModel):
    """Информация о модели."""

    n_models: int = Field(..., ge=0, description="Количество классификаторов")
    feature_cols: list[str] = Field(..., description="Список признаков модели")
    product_cols: list[str] = Field(..., description="Список продуктов")
    product_names: dict[str, str] = Field(
        ..., description="Маппинг продукт -> название"
    )
