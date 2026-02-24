"""
Конфигурация сервиса рекомендации банковских продуктов с pydantic валидацией.
"""

from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    """
    Конфигурация сервиса с валидацией параметров.

    Attributes:
        model_path: Путь к файлу модели (model.bin).
        default_top_k: Количество рекомендаций по умолчанию.
            Должно быть от 1 до 22.
    """

    model_path: str = Field(
        default="/app/models/model.bin",
        min_length=1,
        description="Путь к файлу модели",
    )
    default_top_k: int = Field(
        default=7,
        ge=1,
        le=22,
        description="Количество рекомендаций по умолчанию",
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }
