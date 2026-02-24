"""
Хранилище модели: загрузка и валидация артефакта model.bin.
"""

import logging
import time

import joblib

from app.core.metrics import MODEL_LOAD_TIME, MODEL_INFO

logger = logging.getLogger(__name__)

# Ожидаемые ключи в артефакте модели
EXPECTED_KEYS = {"models", "feature_cols", "target_cols", "product_cols",
                 "label_encoders", "product_names", "top20_canal", "top20_prov"}


class ModelStore:
    """
    Хранилище модели — загрузка, валидация и доступ к артефактам.

    Загружает model.bin (dict) с 22 LightGBM-классификаторами,
    LabelEncoder'ами, списками признаков и продуктов.
    """

    def __init__(self):
        self._artifact = None
        self._loaded = False

    def load_model(self, model_path: str) -> None:
        """
        Загрузка и валидация модели из файла.

        Args:
            model_path: Путь к файлу model.bin

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если структура артефакта некорректна
        """
        logger.info(f"Загрузка модели из {model_path}")
        start = time.perf_counter()
        self._artifact = joblib.load(model_path)
        self._validate()
        self._loaded = True
        load_time = time.perf_counter() - start

        MODEL_LOAD_TIME.set(load_time)
        MODEL_INFO.info({
            "n_models": str(self.n_models),
            "n_features": str(self.n_features),
            "n_products": str(self.n_products),
            "path": model_path,
        })

        logger.info(
            f"Модель загружена за {load_time:.2f}с: {self.n_models} классификаторов, "
            f"{self.n_features} признаков, {self.n_products} продуктов"
        )

    def _validate(self) -> None:
        """Валидация структуры артефакта."""
        errors = []

        # Проверка наличия всех ключей
        missing_keys = EXPECTED_KEYS - set(self._artifact.keys())
        if missing_keys:
            errors.append(f"Отсутствуют ключи: {missing_keys}")

        # Проверка моделей
        if "models" in self._artifact:
            n_models = len(self._artifact["models"])
            if n_models == 0:
                errors.append("Словарь моделей пуст")
            logger.debug(f"Загружено моделей: {n_models}")

        # Проверка совпадения product_cols и models
        if "models" in self._artifact and "product_cols" in self._artifact:
            model_keys = set(self._artifact["models"].keys())
            product_set = set(self._artifact["product_cols"])
            if model_keys != product_set:
                errors.append(
                    f"Несовпадение product_cols и ключей models: "
                    f"только в models={model_keys - product_set}, "
                    f"только в product_cols={product_set - model_keys}"
                )

        # Проверка label_encoders
        if "label_encoders" in self._artifact:
            n_enc = len(self._artifact["label_encoders"])
            if n_enc == 0:
                errors.append("Словарь label_encoders пуст")
            logger.debug(f"Загружено энкодеров: {n_enc}")

        if errors:
            error_msg = "; ".join(errors)
            raise ValueError(f"Ошибка валидации модели: {error_msg}")

    def is_loaded(self) -> bool:
        """Загружена ли модель."""
        return self._loaded

    def get_stats(self) -> dict:
        """Статистика по загруженной модели."""
        if not self._loaded:
            return {"n_models": 0, "n_features": 0, "n_products": 0}
        return {
            "n_models": self.n_models,
            "n_features": self.n_features,
            "n_products": self.n_products,
        }

    # --- Свойства для удобного доступа к артефактам ---

    @property
    def artifact(self) -> dict:
        return self._artifact

    @property
    def models(self) -> dict:
        return self._artifact["models"]

    @property
    def feature_cols(self) -> list:
        return self._artifact["feature_cols"]

    @property
    def product_cols(self) -> list:
        return self._artifact["product_cols"]

    @property
    def product_names(self) -> dict:
        return self._artifact["product_names"]

    @property
    def label_encoders(self) -> dict:
        return self._artifact["label_encoders"]

    @property
    def top20_canal(self) -> list:
        return self._artifact["top20_canal"]

    @property
    def top20_prov(self) -> list:
        return self._artifact["top20_prov"]

    @property
    def n_models(self) -> int:
        return len(self._artifact["models"])

    @property
    def n_features(self) -> int:
        return len(self._artifact["feature_cols"])

    @property
    def n_products(self) -> int:
        return len(self._artifact["product_cols"])
