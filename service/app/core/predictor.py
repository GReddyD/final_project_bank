"""
Препроцессинг и инференс для модели рекомендации банковских продуктов.

Воспроизводит пайплайн обработки из ноутбука modeling.ipynb:
1. Численные признаки: age (clip), log_renta, antiguedad, ind_nuevo, indrel, и др.
2. Категориальные: 8 малокардинальных (LabelEncoder), canal_entrada (top-20),
   pais_residencia (бинарный), nomprov (top-20)
3. Лаговые: 22 prev_product из prev_products
4. Агрегатные: n_products, product_changes
5. Временные: month, months_since_start
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.config import ServiceConfig
from app.core.store import ModelStore
from app.core.metrics import (
    PREDICTION_LATENCY,
    PREDICTION_ERRORS,
    PREDICTION_PROBABILITY,
    TOP1_PROBABILITY,
    RECOMMENDATIONS_COUNT,
    RECOMMENDED_PRODUCT,
    CLIENT_PRODUCTS_COUNT,
    CLIENT_AGE,
    TOP_K_REQUESTED,
)

logger = logging.getLogger(__name__)

# Дефолтные значения (моды/медианы из обучающей выборки)
DEFAULT_RENTA_BY_SEGMENT = {
    "01 - TOP": 234340.0,
    "02 - PARTICULARES": 101340.0,
    "03 - UNIVERSITARIO": 86200.0,
}
DEFAULT_RENTA = 101850.0
DEFAULT_AGE = 40
DEFAULT_ANTIGUEDAD = 70

# Начальная дата для расчёта months_since_start
DATASET_START = datetime(2015, 2, 1)


def _safe_label_encode(encoder, value: str) -> int:
    """
    Безопасное кодирование через LabelEncoder.

    Если значение неизвестно, пробует закодировать 'nan',
    иначе возвращает 0.
    """
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        if "nan" in encoder.classes_:
            return int(encoder.transform(["nan"])[0])
        return 0


class BankPredictor:
    """
    Предиктор банковских продуктов.

    Получает признаки клиента, выполняет препроцессинг,
    прогоняет через 22 LightGBM-классификатора и возвращает
    рекомендации новых продуктов (исключая уже имеющиеся).

    Attributes:
        store: Хранилище модели с артефактами.
        config: Конфигурация сервиса.
    """

    def __init__(self, store: ModelStore, config: ServiceConfig):
        self._store = store
        self._config = config

    def _preprocess(self, client_data) -> pd.DataFrame:
        """
        Подготовка признаков клиента для модели.

        Args:
            client_data: объект ClientFeatures

        Returns:
            pd.DataFrame с одной строкой и 44 признаками в правильном порядке
        """
        feature_cols = self._store.feature_cols
        product_cols = self._store.product_cols
        label_encoders = self._store.label_encoders
        top20_canal = self._store.top20_canal
        top20_prov = self._store.top20_prov

        features = {}

        # === Численные признаки ===

        # age: clip [18, 100], default 40
        age = client_data.age
        age = max(18, min(100, age)) if age is not None else DEFAULT_AGE
        features["age"] = float(age)

        # renta -> log1p, default по сегменту
        renta = client_data.renta
        if renta is None or renta <= 0:
            seg = client_data.segmento
            renta = DEFAULT_RENTA_BY_SEGMENT.get(seg, DEFAULT_RENTA)
        features["log_renta"] = float(np.log1p(renta))

        # antiguedad: default 70
        antiguedad = client_data.antiguedad
        if antiguedad is None or antiguedad == -999999:
            antiguedad = DEFAULT_ANTIGUEDAD
        features["antiguedad"] = float(antiguedad)

        # ind_nuevo: default 0 (mode)
        features["ind_nuevo"] = float(
            client_data.ind_nuevo if client_data.ind_nuevo is not None else 0
        )

        # indrel: default 1 (mode)
        features["indrel"] = float(
            client_data.indrel if client_data.indrel is not None else 1
        )

        # ind_actividad_cliente: default 1 (mode)
        features["ind_actividad_cliente"] = float(
            client_data.ind_actividad_cliente
            if client_data.ind_actividad_cliente is not None
            else 1
        )

        # cod_prov: default 28 (Madrid, mode)
        features["cod_prov"] = float(
            client_data.cod_prov if client_data.cod_prov is not None else 28
        )

        # === Категориальные признаки (8 малокардинальных) ===

        low_card_fields = {
            "sexo": client_data.sexo,
            "ind_empleado": client_data.ind_empleado,
            "tiprel_1mes": client_data.tiprel_1mes,
            "indresi": client_data.indresi,
            "indext": client_data.indext,
            "indfall": client_data.indfall,
            "segmento": client_data.segmento,
            "indrel_1mes": client_data.indrel_1mes,
        }

        for col_name, value in low_card_fields.items():
            enc = label_encoders[col_name]
            val_str = str(value) if value is not None else "nan"
            features[f"{col_name}_enc"] = _safe_label_encode(enc, val_str)

        # === canal_entrada: top-20 + OTHER ===
        canal = client_data.canal_entrada
        if canal is not None and canal in top20_canal:
            canal_clean = canal
        else:
            canal_clean = "OTHER"
        features["canal_entrada_enc"] = _safe_label_encode(
            label_encoders["canal_entrada"], canal_clean
        )

        # === pais_residencia: бинарный (ES=1) ===
        pais = client_data.pais_residencia
        features["pais_residencia_enc"] = 1 if pais == "ES" else 0

        # === nomprov: top-20 + OTHER ===
        prov = client_data.nomprov
        if prov is not None and prov in top20_prov:
            prov_clean = prov
        else:
            prov_clean = "OTHER"
        features["nomprov_enc"] = _safe_label_encode(
            label_encoders["nomprov"], prov_clean
        )

        # === Предыдущие продукты (22 признака) ===
        prev_products = client_data.prev_products
        n_products = 0
        for col in product_cols:
            val = prev_products.get(col, 0)
            val = 1 if val else 0
            features[f"prev_{col}"] = val
            n_products += val

        # === Агрегатные признаки ===
        features["n_products"] = n_products
        features["product_changes"] = client_data.product_changes or 0

        # === Временные признаки ===
        if client_data.fecha_dato:
            try:
                dt = datetime.strptime(client_data.fecha_dato, "%Y-%m-%d")
            except ValueError:
                dt = datetime.now()
        else:
            dt = datetime.now()

        features["month"] = dt.month
        months_since = (dt.year - DATASET_START.year) * 12 + (dt.month - DATASET_START.month)
        features["months_since_start"] = max(0, months_since)

        # Собираем DataFrame с правильным порядком колонок
        row = {col: features.get(col, 0) for col in feature_cols}
        df = pd.DataFrame([row], columns=feature_cols)

        return df

    def predict(self, client_data) -> dict:
        """
        Получение рекомендаций продуктов для клиента.

        Args:
            client_data: объект ClientFeatures

        Returns:
            dict с ключами:
            - recommendations: list[dict] — рекомендации (product_col, product_name, probability)
            - n_current_products: int — количество текущих продуктов клиента
        """
        start = time.perf_counter()

        try:
            models = self._store.models
            product_cols = self._store.product_cols
            product_names = self._store.product_names

            # Препроцессинг
            X = self._preprocess(client_data)

            # Предсказания: 22 модели -> 22 вероятности
            probabilities = {}
            for col in product_cols:
                model = models[col]
                prob = model.predict_proba(X)[0]
                # predict_proba возвращает [P(0), P(1)] — берём P(1)
                p1 = float(prob[1]) if len(prob) > 1 else float(prob[0])
                probabilities[col] = p1
                PREDICTION_PROBABILITY.observe(p1)

            # Обнуляем вероятности для продуктов, которые клиент уже имеет
            prev_products = client_data.prev_products
            for col in product_cols:
                if prev_products.get(col, 0) == 1:
                    probabilities[col] = 0.0

            # Сортируем по вероятности и берём top-K
            sorted_products = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top_k = client_data.top_k

            recommendations = []
            for col, prob in sorted_products[:top_k]:
                if prob > 0:
                    recommendations.append(
                        {
                            "product_col": col,
                            "product_name": product_names.get(col, col),
                            "probability": round(prob, 6),
                        }
                    )

            n_current = sum(1 for col in product_cols if prev_products.get(col, 0) == 1)

            # --- Метрики ---
            PREDICTION_LATENCY.observe(time.perf_counter() - start)

            if recommendations:
                TOP1_PROBABILITY.observe(recommendations[0]["probability"])

            RECOMMENDATIONS_COUNT.observe(len(recommendations))
            CLIENT_PRODUCTS_COUNT.observe(n_current)
            CLIENT_AGE.observe(client_data.age)
            TOP_K_REQUESTED.observe(top_k)

            for rec in recommendations:
                RECOMMENDED_PRODUCT.labels(product_col=rec["product_col"]).inc()

            return {
                "recommendations": recommendations,
                "n_current_products": n_current,
            }

        except Exception:
            PREDICTION_ERRORS.inc()
            raise
