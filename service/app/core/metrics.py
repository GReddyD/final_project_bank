"""
Prometheus-метрики сервиса рекомендации банковских продуктов.

Все метрики регистрируются в глобальном реестре prometheus_client
и доступны через эндпоинт GET /metrics.

Группы метрик:
- Технические: latency, throughput, error rate
- ML: распределение вероятностей, количество рекомендаций
- Бизнес: покрытие продуктов, портфель клиентов
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# ============================================================================
# Технические метрики
# ============================================================================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Общее количество HTTP-запросов",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Время обработки HTTP-запроса (секунды)",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Время инференса модели (секунды), без учёта HTTP overhead",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Количество ошибок инференса",
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Время загрузки модели при старте (секунды)",
)

MODEL_INFO = Info(
    "model",
    "Информация о загруженной модели",
)

# ============================================================================
# ML-метрики
# ============================================================================

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Распределение предсказанных вероятностей (P(1) по всем продуктам)",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

TOP1_PROBABILITY = Histogram(
    "top1_probability",
    "Вероятность топ-1 рекомендации (максимальная вероятность для клиента)",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

RECOMMENDATIONS_COUNT = Histogram(
    "recommendations_count",
    "Количество возвращённых рекомендаций (с P > 0)",
    buckets=[0, 1, 2, 3, 5, 7, 10, 15, 22],
)

# ============================================================================
# Бизнес-метрики
# ============================================================================

RECOMMENDED_PRODUCT = Counter(
    "recommended_product_total",
    "Количество раз, когда продукт попал в рекомендации",
    ["product_col"],
)

CLIENT_PRODUCTS_COUNT = Histogram(
    "client_products_count",
    "Количество текущих продуктов у клиента при запросе",
    buckets=[0, 1, 2, 3, 4, 5, 7, 10, 15, 22],
)

CLIENT_AGE = Histogram(
    "client_age",
    "Распределение возраста клиентов в запросах",
    buckets=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100],
)

TOP_K_REQUESTED = Histogram(
    "top_k_requested",
    "Запрашиваемое количество рекомендаций (top_k)",
    buckets=[1, 3, 5, 7, 10, 15, 22],
)
