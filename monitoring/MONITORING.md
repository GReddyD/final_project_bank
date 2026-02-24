# Мониторинг сервиса рекомендации банковских продуктов

## Обзор

Сервис экспортирует Prometheus-метрики через эндпоинт `GET /metrics` на том же порту (8000). Метрики собираются из кода сервиса с помощью библиотеки `prometheus_client` и охватывают три уровня: технический, ML и бизнес.

Все метрики определены в `service/app/core/metrics.py` и инструментированы в:
- `service/app/main.py` — HTTP middleware (latency, throughput, error rate)
- `service/app/core/predictor.py` — инференс (latency, вероятности, рекомендации)
- `service/app/core/store.py` — загрузка модели (время загрузки, метаданные)

## Подключение к Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "bank-recommender"
    scrape_interval: 15s
    static_configs:
      - targets: ["bank-recommender:8000"]
```

---

## 1. Технические метрики

| Метрика | Тип | Labels | Описание |
|---------|-----|--------|----------|
| `http_requests_total` | Counter | method, endpoint, status_code | Общее количество HTTP-запросов |
| `http_request_duration_seconds` | Histogram | method, endpoint | Время обработки HTTP-запроса (включая сериализацию) |
| `prediction_duration_seconds` | Histogram | — | Время инференса модели (препроцессинг + 22 predict_proba) |
| `prediction_errors_total` | Counter | — | Количество ошибок при инференсе |
| `model_load_time_seconds` | Gauge | — | Время загрузки model.bin при старте сервиса |
| `model_info` | Info | n_models, n_features, n_products, path | Метаданные загруженной модели |

### Пороги для алертов

| Метрика | Условие | Severity | Описание |
|---------|---------|----------|----------|
| `http_request_duration_seconds` | p99 > 1s | warning | Высокая задержка ответа |
| `http_request_duration_seconds` | p99 > 5s | critical | Критическая задержка |
| `prediction_duration_seconds` | p99 > 500ms | warning | Медленный инференс |
| `prediction_errors_total` | rate > 0.1/s | critical | Ошибки инференса |
| `http_requests_total{status_code=~"5.."}` | rate > 1/s | critical | Серверные ошибки |
| `model_load_time_seconds` | > 60s | warning | Медленная загрузка модели |

### Примеры PromQL-запросов

```promql
# Throughput (RPS) за последние 5 минут
rate(http_requests_total[5m])

# p95 latency эндпоинта /predict
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{endpoint="/predict"}[5m]))

# Error rate (%)
rate(http_requests_total{status_code=~"4..|5.."}[5m]) / rate(http_requests_total[5m]) * 100

# Среднее время инференса
rate(prediction_duration_seconds_sum[5m]) / rate(prediction_duration_seconds_count[5m])
```

---

## 2. ML-метрики

| Метрика | Тип | Labels | Описание |
|---------|-----|--------|----------|
| `prediction_probability` | Histogram | — | Распределение P(1) по всем 22 продуктам для каждого запроса |
| `top1_probability` | Histogram | — | Вероятность топ-1 рекомендации (максимальная для клиента) |
| `recommendations_count` | Histogram | — | Количество рекомендаций с P > 0 в ответе |

### Детекция дрифта предсказаний

Основной индикатор дрифта — **сдвиг распределения вероятностей**. Если модель начинает выдавать систематически более высокие или низкие вероятности, это сигнал деградации.

| Метрика | Условие | Severity | Описание |
|---------|---------|----------|----------|
| `top1_probability` | median < 0.0001 | warning | Модель стала менее уверенной |
| `top1_probability` | median > 0.5 | warning | Аномально высокая уверенность |
| `recommendations_count` | avg < 1 | critical | Модель не выдаёт рекомендаций |
| `prediction_probability` | std изменился > 50% | warning | Дрифт распределения вероятностей |

### Примеры PromQL-запросов

```promql
# Медиана вероятности топ-1 рекомендации
histogram_quantile(0.5, rate(top1_probability_bucket[1h]))

# Среднее количество рекомендаций на запрос
rate(recommendations_count_sum[5m]) / rate(recommendations_count_count[5m])

# Доля запросов с 0 рекомендациями
rate(recommendations_count_bucket{le="0"}[5m]) / rate(recommendations_count_count[5m])
```

---

## 3. Бизнес-метрики

| Метрика | Тип | Labels | Описание |
|---------|-----|--------|----------|
| `recommended_product_total` | Counter | product_col | Сколько раз продукт попал в рекомендации |
| `client_products_count` | Histogram | — | Количество текущих продуктов у клиента |
| `client_age` | Histogram | — | Возраст клиентов в запросах |
| `top_k_requested` | Histogram | — | Запрашиваемое количество рекомендаций |

### Покрытие продуктов

Метрика `recommended_product_total` с label `product_col` позволяет отслеживать, какие продукты рекомендуются чаще других. Если какие-то продукты никогда не попадают в рекомендации — это сигнал о проблеме модели или дисбалансе данных.

### Пороги для алертов

| Метрика | Условие | Severity | Описание |
|---------|---------|----------|----------|
| `recommended_product_total` | rate = 0 за 24h для продукта | warning | Продукт не рекомендуется |
| `client_products_count` | avg > 15 | warning | Аномально большой портфель |

### Примеры PromQL-запросов

```promql
# Топ-5 рекомендуемых продуктов
topk(5, rate(recommended_product_total[1h]))

# Среднее количество продуктов у клиентов
rate(client_products_count_sum[1h]) / rate(client_products_count_count[1h])

# Распределение возраста: доля клиентов до 30 лет
rate(client_age_bucket{le="30"}[1h]) / rate(client_age_count[1h])

# Самый популярный top_k
histogram_quantile(0.5, rate(top_k_requested_bucket[1h]))
```

---

## 4. Дашборд Grafana

Рекомендуемая структура дашборда:

### Панель 1: Обзор сервиса
- RPS (requests per second) по эндпоинтам
- Error rate (%)
- p50 / p95 / p99 latency

### Панель 2: Инференс
- Prediction latency (p50 / p95 / p99)
- Prediction errors rate
- Среднее время инференса

### Панель 3: ML-качество
- Распределение top1_probability (heatmap)
- Среднее количество рекомендаций
- Доля запросов с 0 рекомендациями

### Панель 4: Бизнес
- Покрытие продуктов (stacked bar по product_col)
- Распределение портфеля клиентов
- Распределение возраста клиентов

---

## 5. Стратегия дообучения модели

### Триггеры для дообучения

| Индикатор | Порог | Действие |
|-----------|-------|----------|
| Дрифт вероятностей | median(top1_probability) изменился > 30% за неделю | Переобучение |
| Покрытие продуктов | Продукт не рекомендуется > 7 дней | Анализ, возможное переобучение |
| Новые данные | Накоплен 1 месяц новых транзакций | Плановое переобучение |
| Точность на holdout | MAP@7 упал > 20% от baseline | Экстренное переобучение |

### Процесс дообучения

1. Выгрузка нового месяца данных из банковской системы
2. Формирование целевых переменных (переходы 0→1)
3. Переобучение 22 LightGBM-моделей на расширенном train (временное разбиение)
4. Валидация на holdout: MAP@7, F1 macro, AUC-ROC по продуктам
5. Сравнение с текущей моделью в MLflow
6. Замена model.bin при улучшении метрик
7. Перезапуск сервиса (или hot-reload модели)

### Рекомендуемая периодичность

- **Плановое дообучение:** 1 раз в месяц (при поступлении нового среза данных)
- **Экстренное:** при срабатывании алертов дрифта
