"""
Тестирование микросервиса рекомендации банковских продуктов.

Сценарии тестирования:
1. Проверка состояния сервиса (GET /health)
2. Информация о модели (GET /model/info)
3. Предсказание с минимальными признаками (POST /predict)
4. Предсказание с полным набором признаков (POST /predict)
5. Предсказание с изменённым top_k (POST /predict)
6. Проверка исключения имеющихся продуктов (POST /predict)
7. Валидация ошибок (POST /predict с невалидными данными)

Запуск:
    python -m tests.test_service

Результаты сохраняются в test_service.log
"""

import sys
import logging
from typing import Optional

import requests

# ============================================================================
# Настройка
# ============================================================================

SERVICE_URL = "http://127.0.0.1:8000"

# Настройка логирования в файл и консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_service.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Вспомогательные функции
# ============================================================================

def get_health() -> Optional[dict]:
    """Проверка состояния сервиса."""
    try:
        response = requests.get(f"{SERVICE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при проверке здоровья сервиса: {e}")
        return None


def get_model_info() -> Optional[dict]:
    """Получение информации о модели."""
    try:
        response = requests.get(f"{SERVICE_URL}/model/info")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении информации о модели: {e}")
        return None


def post_predict(payload: dict) -> Optional[dict]:
    """Отправка запроса предсказания."""
    try:
        response = requests.post(f"{SERVICE_URL}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе предсказания: {e}")
        return None


def post_predict_raw(payload: dict) -> requests.Response:
    """Отправка запроса предсказания с возвратом полного Response."""
    return requests.post(f"{SERVICE_URL}/predict", json=payload)


def print_separator(title: str):
    """Печать разделителя с заголовком."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def print_recommendations(result: dict):
    """Красивая печать результатов рекомендаций."""
    logger.info(f"  Текущих продуктов: {result['n_current_products']}")
    logger.info(f"  Рекомендаций: {len(result['recommendations'])}")
    for rec in result["recommendations"]:
        logger.info(
            f"    {rec['product_col']:30s} | {rec['product_name']:25s} | "
            f"P={rec['probability']:.6f}"
        )


# ============================================================================
# Тестовые сценарии
# ============================================================================

def test_health():
    """Тест 0: Проверка состояния сервиса."""
    print_separator("ТЕСТ 0: ПРОВЕРКА СОСТОЯНИЯ СЕРВИСА")

    health = get_health()
    if health is None:
        logger.error("FAIL Сервис недоступен!")
        return False

    logger.info(f"OK Сервис работает")
    logger.info(f"  Статус: {health['status']}")
    logger.info(f"  Модель загружена: {health['model_loaded']}")
    logger.info(f"  Классификаторов: {health['n_models']}")
    logger.info(f"  Признаков: {health['n_features']}")
    logger.info(f"  Продуктов: {health['n_products']}")

    if not health["model_loaded"]:
        logger.error("FAIL Модель не загружена!")
        return False

    return True


def test_model_info():
    """Тест 1: Информация о модели."""
    print_separator("ТЕСТ 1: ИНФОРМАЦИЯ О МОДЕЛИ")

    info = get_model_info()
    if info is None:
        logger.error("FAIL Не удалось получить информацию о модели")
        return False

    logger.info(f"OK Информация о модели получена")
    logger.info(f"  Классификаторов: {info['n_models']}")
    logger.info(f"  Признаков: {len(info['feature_cols'])}")
    logger.info(f"  Продуктов: {len(info['product_cols'])}")
    logger.info(f"  Первые 5 признаков: {info['feature_cols'][:5]}")
    logger.info(f"  Первые 5 продуктов: {info['product_cols'][:5]}")

    # Проверки
    if info["n_models"] != 22:
        logger.error(f"FAIL Ожидалось 22 классификатора, получено {info['n_models']}")
        return False

    if len(info["feature_cols"]) != 44:
        logger.error(f"FAIL Ожидалось 44 признака, получено {len(info['feature_cols'])}")
        return False

    if len(info["product_cols"]) != 22:
        logger.error(f"FAIL Ожидалось 22 продукта, получено {len(info['product_cols'])}")
        return False

    logger.info("OK Структура модели корректна: 22 модели, 44 признака, 22 продукта")
    return True


def test_predict_minimal():
    """
    Тест 2: Предсказание с минимальными признаками.

    Передаём только обязательные поля: age и prev_products.
    Все остальные признаки должны заполниться значениями по умолчанию.
    """
    print_separator("ТЕСТ 2: ПРЕДСКАЗАНИЕ С МИНИМАЛЬНЫМИ ПРИЗНАКАМИ")

    payload = {
        "age": 35,
        "prev_products": {},
    }
    logger.info(f"Запрос: age={payload['age']}, prev_products={{}}")

    result = post_predict(payload)
    if result is None:
        logger.error("FAIL Не удалось получить предсказание")
        return False

    print_recommendations(result)

    # Проверки
    if result["n_current_products"] != 0:
        logger.error(f"FAIL Ожидалось 0 текущих продуктов, получено {result['n_current_products']}")
        return False

    if len(result["recommendations"]) == 0:
        logger.error("FAIL Пустой список рекомендаций")
        return False

    # Вероятности должны быть отсортированы по убыванию
    probs = [r["probability"] for r in result["recommendations"]]
    if probs != sorted(probs, reverse=True):
        logger.error("FAIL Рекомендации не отсортированы по убыванию вероятности")
        return False

    logger.info("OK Предсказание с минимальными признаками работает корректно")
    return True


def test_predict_full():
    """
    Тест 3: Предсказание с полным набором признаков.

    Передаём все возможные поля, включая категориальные.
    """
    print_separator("ТЕСТ 3: ПРЕДСКАЗАНИЕ С ПОЛНЫМ НАБОРОМ ПРИЗНАКОВ")

    payload = {
        "age": 45,
        "renta": 150000,
        "antiguedad": 120,
        "sexo": "H",
        "segmento": "02 - PARTICULARES",
        "canal_entrada": "KHE",
        "pais_residencia": "ES",
        "nomprov": "MADRID",
        "ind_empleado": "N",
        "tiprel_1mes": "A",
        "indresi": "S",
        "indext": "N",
        "indfall": "N",
        "indrel_1mes": "1",
        "ind_nuevo": 0,
        "indrel": 1,
        "ind_actividad_cliente": 1,
        "cod_prov": 28,
        "prev_products": {
            "ind_cco_fin_ult1": 1,
            "ind_recibo_ult1": 1,
        },
        "product_changes": 0,
        "fecha_dato": "2016-05-28",
        "top_k": 7,
    }
    logger.info(f"Запрос: age=45, renta=150000, segmento=02 - PARTICULARES, ...")

    result = post_predict(payload)
    if result is None:
        logger.error("FAIL Не удалось получить предсказание")
        return False

    print_recommendations(result)

    # Проверки
    if result["n_current_products"] != 2:
        logger.error(f"FAIL Ожидалось 2 текущих продукта, получено {result['n_current_products']}")
        return False

    # Проверяем, что имеющиеся продукты не рекомендуются
    rec_products = {r["product_col"] for r in result["recommendations"]}
    if "ind_cco_fin_ult1" in rec_products:
        logger.error("FAIL ind_cco_fin_ult1 не должен рекомендоваться (клиент уже имеет)")
        return False
    if "ind_recibo_ult1" in rec_products:
        logger.error("FAIL ind_recibo_ult1 не должен рекомендоваться (клиент уже имеет)")
        return False

    logger.info("OK Предсказание с полным набором признаков работает корректно")
    logger.info("OK Имеющиеся продукты корректно исключены из рекомендаций")
    return True


def test_predict_top_k():
    """
    Тест 4: Проверка параметра top_k.

    Запрашиваем разное количество рекомендаций.
    """
    print_separator("ТЕСТ 4: ПРОВЕРКА ПАРАМЕТРА TOP_K")

    for top_k in [1, 3, 10]:
        payload = {
            "age": 30,
            "prev_products": {},
            "top_k": top_k,
        }
        logger.info(f"  Запрос с top_k={top_k}")

        result = post_predict(payload)
        if result is None:
            logger.error(f"FAIL Не удалось получить предсказание для top_k={top_k}")
            return False

        n_recs = len(result["recommendations"])
        logger.info(f"  Получено рекомендаций: {n_recs}")

        if n_recs > top_k:
            logger.error(f"FAIL Получено {n_recs} рекомендаций, ожидалось <= {top_k}")
            return False

    logger.info("OK Параметр top_k работает корректно")
    return True


def test_predict_exclude_products():
    """
    Тест 5: Проверка исключения всех имеющихся продуктов.

    Передаём клиента, у которого уже есть много продуктов.
    Ни один из них не должен появиться в рекомендациях.
    """
    print_separator("ТЕСТ 5: ИСКЛЮЧЕНИЕ ИМЕЮЩИХСЯ ПРОДУКТОВ")

    # Клиент с множеством продуктов
    many_products = {
        "ind_cco_fin_ult1": 1,
        "ind_recibo_ult1": 1,
        "ind_nomina_ult1": 1,
        "ind_nom_pens_ult1": 1,
        "ind_tjcr_fin_ult1": 1,
        "ind_ecue_fin_ult1": 1,
        "ind_cno_fin_ult1": 1,
        "ind_dela_fin_ult1": 1,
    }

    payload = {
        "age": 50,
        "prev_products": many_products,
        "top_k": 22,
    }
    logger.info(f"Запрос: age=50, {len(many_products)} имеющихся продуктов, top_k=22")

    result = post_predict(payload)
    if result is None:
        logger.error("FAIL Не удалось получить предсказание")
        return False

    print_recommendations(result)

    # Проверяем, что ни один имеющийся продукт не рекомендуется
    rec_products = {r["product_col"] for r in result["recommendations"]}
    owned_in_recs = rec_products & set(many_products.keys())

    if owned_in_recs:
        logger.error(f"FAIL Имеющиеся продукты в рекомендациях: {owned_in_recs}")
        return False

    if result["n_current_products"] != len(many_products):
        logger.error(
            f"FAIL n_current_products={result['n_current_products']}, "
            f"ожидалось {len(many_products)}"
        )
        return False

    logger.info(f"OK Все {len(many_products)} имеющихся продуктов исключены из рекомендаций")
    return True


def test_predict_validation_errors():
    """
    Тест 6: Проверка валидации входных данных.

    Отправляем невалидные запросы и проверяем, что сервис возвращает 422.
    """
    print_separator("ТЕСТ 6: ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ")

    test_cases = [
        ("Отсутствует age", {"prev_products": {}}, 422),
        ("Отсутствует prev_products", {"age": 30}, 422),
        ("age < 0", {"age": -5, "prev_products": {}}, 422),
        ("age > 150", {"age": 200, "prev_products": {}}, 422),
        ("top_k < 1", {"age": 30, "prev_products": {}, "top_k": 0}, 422),
        ("top_k > 22", {"age": 30, "prev_products": {}, "top_k": 30}, 422),
    ]

    all_passed = True
    for name, payload, expected_status in test_cases:
        try:
            response = post_predict_raw(payload)
            status = response.status_code
            if status == expected_status:
                logger.info(f"  OK {name}: статус {status}")
            else:
                logger.error(f"  FAIL {name}: ожидался {expected_status}, получен {status}")
                all_passed = False
        except requests.exceptions.RequestException as e:
            logger.error(f"  FAIL {name}: исключение {e}")
            all_passed = False

    if all_passed:
        logger.info("OK Валидация входных данных работает корректно")
    return all_passed


def test_predict_unknown_categories():
    """
    Тест 7: Предсказание с неизвестными категориальными значениями.

    Передаём значения, которых не было в обучающей выборке.
    Сервис должен корректно обработать их через safe_label_encode.
    """
    print_separator("ТЕСТ 7: НЕИЗВЕСТНЫЕ КАТЕГОРИАЛЬНЫЕ ЗНАЧЕНИЯ")

    payload = {
        "age": 28,
        "sexo": "X",
        "canal_entrada": "UNKNOWN_CHANNEL",
        "nomprov": "UNKNOWN_PROVINCE",
        "pais_residencia": "FR",
        "segmento": "UNKNOWN",
        "ind_empleado": "Z",
        "prev_products": {"ind_cco_fin_ult1": 1},
    }
    logger.info("Запрос с неизвестными категориями: sexo=X, canal=UNKNOWN_CHANNEL, ...")

    result = post_predict(payload)
    if result is None:
        logger.error("FAIL Сервис упал на неизвестных категориях")
        return False

    print_recommendations(result)
    logger.info("OK Неизвестные категориальные значения обработаны корректно")
    return True


# ============================================================================
# Главная функция
# ============================================================================

def main():
    """Запуск всех тестов."""
    logger.info("")
    logger.info("*" * 70)
    logger.info("*" + " " * 8 + "ТЕСТИРОВАНИЕ СЕРВИСА РЕКОМЕНДАЦИИ БАНКОВСКИХ ПРОДУКТОВ" + " " * 6 + "*")
    logger.info("*" * 70)
    logger.info("")

    # Проверка доступности сервиса
    if not test_health():
        logger.error("")
        logger.error("Сервис недоступен! Убедитесь, что сервис запущен:")
        logger.error("  cd service && MODEL_PATH=../models/model.bin python bank_service.py")
        logger.error("")
        return 1

    # Запуск тестов
    results = []

    results.append(("Тест 1: Информация о модели", test_model_info()))
    results.append(("Тест 2: Предсказание с минимальными признаками", test_predict_minimal()))
    results.append(("Тест 3: Предсказание с полным набором признаков", test_predict_full()))
    results.append(("Тест 4: Проверка параметра top_k", test_predict_top_k()))
    results.append(("Тест 5: Исключение имеющихся продуктов", test_predict_exclude_products()))
    results.append(("Тест 6: Валидация входных данных", test_predict_validation_errors()))
    results.append(("Тест 7: Неизвестные категориальные значения", test_predict_unknown_categories()))

    # Итоги
    print_separator("ИТОГИ ТЕСТИРОВАНИЯ")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"  {status}: {test_name}")

    logger.info("")
    logger.info(f"  Всего тестов: {total}")
    logger.info(f"  Успешных: {passed}")
    logger.info(f"  Неуспешных: {total - passed}")
    logger.info("")

    if passed == total:
        logger.info("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        logger.error("ЕСТЬ НЕУСПЕШНЫЕ ТЕСТЫ")

    logger.info("")
    logger.info("Результаты сохранены в test_service.log")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
