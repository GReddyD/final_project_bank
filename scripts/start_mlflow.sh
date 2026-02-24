#!/usr/bin/env bash
# =============================================================================
# start_mlflow.sh — Запуск MLflow Tracking Server с хранилищем артефактов
# =============================================================================
#
# Использование:
#   bash scripts/start_mlflow.sh           # запуск с параметрами по умолчанию
#   bash scripts/start_mlflow.sh --port 6000
#   bash scripts/start_mlflow.sh --host 0.0.0.0 --port 5050
#
# Параметры (переменные окружения или аргументы):
#   MLFLOW_PORT  / --port   — порт сервера (по умолчанию 5000)
#   MLFLOW_HOST  / --host   — хост (по умолчанию 127.0.0.1)
#
# После запуска:
#   - UI:        http://<host>:<port>
#   - Tracking:  подключение из ноутбуков:
#       import mlflow
#       mlflow.set_tracking_uri("http://127.0.0.1:5000")
# =============================================================================

set -euo pipefail

# ----- Определение корня проекта -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----- Настройки по умолчанию -----
HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5000}"

# ----- Разбор аргументов командной строки -----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            head -n 17 "$0" | tail -n +3 | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Неизвестный аргумент: $1 (используйте --help)"
            exit 1
            ;;
    esac
done

# ----- Хранилища -----
BACKEND_STORE="sqlite:///${PROJECT_DIR}/mlflow.db"
ARTIFACT_STORE="${PROJECT_DIR}/mlartifacts"

# ----- Проверка наличия mlflow -----
if ! command -v mlflow &>/dev/null; then
    echo "[INFO] mlflow не найден, устанавливаю..."
    pip install mlflow
    echo "[INFO] mlflow установлен: $(mlflow --version)"
fi

# ----- Создание директории артефактов -----
mkdir -p "${ARTIFACT_STORE}"

# ----- Проверка занятости порта -----
if lsof -i :"${PORT}" -sTCP:LISTEN &>/dev/null; then
    echo "[WARN] Порт ${PORT} уже занят."
    echo "       Возможно, MLflow уже запущен: http://${HOST}:${PORT}"
    echo "       Остановите процесс или укажите другой порт: --port <NUMBER>"
    exit 1
fi

# ----- Вывод конфигурации -----
echo "============================================="
echo "  MLflow Tracking Server"
echo "============================================="
echo "  UI:              http://${HOST}:${PORT}"
echo "  Backend store:   ${BACKEND_STORE}"
echo "  Artifact store:  ${ARTIFACT_STORE}"
echo "  PID-файл:        ${PROJECT_DIR}/.mlflow.pid"
echo "============================================="
echo ""
echo "  Подключение из ноутбука:"
echo "    import mlflow"
echo "    mlflow.set_tracking_uri(\"http://${HOST}:${PORT}\")"
echo ""
echo "  Остановка сервера:"
echo "    kill \$(cat ${PROJECT_DIR}/.mlflow.pid)"
echo "============================================="
echo ""

# ----- Запуск MLflow -----
mlflow server \
    --host "${HOST}" \
    --port "${PORT}" \
    --backend-store-uri "${BACKEND_STORE}" \
    --default-artifact-root "${ARTIFACT_STORE}" &

MLFLOW_PID=$!
echo "${MLFLOW_PID}" > "${PROJECT_DIR}/.mlflow.pid"

echo "[OK] MLflow запущен (PID: ${MLFLOW_PID})"
echo "[OK] UI доступен: http://${HOST}:${PORT}"

# ----- Ожидание готовности -----
for i in $(seq 1 10); do
    if curl -s "http://${HOST}:${PORT}/health" &>/dev/null || \
       curl -s "http://${HOST}:${PORT}/api/2.0/mlflow/experiments/search" &>/dev/null; then
        echo "[OK] Сервер готов к работе."
        exit 0
    fi
    sleep 1
done

echo "[OK] Сервер запущен. Если UI не открывается, подождите несколько секунд."
