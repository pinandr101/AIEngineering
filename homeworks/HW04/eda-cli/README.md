# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports --max-hist-columns 5 --top-k-categories 3 --title "Кастомный отчёт" --min-missing-share 0.01
```

Параметры:
- `--out-dir` - Каталог для отчёт (по умолчанию `reports`);
- `sep` - Разделителть в csv (по умолчанию `,`);
- `encoding` - Кодировка файла (по умолчанию `utf-8`);
- `max_hist_columns` - Максимум числовых колонок для гистограмм (по умолчанию 6);
- `top_k_categories` - Сколько top-значений выводить для категориальных признаков (по умолчанию 5);
- `title` - Заголовок отчёта (по умолчанию `EDA-отчёт`);
- `min_missing_share` - Порог допустимой доли пропусков (по умолчанию 0.05).

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

### Вывод первых n строк датасета

```bash
uv run eda-cli head data/example.csv --n 10
```

Параметры:
- `--n` - Количество выводимых строк
- `--sep` - Разделитель в csv файле
- `--encoding` - Кодировка csv файла

### Тесты

```bash
uv run pytest -q
```

# S04 - API: HTTP-сервис оценки качества датасетов

Небольшой HTTP-сервис на базе fastAPI для оценки качества CSV-файлов, построенный поверх EDA-ядра, описанного выше.

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S04):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект в окружение.

## Запуск сервиса

Запуск осуществляется с помощью следующей команды

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После введённой команды, сервер будет доступен по ссылке http://localhost:8000.

Также станет доступна интерактивная документация Swagger UI: http://localhost:8000/docs.

### Эндпоинт GET /health

Простейший health-check сервиса.

Пример запроса:

```bash
curl -X 'GET' \
  'http://localhost:8000/health' \
  -H 'accept: application/json'
```

Пример ответа:

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

### Эндпоинт POST /quality

Эндпоинт-заглушка, который принимает агрегированные признаки датасета и возвращает эвристическую оценку качества.

Пример запроса:

```bash
curl -X 'POST' \
  'http://localhost:8000/quality' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "n_rows": 100,
  "n_cols": 10,
  "max_missing_share": 0.05,
  "numeric_cols": 5,
  "categorical_cols": 3
}'
```

Пример ответа:
```json
{
  "ok_for_model": true,
  "quality_score": 0.75,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 0.002700020559132099,
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 100,
    "n_cols": 10
  }
}
```

### Эндпоинт /quality-from-csv

Эндпоинт, который принимает CSV-файл, запускает EDA-ядро (summarize_dataset + missing_table + compute_quality_flags) и возвращает оценку качества данных.

Параметры:
- file: csv-файл

Пример запроса:

```bash
curl -X 'POST' \
  'http://localhost:8000/quality-from-csv' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@example.csv;type=text/csv'
```

Пример ответа:
```json
{
  "ok_for_model": false,
  "quality_score": 0.4444444444444445,
  "message": "CSV требует доработки перед обучением модели (по текущим эвристикам).",
  "latency_ms": 7.173900026828051,
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": true,
    "has_many_zero_values": false,
    "has_suspicious_id_duplicates": true
  },
  "dataset_shape": {
    "n_rows": 36,
    "n_cols": 14
  }
}
```

### Эндпоинт /quality-flags-from-csv

Эндпоинт, который принимает CSV-файл, запускает EDA-ядро (summarize_dataset + missing_table + compute_quality_flags) и возвращает JSON, который содержит словарь флагов качества и их значений.

Параметры:
- file: csv-файл

Пример запроса:

```bash
curl -X 'POST' \
  'http://localhost:8000/quality-flags-from-csv' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@example.csv;type=text/csv'
```

Пример ответа:

```json
{
  "too_few_rows": true,
  "too_many_columns": false,
  "max_missing_share": 0.05555555555555555,
  "too_many_missing": false,
  "has_constant_columns": false,
  "has_high_cardinality_categoricals": true,
  "has_many_zero_values": false,
  "has_suspicious_id_duplicates": true,
  "quality_score": 0.4444444444444445
}
```

### Эндпоинт /head

HTTP-обёртка над CLI-командой head, принимающая CSV файл и количество n необходимых для возврата первых строк датасета (по умолчанию 5). Возвращает первые n строк в формате JSON в случае прохождения выборкой минимальных проверок на качество или же сообщение-предупреждение, если проверки пройдены не были.

Параметры:
- file: csv-файл
- n: количество необходимых для возврата первых строк датасета

Пример запроса:

```bash
curl -X 'POST' \
  'http://localhost:8000/head?n=5' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@example.csv;type=text/csv'
```

Пример ответа:
```json
{
  "data": [
    {
      "user_id": 1001,
      "country": "RU",
      "city": "Moscow",
      "device": "Desktop",
      "channel": "Organic",
      "sessions_last_30d": 25,
      "avg_session_duration_min": 12.5,
      "pages_per_session": 6.2,
      "purchases_last_30d": 3,
      "revenue_last_30d": 4500,
      "churned": 0,
      "signup_year": 2021,
      "plan": "Pro",
      "n_support_tickets": 1
    },
    {
      "user_id": 1002,
      "country": "RU",
      "city": "Saint Petersburg",
      "device": "Mobile",
      "channel": "Ads",
      "sessions_last_30d": 5,
      "avg_session_duration_min": 4.2,
      "pages_per_session": 3.1,
      "purchases_last_30d": 0,
      "revenue_last_30d": 0,
      "churned": 1,
      "signup_year": 2020,
      "plan": "Free",
      "n_support_tickets": 0
    },
    {
      "user_id": 1003,
      "country": "KZ",
      "city": "Almaty",
      "device": "Desktop",
      "channel": "Referral",
      "sessions_last_30d": 12,
      "avg_session_duration_min": 8.3,
      "pages_per_session": 5,
      "purchases_last_30d": 1,
      "revenue_last_30d": 1200,
      "churned": 0,
      "signup_year": 2022,
      "plan": "Basic",
      "n_support_tickets": 2
    },
    {
      "user_id": 1004,
      "country": "BY",
      "city": "Minsk",
      "device": "Mobile",
      "channel": "Organic",
      "sessions_last_30d": 18,
      "avg_session_duration_min": 9.1,
      "pages_per_session": 4.7,
      "purchases_last_30d": 2,
      "revenue_last_30d": 2600.5,
      "churned": 0,
      "signup_year": 2021,
      "plan": "Pro",
      "n_support_tickets": 0
    },
    {
      "user_id": 1005,
      "country": "RU",
      "city": "Moscow",
      "device": "Tablet",
      "channel": "Email",
      "sessions_last_30d": 3,
      "avg_session_duration_min": 3,
      "pages_per_session": 2.2,
      "purchases_last_30d": 0,
      "revenue_last_30d": 0,
      "churned": 1,
      "signup_year": 2019,
      "plan": "Free",
      "n_support_tickets": 1
    }
  ]
}