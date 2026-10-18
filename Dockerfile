FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models

# torch берём сборку только под процессор: образ меньше на несколько гигабайт
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt

# модели скачиваются при сборке, в работе сеть контейнеру не нужна
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

COPY . .
# индекс КСР (подсказки при «не найдено») собирается из xlsx один раз при сборке
RUN python -c "from app.ksr import Index; Index.load(); print('индекс КСР собран')"

EXPOSE 8000
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CMD ["uvicorn", "app.web:app", "--host", "0.0.0.0", "--port", "8000"]
