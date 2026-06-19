FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CURRENCY_MODEL_DIR=/app/models
ENV CURRENCY_DATA_PATH=/app/data/Foreign_Exchange_Rates.xls
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
