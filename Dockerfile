FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK
RUN python - <<'PY'
import nltk, os
path="/usr/local/nltk_data"
os.makedirs(path, exist_ok=True)
nltk.download("stopwords", download_dir=path)
nltk.download("punkt", download_dir=path)
PY
ENV NLTK_DATA=/usr/local/nltk_data

COPY . /app
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]