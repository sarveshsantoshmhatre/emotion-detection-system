FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPFACE_HOME=/app/.deepface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install only the standalone website's runtime dependencies. Streamlit and
# test tooling stay available for local development but are not needed here.
RUN pip install --upgrade pip && pip install \
    "deepface>=0.0.93" \
    "tf-keras>=2.15" \
    "opencv-python-headless>=4.9" \
    "numpy>=1.26,<3" \
    "fastapi>=0.115,<1" \
    "uvicorn[standard]>=0.30,<1"

COPY . .
RUN mkdir -p /app/.deepface/weights

EXPOSE 8000
CMD ["sh", "-c", "uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
