FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Cache pip packages
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# copy app
COPY . .

EXPOSE 5000

# server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
