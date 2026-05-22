# ─────────────────────────────────────────────────────────────
# CAMNET-P  Thai Parliamentary Summarization  — Model Submission
# ─────────────────────────────────────────────────────────────

# FROM python:3.11-slim

# ── Option B: GPU (uncomment + comment Option A if GPU server) ─
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── System deps ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── Application code ───────────────────────────────────────────
COPY run.py      /model/run.py
COPY src/        /model/src/

# NOTE: model weights ไม่ได้ COPY เข้า image
# จะถูก mount เข้ามาที่ /model/weights/ ตอน run
# ดูใน docker-compose.yml หัวข้อ volumes

# ── Working directory & entry point ────────────────────────────
WORKDIR /model
CMD ["python3", "run.py"]