ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}
ARG CAMNET_LLM_MODEL_NAME=final_merged

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CAMNET_MODEL_DIR=/model/weights \
    CAMNET_EMBED_MODEL_PATH=/model/weights/Qwen3-Embedding-8B \
    CAMNET_LLM_MODEL_NAME=${CAMNET_LLM_MODEL_NAME} \
    CAMNET_TEST_PATH=/model/test/test_set.json \
    CAMNET_OUTPUT_DIR=/result \
    CAMNET_PROGRESS_LIB=/benchmark_lib/progress \
    CAMNET_STARTUP_SLEEP_SECONDS=10

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        build-essential; \
    ln -sf /usr/bin/python3 /usr/bin/python; \
    python3 -m pip install --upgrade pip; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --force-reinstall \
        torch==2.6.0

RUN mkdir -p /model/weights /model/test /result
COPY weight/ /model/weights/
RUN test -d /model/weights/Qwen3-Embedding-8B
RUN test -d "/model/weights/${CAMNET_LLM_MODEL_NAME}"

COPY run.py /model/run.py
COPY src/ /model/src/

WORKDIR /model
CMD ["python3", "run.py"]
