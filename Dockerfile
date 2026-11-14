FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Der Code wird per Volume nach /app gemountet (siehe docker-compose.yml),
# damit Änderungen ohne Neubau wirken.
ENTRYPOINT ["python", "-m", "pcai"]
CMD ["--help"]
