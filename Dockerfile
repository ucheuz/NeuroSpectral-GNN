# CPU image for Linux CI / reviewer reproducibility.
# Local M1 development: prefer the Conda env (see README).

FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir torch-geometric

RUN pip install --no-cache-dir \
    nilearn nibabel scikit-learn pandas numpy matplotlib joblib tensorboard scipy pytest pyyaml

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY requirements-dev.txt ./

ENV PYTHONPATH=/app
CMD ["python", "-m", "pytest", "tests", "-q", "--tb=short"]
