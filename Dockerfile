# --------------------------
# Base Image
# --------------------------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --------------------------
# Install system dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        && rm -rf /var/lib/apt/lists/*

# --------------------------
# Prepare SageMaker directories
# --------------------------
RUN mkdir -p /opt/ml/code \
    /opt/ml/input/data \
    /opt/ml/output \
    /opt/ml/model

WORKDIR /opt/ml/code

# --------------------------
# Copy requirements first (cache optimization)
# --------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# Copy full project into container
# --------------------------
COPY src/ src/
COPY sm_jobs/ sm_jobs/
COPY utils/ utils/
COPY config/ config/
COPY entrypoint.py entrypoint.py

# --------------------------
# Add PYTHONPATH
# --------------------------
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

# --------------------------
# No ENTRYPOINT — SageMaker manages scripts
# --------------------------
ENTRYPOINT []
CMD ["python"]
