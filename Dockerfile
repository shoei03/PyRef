FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies (git + build tools for pandas)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (order matters for pandas version)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command
ENTRYPOINT ["python3", "main.py"]
CMD ["--help"]