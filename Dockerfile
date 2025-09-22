# Use Python 3.9 for compatibility with pandas 1.2.2
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including build tools for pandas compilation
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: pandas version is specifically < 2.0.0 as required by PyRef
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create a non-root user for security
RUN groupadd -r pyref && useradd -r -g pyref pyref
RUN chown -R pyref:pyref /app
USER pyref

# Default command
ENTRYPOINT ["python3", "main.py"]
CMD ["--help"]
