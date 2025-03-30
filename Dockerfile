FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy language model
RUN python -m spacy download en_core_web_sm

# Install marker_pdf for PDF processing
RUN pip install --no-cache-dir marker-pdf

# Install Jupyter explicitly
RUN pip install --no-cache-dir jupyter notebook

# Create notebooks directory
RUN mkdir -p /app/notebooks

# Copy the package
COPY . .

# Set environment variable for IRIS
ENV IRISINSTALLDIR="/usr"

# Expose port for API (if we add a REST API)
EXPOSE 8000

# Install the package in development mode
RUN pip install -e .

# Default command
CMD ["sleep", "infinity"]