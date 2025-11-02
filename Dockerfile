# Lightweight CPU PyTorch base image
FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
