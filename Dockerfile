FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p artifacts/checkpoints artifacts/features

# Expose the port (Hugging Face uses 7860)
ENV PORT=7860
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
