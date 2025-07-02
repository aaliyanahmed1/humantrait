# Use Python 3.8 slim image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for OpenCV and other libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to improve Docker layer caching
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app code
COPY fairface.pt . 
COPY app.py . 

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
