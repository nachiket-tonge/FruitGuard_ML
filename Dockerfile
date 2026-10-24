# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (layer caching)
COPY requirements.txt .

# Install CPU-only PyTorch (smaller image; swap for GPU build if needed)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY model.py .
COPY severity.py .
COPY treatments.py .

# Copy the trained model weights
COPY model_output/ model_output/

# Expose port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
