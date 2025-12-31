# Use specific Python version for reproducibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies
# libgl1 and libglib2.0 are often needed for OpenCV/image libraries if used
# tesseract-ocr is needed for OCR if pytesseract is used
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for Tesseract data if needed or other temp dirs
RUN mkdir -p /app/static/images

# Create user to run app (security best practice, required by some platforms)
RUN useradd -m -u 1000 user
USER user

# Expose port (7860 is default for HF Spaces)
EXPOSE 7860

# Command to run the application
# Use 0.0.0.0 to listen on all interfaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
