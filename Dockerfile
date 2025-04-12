# Gunakan image Python resmi sebagai base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install sistem dependensi untuk OpenCV, face_recognition, dan MySQL
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    build-essential \
    cmake \
    libjpeg-dev \
    libpng-dev \
    default-libmysqlclient-dev \
    pkg-config \
    python3-dev \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh proyek ke container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=Absen.settings

# Expose port (default Django)
EXPOSE 8000

# Jalankan migrasi dan start server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]