# Use an official NVIDIA CUDA runtime as a parent image (with Python 3.9)
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

# Copy the requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies like detectron2 and boto3
RUN pip install git+https://github.com/facebookresearch/detectron2.git
RUN pip install --no-cache-dir boto3

# Copy the rest of the application
COPY . .

# Run the service
CMD ["python", "video_processing_service.py"]
