#!/bin/bash

# Load environment variables from .env file
export $(cat .env | xargs)

# Install dependencies inside the container
echo "Installing dependencies using script.sh..."
/app/script.sh

# Run the Python video processing service
echo "Starting video processing service..."
if python video_processing_service.py; then
    echo "Service exited normally."
else
    echo "Service failed. Keeping container alive for debugging..."
    tail -f /dev/null
fi