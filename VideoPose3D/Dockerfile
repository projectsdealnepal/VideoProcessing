# Use your existing base image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip
# Copy necessary files into the container
COPY . . 
# Ensure bash.sh and script.sh are executable
RUN chmod +x /app/bash.sh /app/script.sh
RUN pip install -r requirements.txt

# Set the bash.sh as the container's entry point
ENTRYPOINT ["/app/bash.sh"]