#!/bin/bash

# Load environment variables from .env file
export $(cat .env | xargs)

# Build the Docker image
docker build -t videopose3d-local .

# Run the container with environment variables
docker run \
  --name videopose3d-container \
  --runtime=nvidia \
  -e SQS_QUEUE_URL=$SQS_QUEUE_URL \
  -e INPUT_S3_BUCKET=$INPUT_S3_BUCKET \
  -e OUTPUT_S3_BUCKET=$OUTPUT_S3_BUCKET \
  -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  videopose3d-local:latest
