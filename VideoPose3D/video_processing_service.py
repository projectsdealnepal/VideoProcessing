import os
import json
import boto3
import logging
from botocore.exceptions import ClientError
import subprocess
import tempfile
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessingService:
    def __init__(self):
        # InitializeREGION AWS clients
        # print(os.environ.get("AWS_ACCESS_KEY_ID"))
        # print(os.environ.get("AWS_SECRET_ACCESS_KEY"))
        self.sqs = boto3.client('sqs', region_name=os.environ['AWS_DEFAULT_REGION'],
                                        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        self.s3 = boto3.client('s3',    region_name=os.environ['AWS_DEFAULT_REGION'],
                                        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        
        # Get environment variables
        self.queue_url = os.environ['SQS_QUEUE_URL']
        self.input_bucket = os.environ['INPUT_S3_BUCKET']
        self.output_bucket = os.environ['OUTPUT_S3_BUCKET']
        
        # Add the current directory to Python path to import process_video
        current_dir = Path(__file__).parent.absolute()
        sys.path.append(str(current_dir))
        
        # Import process_video module
        try:
            import process_video_10fps
            self.process_video_module = process_video_10fps
            logger.info("Successfully imported process_video module")
        except ImportError as e:
            logger.error(f"Failed to import process_video module: {e}")
            raise

    def download_from_s3(self, bucket, key, local_path):
        """Download file from S3"""
        try:
            self.s3.download_file(bucket, key, local_path)
            logger.info(f"Downloaded {key} from {bucket} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            return False

    def upload_to_s3(self, local_path, bucket, key):
        """Upload file to S3"""
        try:
            self.s3.upload_file(local_path, bucket, key)
            logger.info(f"Uploaded {local_path} to {bucket}/{key}")
            os.remove(local_path)
            return True
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            return False

    def process_video(self, input_path, output_dir):
        """Process video using VideoPose3D"""
        try:
            # Instead of using subprocess, we'll use the imported module directly
            sys.argv = [
                'process_video.py',
                input_path,
                '--output-dir', output_dir
            ]
            print(input_path, output_dir)
            # Call the main function of process_video
            self.process_video_module.process_video(input_path, output_dir, "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml","pretrained_h36m_detectron_coco.bin")
            logger.info(f"Successfully processed video: {input_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False

    def handle_message(self, message):
        """Handle incoming SQS message"""
        try:
            message_body = json.loads(message['Body'])
            video_key = message_body['video_key']
            
            # Create temporary directories for processing
            os.makedirs('inputs', exist_ok=True)
            input_path = os.path.join('inputs',video_key)
            output_dir = os.path.join('output')
            # print("inputinputinput:",input_path)
            # print("outputoutputoutput",output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Download video from S3
            if not self.download_from_s3(self.input_bucket, video_key, input_path):
                return False

            # Process video
            if not self.process_video(input_path, output_dir):
                return False

            # Upload results to S3
            output_prefix = f"{os.path.splitext(video_key)[0]}"
            for root, _, files in os.walk(output_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    s3_key = f"{output_prefix}/{file}"
                    if not self.upload_to_s3(local_file_path, self.output_bucket, s3_key):
                        return False

            return True
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False

    def run(self):
        """Main service loop"""
        logger.info("Starting video processing service...")
        while True:
            try:
                # Receive messages from SQS
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20
                )

                messages = response.get('Messages', [])
                
                for message in messages:
                    receipt_handle = message['ReceiptHandle']
                    
                    if self.handle_message(message):
                        # Delete message if processed successfully
                        self.sqs.delete_message(
                            QueueUrl=self.queue_url,
                            ReceiptHandle=receipt_handle
                        )
                        logger.info("Successfully processed and deleted message")
                    else:
                        logger.error("Failed to process message")

            except Exception as e:
                logger.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    service = VideoProcessingService()
    service.run()
