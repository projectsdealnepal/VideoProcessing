import os
import argparse
import subprocess
import shutil
import time

def run_command(command):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command: {command}")
        print(e)
        exit(1)

def process_video(video_path, output_dir, detectron_cfg, checkpoint):
    start_time = time.time()
    
    # Ensure output directories exist
    detectron_output = os.path.join(output_dir, "detectron_output")
    os.makedirs(detectron_output, exist_ok=True)
    
    custom_data_dir = os.path.join(output_dir, "custom_data")
    os.makedirs(custom_data_dir, exist_ok=True)
    
    temp_videos = os.path.join(output_dir, "temp_videos")
    os.makedirs(temp_videos, exist_ok=True)
    # Dedicated folder for processed videos
    output_videos_dir = os.path.join(output_dir, "output_videos")
    os.makedirs(output_videos_dir, exist_ok=True)
 
    # Get the base name of the input video without extension
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a temporary file for the 10 FPS version
    temp_video_path = os.path.join(temp_videos, f"{video_base_name}.mp4")

    # Step 0: Reduce video FPS to 10
    print("Step 0: Reducing video FPS to 10...")
    fps_command = (
        f"ffmpeg -i {video_path} -filter:v fps=10 -c:v libx264 -preset medium "
        f"-crf 23 -c:a copy {temp_video_path}"
    )
    run_command(fps_command)
    os.remove(video_path)

    # Set the final output video path in the output_videos directory
    final_output_video = os.path.join(output_videos_dir, f"{video_base_name}.mp4")

    # Get the file extension from the temp video path
    video_ext = os.path.splitext(temp_video_path)[1].lstrip('.').lower()
    
    # Step 1: Inferring 2D keypoints with Detectron
    print(f"Processing {temp_video_path}")
    print("Step 1: Running Detectron...")
    detectron_command = (
        f"python inference/infer_video_d2.py "
        f"--cfg {detectron_cfg} "
        f"--output-dir {detectron_output} "
        f"--image-ext {video_ext} "
        f"{temp_video_path}"
    )
    run_command(detectron_command)
    
    # Step 2: Creating a custom dataset
    print("Step 2: Preparing custom dataset...")
    npz_file_name = f"data_2d_custom_{video_base_name}.npz"
    source_file_path = os.path.join("data", npz_file_name)
    destination_dir = os.path.join(output_dir, "custom_data")
    destination_file_path = os.path.join(destination_dir, npz_file_name)

    # Ensure the target directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Run the data preparation command
    run_command(
        f"cd data && python prepare_data_2d_custom.py "
        f"-i ../{detectron_output} "
        f"-o {video_base_name}"
    )

    # Move the file after the script has finished
    shutil.move(source_file_path, destination_file_path)
    output_dir = output_dir
    # Step 3: Rendering the custom video and exporting coordinates
    print("Step 3: Rendering video...")
    run_command(
        f"python run.py -d custom -k {video_base_name} "
        f"-arc 3,3,3,3,3 -c checkpoint "
        f"--evaluate {checkpoint} "
        f"--render --viz-subject {os.path.basename(temp_video_path)} "
        f"--viz-action custom --viz-camera 0 "
        f"--viz-video {temp_video_path} "
        f"--viz-output {final_output_video} "
        f"--viz-size 8 "
        f"--output-dir {output_dir} "
    )
    
    # Clean up temporary file
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(detectron_output):
        npz_file = os.path.join(detectron_output, f"{video_base_name}.mp4.npz")
        os.remove(npz_file)
    if os.path.exists(destination_dir):
        npz_file_name_to_delete = os.path.join(destination_dir, npz_file_name)
        os.remove(npz_file_name_to_delete)
    
    
    
    # Record the end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing complete for {video_path}. Final video saved to {final_output_video}")
    print(f"Total processing time: {elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for 3D pose estimation.")
    parser.add_argument(
        "video", type=str, help="Path to the input video file."
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory to save outputs."
    )
    parser.add_argument(
        "--detectron-cfg", type=str, default="COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        help="Path to the Detectron configuration file."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="pretrained_h36m_detectron_coco.bin",
        help="Path to the pretrained checkpoint file."
    )
    args = parser.parse_args()
    
    if not os.path.isfile(args.video):
        print(f"Error: Video file '{args.video}' does not exist.")
        exit(1)
    process_video(args.video, args.output_dir, args.detectron_cfg, args.checkpoint)