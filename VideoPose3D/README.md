# 3D Pose Estimation Video Processing Pipeline

This repository contains a Python-based pipeline to process videos for 3D pose estimation. The pipeline utilizes Detectron2 for 2D keypoint detection, prepares custom datasets, and renders the final output with 3D poses overlaid on the original video.

---

## Features
- Detect 2D keypoints using Detectron2.
- Prepare a custom dataset for 3D pose estimation.
- Render 3D pose estimation results over input videos.
- Save processed videos and datasets in organized output directories.

---

## Prerequisites

### 1. Install Dependencies
Ensure the following dependencies are installed:
- Python 3.7+
- [Detectron2](https://github.com/facebookresearch/detectron2)
- PyTorch compiled with Cuda
- NumPy
- OpenCV
- Other dependencies specified in your `requirements.txt`

You can install all dependencies via pip:
```bash
pip install -r requirements.txt
```

### 2. Install Detectron2
Follow the installation guide on the [official Detectron2 repository](https://github.com/facebookresearch/detectron2).

---

## Usage

### Command-Line Arguments
- `video` (required): Path to the input video file.
- `--output-dir` (optional): Directory to save outputs (default: `outputs`).
- `--detectron-cfg` (optional): Path to the Detectron2 configuration file (default: `COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml`).
- `--checkpoint` (optional): Path to the pretrained checkpoint file (default: `pretrained_h36m_detectron_coco.bin`).

### Example Command
```bash
python process_video.py input_video.mp4 --output-dir output
```

### Steps

1. **Detectron2 Inference:**
   The pipeline runs Detectron2 to detect 2D keypoints from the input video.

2. **Prepare Custom Dataset:**
   Converts Detectron2 output into a format suitable for 3D pose estimation.

3. **3D Pose Estimation and Rendering:**
   Processes the custom dataset, applies a pretrained 3D pose estimation model, and renders the final output video.

---

## Output Directory Structure
The pipeline organizes outputs into the following structure:
```
outputs/
├── detectron_output/       # 2D keypoints detected by Detectron2
├── custom_data/            # Prepared custom datasets
├── output_videos/          # Final rendered videos with 3D poses
```

---

## File Descriptions
- `process_video.py`: Main script for processing videos.
- `inference/infer_video_d2.py`: Script to run Detectron2 inference.
- `data/prepare_data_2d_custom.py`: Script to prepare custom datasets.
- `run.py`: Script to perform 3D pose estimation and render videos.

---

## Error Handling
- If the video file does not exist, an error message is displayed.
- Errors encountered during command execution (e.g., Detectron2 inference) will terminate the script with a detailed message.

---

## Example Output
After running the script, the final video with 3D poses is saved to the `output_videos` directory, with the following structure:
```
outputs/
├── detectron_output/
│   ├── frame_0001_keypoints.json
│   ├── ...
├── custom_data/
│   ├── data_2d_custom_video_name.npz
├── output_videos/
│   ├── video_name.mp4
```

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- [Detectron2](https://github.com/facebookresearch/detectron2)
- Pretrained models and scripts used for 3D pose estimation.



## License
This work is licensed under CC BY-NC. See LICENSE for details. Third-party datasets are subject to their respective licenses.
If you use our code/models in your research, please cite our paper:
```
@inproceedings{pavllo:videopose3d:2019,
  title={3D human pose estimation in video with temporal convolutions and semi-supervised training},
  author={Pavllo, Dario and Feichtenhofer, Christoph and Grangier, David and Auli, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
