"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

import cv2

def get_resolution(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {filename}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)
    print(f"Using device: {cfg.MODEL.DEVICE}")
    
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for video_name in im_list:
        out_name = os.path.join(
                args.output_dir, os.path.basename(video_name)
            )
        print('Processing {}'.format(video_name))

        boxes = []
        segments = []
        keypoints = []
        keypoint_scores = []  # New list for keypoint confidence scores

        for frame_i, im in enumerate(read_video(video_name)):
            t = time.time()
            outputs = predictor(im)['instances'].to('cpu')
            
            print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

            # Initialize empty arrays with consistent shapes
            bbox_tensor = np.zeros((0, 5))  # 4 coords + 1 score
            kps = np.zeros((0, 3, 17))  # Standard COCO format: N x 3 x 17
            kps_scores = np.zeros((0, 17))  # New array for keypoint scores: N x 17

            if outputs.has('pred_boxes'):
                bbox_tensor_tmp = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor_tmp) > 0:
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor_tmp, scores), axis=1)
                    
                    if outputs.has('pred_keypoints'):
                        kps = outputs.pred_keypoints.numpy()
                        kps_xy = kps[:, :, :2]
                        kps_prob = kps[:, :, 2:3]
                        kps_scores = kps[:, :, 2]  # Extract confidence scores: N x 17
                        kps_logit = np.zeros_like(kps_prob)  # Dummy logits
                        kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
                        kps = kps.transpose(0, 2, 1)
            
            # Ensure consistent structure
            cls_boxes = [np.zeros((0, 5)), bbox_tensor]  # Empty array for first class
            cls_keyps = [np.zeros((0, 3, 17)), kps]  # Empty array for first class
            cls_kps_scores = [np.zeros((0, 17)), kps_scores]  # Empty array for first class
            
            boxes.append(cls_boxes)
            segments.append(None)
            keypoints.append(cls_keyps)
            keypoint_scores.append(cls_kps_scores)  # Add scores for this frame

        # Convert lists to arrays with consistent shapes
        boxes = np.array(boxes, dtype=object)
        segments = np.array(segments, dtype=object)
        keypoints = np.array(keypoints, dtype=object)
        keypoint_scores = np.array(keypoint_scores, dtype=object)  # Convert scores to array
        
        # Video resolution
        metadata = {
            'w': im.shape[1],
            'h': im.shape[0],
        }
        
        # Save with the additional keypoint_scores array
        np.savez_compressed(out_name, 
                          boxes=boxes, 
                          segments=segments, 
                          keypoints=keypoints, 
                          keypoint_scores=keypoint_scores,  # New field
                          metadata=metadata)

if __name__ == '__main__':
    setup_logger()
    args = parse_args()
    main(args)