#!/bin/bash

set -e
# Install PyTorch and dependencies
echo "Installing PyTorch and related dependencies..."
#pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 torchvision==0.20.1+cu118 \
#    --index-url https://download.pytorch.org/whl/cu118

#pip3 install torch torchvision torchaudio
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124


# Install Detectron2

echo "Installing Detectron2..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


# Install boto3 for AWS interactions
pip install boto3
pip install python-dotenv
conda install conda-forge::ffmpeg
pip install opencv-python
pip install scipy 
