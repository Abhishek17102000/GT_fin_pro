#!/usr/bin/env bash

# Create data directory if not exists
mkdir -p my_data_directory
cd my_data_directory

# Download VIBE data
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"

# Unzip the downloaded file
unzip my_vibe_data.zip

# Remove the downloaded zip file
rm my_vibe_data.zip

# Go back to the main directory
cd ..

# Move sample video to the main directory
mv my_data_directory/vibe_data/sample_video.mp4 .

# Create directory for PyTorch models
mkdir -p $HOME/.torch/models/

# Move YOLOv3 weights to the PyTorch models directory
mv my_data_directory/vibe_data/yolov3.weights $HOME/.torch/models/
