#!/usr/bin/env bash

# Set the conda environment name
MY_ENV_NAME=my-vibe-environment
echo $MY_ENV_NAME

# Create a conda environment
conda create -n $MY_ENV_NAME python=3.7

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate $MY_ENV_NAME

# Print the Python and pip locations
which python
which pip

# Install required Python packages
pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0

# Install or upgrade the pytube package from the GitHub repository
pip install git+https://github.com/giacaglia/pytube.git --upgrade

# Install dependencies from the requirements.txt file
pip install -r requirements.txt
