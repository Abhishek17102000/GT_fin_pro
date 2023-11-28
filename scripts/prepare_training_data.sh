#!/usr/bin/env bash

# Create directory for VIBE database
mkdir -p ./data/my_vibe_database

# Add current directory to Python path
export PYTHONPATH="./:$PYTHONPATH"

# AMASS
python lib/data_utils/amass_utils.py --directory ./data/amass

# InstaVariety
# Skip this if you already downloaded the preprocessed file
python lib/data_utils/insta_utils.py --directory ./data/insta_variety

# 3DPW
python lib/data_utils/threedpw_utils.py --directory ./data/3dpw

# MPI-INF-3D-HP
python lib/data_utils/mpii3d_utils.py --directory ./data/mpi_inf_3dhp

# PoseTrack
python lib/data_utils/posetrack_utils.py --directory ./data/posetrack

# PennAction
python lib/data_utils/penn_action_utils.py --directory ./data/penn_action
