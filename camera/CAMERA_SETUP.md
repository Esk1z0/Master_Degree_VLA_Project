# Stereo Camera Calibration & Vision Module

This directory contains the tools, notebooks, and calibration matrices used for **3D Stereo Vision** processing and depth estimation for the SmolVLA-D and SmolVLA-MD models.

## Overview

To enable physical 3D spatial awareness on the low-cost SO-101 robot arm without expensive depth sensors, a custom **stereo camera rig** was assembled. Using OpenCV Semi-Global Block Matching (SGBM) with Weighted Least Squares (WLS) filtering, the system projects depth point clouds into SmolVLA's observation pipeline.

## Contents

- [`stereo_calibration.py`](stereo_calibration.py): Automated stereo camera calibration script computing camera matrices, distortion coefficients, and rectification parameters ($R, T, E, F, Q$).
- [`calibracion_camara.ipynb`](calibracion_camara.ipynb): Interactive Jupyter notebook for calibration image verification, epipolar line checks, and disparity tuning.
- [`split_preview.py`](split_preview.py): Live visual preview utility for dual-lens/side-by-side stereo USB camera feeds.
- [`calib_stereo.npz`](calib_stereo.npz): Production stereo calibration parameters array used by LeRobot during online policy inference.

## Calibration Data

Raw calibration images and sample depth outputs are located under [`../data/camera_calibration/`](../data/camera_calibration/):
- `session_1_20260428/`: Initial calibration capture session.
- `session_2_20260505/`: Intermediate capture session.
- `session_3_active/`: Active stereo calibration image set (`photo_1.jpeg` - `photo_24.jpeg`) used by `stereo_calibration.py`.
- `samples/`: Sample original and stereo depth point cloud outputs.

