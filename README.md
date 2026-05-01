# Human Activity Recognition — AI Lab Research

A research project focused on building a machine learning pipeline to classify human physical activities from raw inertial sensor data.

## Overview

This project uses the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which contains motion recordings from a smartphone worn on the waist while subjects performed six activities (walking, walking upstairs, walking downstairs, sitting, standing, and laying).

The pipeline processes raw 6-axis IMU (Inertial Measurement Unit) signals — three axes of body acceleration and three axes of gyroscope readings — and prepares them for activity classification using deep learning.

## Dataset

| Signal | Description |
|---|---|
| `body_acc_x/y/z` | Body linear acceleration along each axis |
| `body_gyro_x/y/z` | Body angular velocity along each axis |

Each sample contains 128 time steps recorded at 50 Hz, producing an input tensor of shape `(samples, 128 timesteps, 6 features)`.

## Pipeline

1. **Data Loading** — Raw `.txt` signal files are read into a NumPy array and reshaped into `(sample, timestep, feature)` format.
2. **Normalization** — Each sensor channel is standardized (zero mean, unit variance) across all samples and timesteps to put every axis on a comparable scale before model training.

## Environment

The notebook is developed and run on [Kaggle](https://www.kaggle.com/) using a Python 3 kernel with standard data science libraries (`numpy`, `pandas`).

## Getting Started

1. Upload the UCI HAR Dataset to your Kaggle input directory (or update `raw_data_path` to your local path).
2. Open `Main` in Kaggle Notebooks (or Jupyter).
3. Run all cells in order.
