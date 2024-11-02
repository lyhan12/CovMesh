## Overview

Given a scene mesh and images with poses, this code calculates the covisibility between images and extract covisible pairs.

## Prerequisites
- torch
- pytorch3d
- etc

## Installation
```
conda env create --file environment.yml
```

## Usage
```
python main.py --data_dir ~/Dataset/stanford2d3d/area_4 --mesh_path 3d/rgb_dec.ply --output_dir area_4_output --debug_output_dir area_4_output_debug --num_nn 2,3,4
```

## Supported Datasets
- Currently only Standford2D3D dataset is supported.
