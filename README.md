# 3D LiDAR Obstacle Detection

## Demo

[![3D LiDAR Obstacle Detection Demo](https://img.youtube.com/vi/l_D-Qne8VbU/maxresdefault.jpg)](https://www.youtube.com/watch?v=l_D-Qne8VbU)

## Overview

Developed a 3D LiDAR point-cloud pipeline to detect vertical obstacles and estimate their distance from a robot frame. The pipeline performs voxel downsampling, iterative RANSAC plane segmentation, DBSCAN outlier removal, and geometric filtering based on plane orientation and obstacle height.

**Architecture:**  
3D Point Cloud → Voxel Downsampling → RANSAC Plane Segmentation → DBSCAN Filtering → Vertical Plane Filtering → Height Filtering → Obstacle Detection → Distance Estimation

## Key Features & Results
- Reduced a ~5M-point scene to ~1M points using 7 mm voxel downsampling
- Segmented up to 12 planar surfaces using iterative RANSAC
- Removed plane-segmentation outliers using DBSCAN clustering
- Identified candidate obstacles using surface orientation and robot-height constraints
- Estimated minimum obstacle distance relative to the robot frame

