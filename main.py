"""
3D Point Cloud Obstacle Detection and Distance Estimation
--------------------------------------------------------------
1. Loads and voxel-downsamples a 3D scene point cloud (.ply).
2. Performs iterative RANSAC plane segmentation and DBSCAN outlier removal.
3. Filters candidates for vertical obstacle surfaces located below the robot's height threshold.
4. Calculates minimum Euclidean distance from each valid obstacle to the robot frame.
"""

import open3d as o3d
import numpy as np
import point_cloud_utils as pc_utils
import copy

# Load point cloud and downsample
filename = "sart-tilman_kitchen_5M.ply"
pcd = pc_utils.load_point_cloud(filename, voxel_size=.007)

# Select number of planes to segement
NUM_PLANES = 12

# Robot coordinates w.r.t world frame
robot_x = -3.5
robot_y = 1.0 
robot_z = -1.2 # Max height of the robot
robot_pos = [robot_x, robot_y, robot_z]

robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=.5,               
    origin=robot_pos
)

# Iteratively segment planes with RANSAC.
# Returns plane data (plane cloud, plane normal, and normalized d)
plane_data, remaining = pc_utils.segment_planes(pcd, NUM_PLANES, pc_utils.COLORS)

raw_plane_data = copy.deepcopy(plane_data)

# DBSCAN to get largest plane cluster (remove outliers)
for p in plane_data:
    pcd_cluster = pc_utils.extract_largest_cluster(p['cloud'], eps=0.05, min_points=10, print_progress=False)
    p["cloud"] = pcd_cluster

# Color unsegmented pcd as green
remaining.paint_uniform_color(pc_utils.GREEN)

# Filter by vertical plane
vert_planes_data = pc_utils.filter_vertical_planes(plane_data)

# Filter by robot's max height
obstacle_planes = pc_utils.filter_planes_below_height(vert_planes_data, max_z=robot_z)

# Visulization of Data
pc_utils.visualize([p["cloud"] for p in raw_plane_data], "Raw Plane Segmentation")
pc_utils.visualize([p["cloud"] for p in plane_data], "Cleaned Plane Segmentation")
pc_utils.visualize([remaining], "Leftover Point Cloud")
pc_utils.visualize([p["cloud"] for p in vert_planes_data] + [robot_frame], "Vertical Plane")
pc_utils.visualize([p["cloud"] for p in obstacle_planes] + [robot_frame], "Obstacle Plane")

# Visualization of obstacle plane detected and print distance
for planes in obstacle_planes:
    distance = pc_utils.get_distance_to_robot(planes, robot_pos)
    print(f"Distance: {round(distance, 2)} meters")
    pc_utils.visualize([planes["cloud"]] + [robot_frame])
