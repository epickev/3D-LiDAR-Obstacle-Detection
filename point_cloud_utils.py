# Functions for point cloud manipulation

import open3d as o3d
import numpy as np

# Plane colors
COLORS = [
    [0.5, 0.5, 0.5],      # Gray
    [0.65, 0.4, 0.2],     # Brown
    [1.0, 0.6, 0.8],      # Pink
    [0.6, 0.2, 0.8],      # Purple
    [0.0, 0.0, 1.0],      # Blue
    [0.0, 1.0, 1.0],      # Cyan
    [0.0, 0.0, 0.0],      # Black
    [1.0, 0.5, 0.0],      # Orange
    [1.0, 0.0, 0.0],      # Red
    [1.0, 1.0, 0.0],      # Yellow
    [0.40, 0.25, 0.10],   # Dark Brown
    [0.5, 0.0, 1.0],      # Violet
]

# Remaining point cloud color
GREEN = [0.0, 1.0, 0.0]

def load_point_cloud(filename, voxel_size=0.005):
    """Load and downsample a point cloud."""
    pcd = o3d.io.read_point_cloud(filename)
    print(pcd)
    
    pcd = pcd.voxel_down_sample(voxel_size)
    print("Downsampled:", pcd)

    return pcd

def segment_planes(pcd, num_planes, colors):
    """Segment multiple planes using iterative RANSAC."""

    remaining = pcd
    plane_data = []

    for i in range(num_planes):
        # Run RANSAC to extract the largest planar model in the remaining cloud
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=2000
        )

        # Extract inlier points corresponding to the current plane and apply color
        plane = remaining.select_by_index(inliers)
        plane.paint_uniform_color(colors[i])

        # Normalize the normal vector (a, b, c) to unit length
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)

        # Normalize the plane equation distance constant d (orthogonal distance to world origin)
        d = plane_model[3] / np.linalg.norm(plane_model[:3])

        # Store plane data dictionary in the results list
        plane_data.append({
            "cloud": plane,
            "normal": normal,
            "d": d
        })

        # Remove extracted inlier points to leave non-segmented points for the next iteration
        remaining = remaining.select_by_index(inliers, invert=True)
        
    print("Num of planes detected: ", len(plane_data))
    print("Remaining Points after plane segmentation: ", len(remaining.points))
    return plane_data, remaining

def extract_largest_cluster(pcd, eps=0.05, min_points=10, print_progress=False):
    """Return largest cluster in the point cloud"""

    # DBSCAN Clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
    max_label = labels.max()

    if max_label < 0:
        print("No valid clusters found. Returning original pcd")
        return pcd  

    print(f"Point Cloud has {max_label + 1} clusters")

    # Count occurrences of each cluster label
    counts = np.bincount(labels[labels >= 0])

    # Get index of largest cluster
    largest_cluster_id = np.argmax(counts)

    # Extract indices corresponding to the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_id)[0]

    # pcd with largest cluster
    pcd_cluster = pcd.select_by_index(largest_cluster_indices)

    return pcd_cluster

def filter_horizontal_planes(plane_data, threshold=0.9):
    """Filter plane_data for horizontal planes only"""
    
    horz_plane = []
    for data in plane_data:
        if abs(data['normal'][2]) > threshold: # horizontal plane
            horz_plane.append(data)
    
    print("Num of horz planes detected: ", len(horz_plane))
    return horz_plane

def filter_vertical_planes(plane_data, threshold=0.1):
    """Filter plane_data for vertical planes only"""
    
    vert_plane = []
    for data in plane_data:
        if abs(data['normal'][2]) < threshold: # vertical plane
            vert_plane.append(data)
    print("Num of vert planes detected: ", len(vert_plane))
    return vert_plane

def filter_by_origin_offset(plane_data, lower, upper):
    """Filter plane_data based on normalized d value from plane to origin."""

    obj = []
    for data in plane_data:
        if abs(data['d']) > lower and abs(data['d']) < upper:
            obj.append(data)

    print("Num of planes detected by d: ", len(obj))
    return obj

def filter_planes_below_height(plane_data, max_z):
    """
    Keep planes that have points at or below a maximum z-height threshold.
    Planes strictly above max_z (e.g. high ceiling cabinets or overhead structures) are removed.
    """
    valid_planes = []
    
    for data in plane_data:
        pcd = data['cloud']
        if pcd.is_empty():
            continue
            
        # Get min and max coordinates [min_x, min_y, min_z], [max_x, max_y, max_z]
        min_bound = pcd.get_min_bound()
        
        # If the lowest point of the plane is below our height threshold, keep it
        if min_bound[2] <= max_z:
            valid_planes.append(data)
            
    print(f"Num of obstacle planes: {len(valid_planes)}")
    return valid_planes

def get_distance_to_robot(plane_data, robot_pos):
    """Distance to the actual nearest 3D point in the cluster"""

    points = np.asarray(plane_data['cloud'].points)
    robot_pos = np.asarray(robot_pos)
    
    # Euclidean Distance for all points -> sqrt(dx^2 + dy^2 + dz^2)
    distances = np.linalg.norm(points - robot_pos, axis=1)
    closest_distance = np.min(distances)

    return closest_distance

def visualize(objects, name="Point Cloud View"):
    """Visualize point cloud objects"""
    o3d.visualization.draw_geometries(
        objects,
        window_name=name,
        width=1280,   # Set custom width in pixels
        height=720,  # Set custom height in pixels
        left=300,     # Optional: X position on screen
        top=100,       # Optional: Y position on screen
        zoom=0.54,
        front=[0.5491, 0.8353, 0.0255],
        lookat=[-3.9559, -0.0550, -0.2759],
        up=[-0.0262, -0.0133, 0.9995]
    )
    
