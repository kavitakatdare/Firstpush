import bosdyn.client
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.graph_nav import GraphNavClient
import open3d as o3d
import numpy as np

# Load the GraphNav map file
graph_nav_map_file = "path/to/graphnav_map.map"
with open(graph_nav_map_file, "rb") as f:
    map_data = map_pb2.Map()
    map_data.ParseFromString(f.read())

# Extract the point clouds from waypoints
point_cloud_data = []
for waypoint in map_data.waypoints:
    if waypoint.annotations.point_cloud:
        for point in waypoint.annotations.point_cloud.points:
            point_cloud_data.append([point.x, point.y, point.z])

# Convert to numpy array
points = np.array(point_cloud_data)

# Save as .pcd using Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("output_map.pcd", pcd)

print("GraphNav map point cloud saved to output_map.pcd")
