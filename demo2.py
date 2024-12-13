import os
import numpy as np
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.math_helpers import SE3Pose

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import open3d as o3d
import asyncio

class GraphNavToRviz(Node):
    def __init__(self, map_path):
        super().__init__('graph_nav_to_rviz')
        self.marker_pub = self.create_publisher(MarkerArray, 'graph_nav_map', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, 'graph_nav_cloud', 10)

        self.get_logger().info(f"Loading map from {map_path}")
        try:
            self.graph, self.waypoints, self.snapshots, self.edges = self.load_map(map_path)
            self.visualize_graph()
            self.convert_map_to_pcl(map_path)
        except Exception as e:
            self.get_logger().error(f"Error during initialization: {e}")

    def load_map(self, path):
        """Load the map data from the provided directory."""
        graph = map_pb2.Graph()

        # Load the graph
        graph_file = os.path.join(path, 'graph')
        if not os.path.exists(graph_file):
            self.get_logger().error(f"Graph file not found at {graph_file}")
            self.destroy_node()
            return

        with open(graph_file, 'rb') as gf:
            graph.ParseFromString(gf.read())
        self.get_logger().info(f"Loaded graph with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges.")

        # Load waypoint snapshots
        snapshots = {}
        for waypoint in graph.waypoints:
            if waypoint.snapshot_id:
                snapshot_file = os.path.join(path, 'waypoint_snapshots', waypoint.snapshot_id)
                if os.path.exists(snapshot_file):
                    with open(snapshot_file, 'rb') as sf:
                        snapshot = map_pb2.WaypointSnapshot()
                        snapshot.ParseFromString(sf.read())
                        snapshots[snapshot.id] = snapshot
        self.get_logger().info(f"Loaded {len(snapshots)} waypoint snapshots.")

        # Load edge snapshots
        edges = {}
        for edge in graph.edges:
            if edge.snapshot_id:
                edge_file = os.path.join(path, 'edge_snapshots', edge.snapshot_id)
                if os.path.exists(edge_file):
                    with open(edge_file, 'rb') as ef:
                        edge_snapshot = map_pb2.EdgeSnapshot()
                        edge_snapshot.ParseFromString(ef.read())
                        edges[edge_snapshot.id] = edge_snapshot
        self.get_logger().info(f"Loaded {len(edges)} edge snapshots.")

        return graph, {wp.id: wp for wp in graph.waypoints}, snapshots, edges

    def visualize_graph(self):
        """Publish GraphNav map data as Rviz markers."""
        try:
            marker_array = MarkerArray()
            marker_id = 0

            # Visualize waypoints
            for waypoint_id, waypoint in self.waypoints.items():
                pose = SE3Pose.from_proto(waypoint.waypoint_tform_ko)
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "waypoints"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = pose.x
                marker.pose.position.y = pose.y
                marker.pose.position.z = pose.z
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)

            # Visualize edges
            for edge in self.graph.edges:
                from_wp = self.waypoints[edge.id.from_waypoint]
                to_wp = self.waypoints[edge.id.to_waypoint]

                from_pose = SE3Pose.from_proto(from_wp.waypoint_tform_ko)
                to_pose = SE3Pose.from_proto(to_wp.waypoint_tform_ko)

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "edges"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.05
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0

                # Add points for the edge
                marker.points.append(Point(x=from_pose.x, y=from_pose.y, z=from_pose.z))
                marker.points.append(Point(x=to_pose.x, y=to_pose.y, z=to_pose.z))
                marker_array.markers.append(marker)

            # Publish markers
            self.marker_pub.publish(marker_array)

            # Visualize point clouds
            asyncio.run(self.batch_publish_point_clouds())
        except Exception as e:
            self.get_logger().error(f"Error in visualize_graph: {e}")

    async def batch_publish_point_clouds(self):
        """Publish point clouds in batches to manage memory usage."""
        try:
            for snapshot_id, snapshot in self.snapshots.items():
                if hasattr(snapshot, 'point_cloud') and snapshot.point_cloud.data:
                    await self.publish_point_cloud(snapshot.point_cloud, snapshot_id, batch_size=5000)
        except Exception as e:
            self.get_logger().error(f"Error in batch publishing point clouds: {e}")

    async def publish_point_cloud(self, cloud_data, snapshot_id, batch_size=5000):
        """Convert a Numpy point cloud to a PointCloud2 message and publish in batches."""
        try:
            points = np.frombuffer(cloud_data.data, dtype=np.float32).reshape(-1, 3)
            total_points = len(points)

            for start_idx in range(0, total_points, batch_size):
                end_idx = min(start_idx + batch_size, total_points)
                batch_points = points[start_idx:end_idx]

                self.get_logger().info(f"Publishing point cloud batch for snapshot {snapshot_id} with {len(batch_points)} points.")

                # Create a Header object with frame_id and timestamp
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"

                # Create and publish the PointCloud2 message
                cloud_msg = point_cloud2.create_cloud_xyz32(header, batch_points.tolist())
                self.cloud_pub.publish(cloud_msg)

                # Add a small delay to prevent system overload
                await asyncio.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f"Failed to publish point cloud for snapshot {snapshot_id}: {e}")

    def convert_map_to_pcl(self, output_path, batch_size=50000):
        """Convert the entire map to a PCL file in batches."""
        all_points = []

        for snapshot_id, snapshot in self.snapshots.items():
            if hasattr(snapshot, 'point_cloud') and snapshot.point_cloud.data:  # Check if point_cloud has data
                try:
                    points = np.frombuffer(snapshot.point_cloud.data, dtype=np.float32).reshape(-1, 3)

                    for start_idx in range(0, len(points), batch_size):
                        batch_points = points[start_idx:start_idx + batch_size]
                        all_points.append(batch_points)

                    self.get_logger().info(f"Processed snapshot {snapshot_id} with {len(points)} points in batches.")
                except Exception as e:
                    self.get_logger().error(f"Failed to process snapshot {snapshot_id}: {e}")

        if all_points:
            try:
                all_points = np.vstack(all_points)

                # Convert to Open3D point cloud and save as PCD
                pcl_cloud = o3d.geometry.PointCloud()
                pcl_cloud.points = o3d.utility.Vector3dVector(all_points)

                pcl_output_file = os.path.join(output_path, "graph_nav_map.pcd")
                o3d.io.write_point_cloud(pcl_output_file, pcl_cloud)
                self.get_logger().info(f"Saved PCL file to {pcl_output_file}")

                # Release Open3D resources
                del pcl_cloud
            except Exception as e:
                self.get_logger().error(f"Failed to save PCL file: {e}")
        else:
            self.get_logger().warning("No point cloud data found to save as PCL file.")

def main():
    rclpy.init()

    # Parse the map path argument
    import argparse
    parser = argparse.ArgumentParser(description="Visualize GraphNav map in Rviz2.")
    parser.add_argument('map_path', type=str, help='Path to the GraphNav map directory.')
    args = parser.parse_args()

    # Run the node
    node = GraphNavToRviz(args.map_path)

    # Use asyncio for point cloud publishing
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
