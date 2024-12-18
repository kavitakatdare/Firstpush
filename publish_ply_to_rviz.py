#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import numpy as np
import struct

class PlyToRvizPublisher(Node):
    def __init__(self):
        super().__init__('ply_to_rviz_publisher')

        # Publisher
        self.publisher_ = self.create_publisher(PointCloud2, '/point_cloud', 10)

        # Timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_point_cloud)

        # File path to the .ply file
        self.file_path = self.declare_parameter('file_path', '/path/to/your/file.ply').get_parameter_value().string_value
        self.get_logger().info(f"Loading .ply file: {self.file_path}")

        # Load the .ply file
        self.ply_data = o3d.io.read_point_cloud(self.file_path)
        self.points = np.asarray(self.ply_data.points)
        self.get_logger().info(f"Loaded {len(self.points)} points from {self.file_path}")

    def create_pointcloud2(self, points):
        """Create a PointCloud2 message from a numpy array of points."""
        header = self.get_clock().now().to_msg()
        header.frame_id = 'map'  # Set your frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = []
        for point in points:
            cloud_data.append(struct.pack('fff', *point))
        cloud_data = b''.join(cloud_data)

        return PointCloud2(
            header=header,
            height=1,
            width=len(points),
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * len(points),
            data=cloud_data,
            is_dense=True
        )

    def publish_point_cloud(self):
        """Publish the PointCloud2 message."""
        point_cloud_msg = self.create_pointcloud2(self.points)
        self.publisher_.publish(point_cloud_msg)
        self.get_logger().info("Published point cloud to /point_cloud")


def main(args=None):
    rclpy.init(args=args)
    node = PlyToRvizPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
