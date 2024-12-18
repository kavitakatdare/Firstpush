#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import struct
import ctypes

def create_pointcloud2(points):
    """
    Create a PointCloud2 message from a numpy array of points.
    Each point is represented as [x, y, z].
    """
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"  # Set the appropriate frame ID for your setup

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]

    # Flatten the points array and pack it into bytes
    cloud_data = []
    for point in points:
        cloud_data.append(struct.pack('fff', *point))
    cloud_data = b''.join(cloud_data)

    point_cloud_msg = PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=12,  # 3 floats * 4 bytes each
        row_step=12 * len(points),
        data=cloud_data,
        is_dense=True
    )

    return point_cloud_msg

def publish_ply_to_rviz(file_path):
    """
    Load a .ply file, convert it to a PointCloud2 message, and publish it.
    """
    rospy.init_node('ply_to_rviz_publisher', anonymous=True)
    pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)  # Publish at 1 Hz

    # Load the .ply file using Open3D
    ply_data = o3d.io.read_point_cloud(file_path)
    points = np.asarray(ply_data.points)  # Extract point coordinates

    rospy.loginfo(f"Loaded {len(points)} points from {file_path}")

    while not rospy.is_shutdown():
        # Create and publish the PointCloud2 message
        cloud_msg = create_pointcloud2(points)
        pub.publish(cloud_msg)
        rospy.loginfo("Point cloud published to /point_cloud")
        rate.sleep()

if __name__ == '__main__':
    try:
        file_path = "/path/to/your/file.ply"  # Replace with your .ply file path
        publish_ply_to_rviz(file_path)
    except rospy.ROSInterruptException:
        pass
