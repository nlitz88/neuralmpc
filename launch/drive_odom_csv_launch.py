# launch file for:
#  - getting parameters in yaml file
#  - launching csv recorder node

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pure_pursuit',
            executable='drive_odom_csv_node',
            name='drive_odom_csv_node',
            parameters=[os.path.join(
                get_package_share_directory('pure_pursuit'),
                'config', 'pure_pursuit_params.yaml')],
            output='screen')
        ])
