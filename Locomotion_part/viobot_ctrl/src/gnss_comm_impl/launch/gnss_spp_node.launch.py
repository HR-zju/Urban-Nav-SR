from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('gnss_comm_impl'),
        'config',
        'gnss_spp_node.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to ROS2 parameters YAML file for gnss_spp_node'
        ),
        Node(
            package='gnss_comm_impl',
            executable='app_node',
            name='gnss_spp_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')]
        )
    ])
