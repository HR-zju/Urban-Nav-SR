from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='viobot_ctrl',
            executable='stereo3_auto_start_node',
            name='stereo3_auto_start_node',
            output='screen',
        )
    ])