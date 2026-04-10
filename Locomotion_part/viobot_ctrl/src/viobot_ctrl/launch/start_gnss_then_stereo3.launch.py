import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    gnss_launch = os.path.join(
        get_package_share_directory('gnss_comm_impl'),
        'launch',
        'gnss_spp_node.launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gnss_launch)
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('viobot_ctrl'),
                    'launch',
                    'stereo3_auto_start.launch.py'
                )
            )
        )
    ])