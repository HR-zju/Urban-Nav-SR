from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gnss_comm_impl',
            executable='app_node',
            name='gnss_spp_node',
            output='screen',
            parameters=[{
                'enable_kml_export': True,
                'kml_directory': '/home/symc/data/kml',
                'kml_raw_directory': '/home/symc/data/kml_raw',
                'yaw_bridge.min_samples': 5,
                'yaw_bridge.min_baseline_m': 2.0,
                'yaw_bridge.gnss_stale_sec': 1.5,
                'yaw_bridge.update_gain': 0.03,
                'yaw_bridge.max_step_deg': 2.0,
                'yaw_bridge.max_innovation_deg': 25.0,
                'gnss.initial_calibration_enu': [0.0, -10.0, 0.0], # [m] (East, North, Up)
                'diagnostics.enable_bridge': False,
                'diagnostics.bridge_topic': '/baton/gnss/bridge_diagnostics',
            }]
        )
    ])
