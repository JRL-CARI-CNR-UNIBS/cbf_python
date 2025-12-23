#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    RegisterEventHandler,
    DeclareLaunchArgument,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ---- Launch arguments (can be overridden from external launch/CLI) ----
    fcutoff_arg = DeclareLaunchArgument(
        'fcutoff_hz',
        default_value='35.0',
        description='Cutoff frequency for ZED skeleton kinematics',
    )

    output_csv_arg = DeclareLaunchArgument(
        'output_csv',
        default_value=(
            '/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/'
            'csv_files/skeleton_vectors.csv'
        ),
        description='Output CSV path for skeleton vectors',
    )
    enable_skeleton_logging_arg = DeclareLaunchArgument(
        'enable_skeleton_logging',
        default_value='false',
        description='Enable skeleton logging in ZED launch file',
    )

    # test_path_arg = DeclareLaunchArgument(
    #     'test_path',
    #     default_value='/home/nyquist/projects/python/cbf_python_ws/src/cbf_python/cbf_python/resullts',
    #     description='Path for trajectory_logger_node test_path parameter',
    # )

    # Use LaunchConfiguration for those args
    fcutoff = LaunchConfiguration('fcutoff_hz')
    output_csv = LaunchConfiguration('output_csv')
    enable_skeleton_logging = LaunchConfiguration('enable_skeleton_logging')
    # test_path = LaunchConfiguration('test_path')
    # cbf_script = LaunchConfiguration('cbf_script')

    # ---- 1) Included ZED launch file ----
    # ros2 launch zed_skeleton_kinematics zed_skeleton_kinematics_logging.launch.py ...
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('zed_skeleton_kinematics'),
                'launch',
                'zed_skeleton_kinematics_logging.launch.py',
            )
        ),
        launch_arguments={
            'fcutoff_hz': fcutoff,
            'output_csv': output_csv,
            'enable_skeleton_logging': enable_skeleton_logging,
        }.items(),
    )
    # ---- 3) python example_cbf_optimal.py ----
    cbf_process = ExecuteProcess(
        cmd=['python3', '/home/nyquist/projects/python/cbf_python_ws/src/cbf_python/cbf_python/example_cbf_PID.py'],
        output='screen',
    )

    # Start #3 when #2 (trajectory_logger_node) starts
    # start_cbf_when_traj_starts = RegisterEventHandler(
    #     OnProcessStart(
    #         target_action=traj_logger,
    #         on_start=[cbf_process],
    #     )
    # )

    # ---- Build LaunchDescription ----
    ld = LaunchDescription()
    ld.add_action(fcutoff_arg)
    ld.add_action(output_csv_arg)
    ld.add_action(enable_skeleton_logging_arg)
    # ld.add_action(test_path_arg)
    ld.add_action(cbf_process)
    ld.add_action(zed_launch)

    return ld
