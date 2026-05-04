import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument,
    SetEnvironmentVariable, RegisterEventHandler
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():

    pkg_omni_desc = get_package_share_directory('omni_dofbot_description')

    urdf_path = os.path.join(
        pkg_omni_desc, 'urdf', 'omni_dofbot_trajectory_controller.xacro'
    )
    rviz_config_path = os.path.join(
        get_package_share_directory('omni_dofbot_bringup'),
        'rviz', 'omni_dofbot_trayectory_rviz.rviz'
    )
    world = os.path.join(
        get_package_share_directory('dofbot_bringup'),
        'world', 'test_world.world'
    )

    workspace_install_dir = os.path.abspath(os.path.join(pkg_omni_desc, '..'))

    set_gz_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=workspace_install_dir
    )

    robot_description = ParameterValue(
        Command(['xacro ', urdf_path]), value_type=str
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': f'-r {world}'}.items()
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'omni_dofbot',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Controladores — se lanzan en cadena después del spawn
    # Orden: joint_state_broadcaster primero (siempre), luego los demás en paralelo
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager', '/controller_manager'],
        output='screen'
    )

    load_wheel_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wheel_velocity_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen'
    )

    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['dofbot_trajectory_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen'
    )

    load_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['dofbot_gripper_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen'
    )

    config_arg = DeclareLaunchArgument(
        name='rvizconfig', default_value=rviz_config_path
    )
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        set_gz_model_path,
        config_arg,
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        rviz2_node,
        # joint_state_broadcaster se lanza cuando spawn termina
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster]
            )
        ),
        # Los controladores de movimiento se lanzan cuando broadcaster está listo
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[
                    load_wheel_controller,
                    load_arm_controller,
                    load_gripper_controller,
                ]
            )
        ),
    ])
