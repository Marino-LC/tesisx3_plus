"""
LAUNCH DE DIAGNÓSTICO — Paso 2 del plan de ataque

Lanza el omni_dofbot en Gazebo SIN ningún plugin de control.
El robot caerá como muñeco de trapo (normal) pero Gazebo NO debe cerrarse.

Si Gazebo crashea aquí → problema en URDF (inercias/colisiones)
Si Gazebo funciona aquí → el bug es solo del plugin gz_ros2_control (confirmado)

Uso:
  ros2 launch omni_dofbot_bringup omni_dofbot_diagnostic.launch.py

El xacro que usa este launch es omni_dofbot_no_control.xacro (ver archivo adjunto)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():

    pkg_omni_desc = get_package_share_directory('omni_dofbot_description')

    # Usa el xacro sin ros2_control (ver omni_dofbot_no_control.xacro)
    urdf_path = os.path.join(
        pkg_omni_desc, 'urdf', 'omni_dofbot_no_control.xacro'
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

    return LaunchDescription([
        set_gz_model_path,
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
    ])
