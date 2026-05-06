import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument,
    SetEnvironmentVariable, RegisterEventHandler, ExecuteProcess
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

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    # ── Estrategia definitiva ─────────────────────────────────────────────────
    #
    # El error de mimic bloquea el primer switch_controller por 5s.
    # Solución: cargar wheels en estado INACTIVE (--inactive flag),
    # lo que evita el switch y no falla. Luego activar explícitamente
    # con ros2 control set-controller-state después de que el error
    # de mimic ya ocurrió (lo dispara el primer switch real: el brazo).
    #
    # Orden:
    # 1. wheels   --inactive  → se carga sin activar (no hay switch, no hay error)
    # 2. arm      → activo    → este dispara el error de mimic y lo absorbe
    # 3. gripper  → activo    → sin problemas, el error ya pasó
    # 4. jsb      → activo    → sin problemas
    # 5. activate wheels → ahora sí funciona

    spawner_wheels_inactive = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'wheel_velocity_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30',
            '--inactive'   # carga y configura pero NO activa → no hay switch → no hay timeout
        ],
        output='screen'
    )

    spawner_arm = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'dofbot_trajectory_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30'
        ],
        output='screen'
    )

    spawner_gripper = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'dofbot_gripper_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30'
        ],
        output='screen'
    )

    spawner_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30'
        ],
        output='screen'
    )

    # Activar ruedas con switch_controller una vez que el error de mimic ya pasó
    activate_wheels = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'switch_controllers',
            '--activate', 'wheel_velocity_controller',
            '--controller-manager', '/controller_manager'
        ],
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
        clock_bridge,
        rviz2_node,

        # 1. Cargar ruedas en INACTIVE (no dispara switch, no falla)
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[spawner_wheels_inactive]
            )
        ),
        # 2. Brazo: primer switch real → dispara y absorbe el error de mimic
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_wheels_inactive,
                on_exit=[spawner_arm]
            )
        ),
        # 3. Gripper: el error ya ocurrió, switch limpio
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_arm,
                on_exit=[spawner_gripper]
            )
        ),
        # 4. JSB
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_gripper,
                on_exit=[spawner_jsb]
            )
        ),
        # 5. Activar ruedas: ahora el physics engine ya procesó el error de mimic
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_jsb,
                on_exit=[activate_wheels]
            )
        ),
    ])
