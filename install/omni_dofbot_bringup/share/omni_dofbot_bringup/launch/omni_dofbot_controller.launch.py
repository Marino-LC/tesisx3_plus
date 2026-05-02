import os
from ament_index_python.packages import get_package_share_path, get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
import xacro

def generate_launch_description():

    # ================= RUTAS =================
    pkg_omni_desc = get_package_share_directory('omni_dofbot_description')
    
    urdf_path = os.path.join(pkg_omni_desc, 'urdf', 'omni_dofbot_trajectory_controller.xacro')
    rviz_config_path = os.path.join(get_package_share_directory('omni_dofbot_bringup'), 'rviz', 'omni_dofbot_trayectory_rviz.rviz')
    world = os.path.join(get_package_share_directory('dofbot_bringup'), 'world', 'test_world.world')

    # ================= VARIABLE DE ENTORNO CRÍTICA PARA GAZEBO =================
    # Le decimos a Gazebo dónde buscar las mallas 3D (archivos .STL o .DAE)
    # Buscamos la carpeta 'share' padre, donde están instalados todos tus paquetes
    workspace_install_dir = os.path.abspath(os.path.join(pkg_omni_desc, '..'))
    
    set_gz_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=workspace_install_dir
    )

    # ================= PROCESAMIENTO URDF =================
    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # ================= NODOS CORE =================
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    # ================= GAZEBO HARMONIC =================
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world}'}.items()
    )
    
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description',
                   '-name', 'omni_dofbot',
                   '-z', '0.1'],
        output='screen'
    )

    # ================= RVIZ2 =================
    config_arg = DeclareLaunchArgument(name='rvizconfig', default_value=rviz_config_path)
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}]
    )

    # ================= CONTROLADORES =================
    load_joint_state_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    wheel_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["wheel_velocity_controller", "--controller-manager", "/controller_manager"],
    )

    dofbot_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["dofbot_trajectory_controller", "--controller-manager", "/controller_manager"],
    )

    dofbot_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["dofbot_gripper_controller", "--controller-manager", "/controller_manager"],
    )

    # ================= RETORNO =================
    return LaunchDescription([
        set_gz_model_path, # Inyectamos la ruta a Gazebo antes de que arranque
        config_arg,
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        rviz2_node,
        load_joint_state_controller,
        wheel_controller
    ])