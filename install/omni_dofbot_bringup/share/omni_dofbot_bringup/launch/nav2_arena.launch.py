import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('omni_dofbot_bringup')
    map_file = os.path.join(pkg_dir, 'maps', 'arena_map.yaml')
    params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    # Añadimos behavior_server a la lista oficial
    lifecycle_nodes = ['map_server', 'planner_server', 'controller_server', 'behavior_server', 'bt_navigator']
    return LaunchDescription([
        # 1. Transformación estática (Reemplazo de AMCL)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),
        # 2. Servidor de Mapas
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            # ¡IMPORTANTE! params_file va primero, el diccionario va segundo
            parameters=[params_file, {'yaml_filename': map_file, 'use_sim_time': True}],
            output='screen'
        ),
        # 3. Planificador Global
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            parameters=[params_file],
            output='screen'
        ),
        # 4. Controlador Local (PID/Seguimiento)
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            parameters=[params_file],
            output='screen'
        ),
        # --- EL NUEVO NODO (Satisfaciendo a bt_navigator) ---
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            parameters=[params_file],
            output='screen'
        ),
        # 5. Behavior Tree (Orquestador de acciones)
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            parameters=[params_file],
            output='screen'
        ),
        # 6. El Administrador de Vida (Enciende todos los anteriores al mismo tiempo)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            parameters=[{'use_sim_time': True, 'autostart': True, 'node_names': lifecycle_nodes, 'bond_timeout':10}],
            output='screen'
        ),
        # 7. Relacionar base_link con base_footprint
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link'],
            output='screen'
        )
    ])