cmake_minimum_required(VERSION 3.8)
project(omni_dofbot_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Directorios a instalar (¡Aquí agregamos 'maps'!)
install(
  DIRECTORY launch rviz world config maps
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(${PROJECT_NAME})

# Ejecutables Python
install(PROGRAMS
  src/mecanum_kinematic_node.py
  src/zn_tuner_openloop.py
  src/Mecanum_odometry_node.py
  src/zn_tuner.py
  src/ag_motion_tests.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()