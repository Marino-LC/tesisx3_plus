## Solución de Problemas (Troubleshooting)

### Gazebo crashea al iniciar (`segmentation fault` en `libgz_hardware_plugins.so`)

**Síntoma:** 
Al lanzar la simulación completa con los controladores habilitados en ROS2 Jazzy (Gazebo Harmonic), Gazebo se cierra abruptamente. El *stack trace* muestra que el error ocurre dentro de `libgz_hardware_plugins.so` (específicamente al buscar articulaciones en el `EntityComponentManager`). El modelo carga perfectamente si se deshabilitan las etiquetas `<ros2_control>`.

**Causa:** 
Existe un bug en los binarios actuales instalados vía `apt` para el paquete `ros-jazzy-gz-ros2-control` (versión `1.2.17` o inferior).

**Solución (Workaround):**
La solución consiste en compilar el plugin de hardware directamente desde la rama fuente y hacer un *overlay* del workspace para que ROS2 priorice nuestra versión compilada en lugar de la del sistema.

**Pasos de la solución:**

1. Crear un workspace dedicado para el fix y clonar el repositorio:
   ```bash
   mkdir -p ~/gz_fix_ws/src
   cd ~/gz_fix_ws/src
   git clone [https://github.com/ros-controls/gz_ros2_control.git](https://github.com/ros-controls/gz_ros2_control.git) --branch jazzy --depth 1

2. Compilar el paquete:
cd ~/gz_fix_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select gz_ros2_control --cmake-args -DCMAKE_BUILD_TYPE=Release

3. Hacer el source en el orden correcto 
source /opt/ros/jazzy/setup.bash
source ~/gz_fix_ws/install/setup.bash        # 1. Overlay del fix
source ~/ROS2Dev/X3_PLUS/install/setup.bash  # 2. Tu workspace principal

