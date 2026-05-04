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
 ```bash
source /opt/ros/jazzy/setup.bash
source ~/gz_fix_ws/install/setup.bash        # 1. Overlay del fix
source ~/ROS2Dev/X3_PLUS/install/setup.bash  # 2. Tu workspace principal

## 📝 Notas de Integración: Gazebo Harmonic y `ros2_control`

Durante la migración del modelo a Gazebo Harmonic (ROS2 Jazzy), se resolvieron las siguientes limitaciones y conflictos del motor de física y los controladores:

### 1. Incompatibilidad con etiquetas `<mimic>` (Gripper)
**Problema:** Gazebo Harmonic y `gz_ros2_control` actualmente no soportan restricciones cinemáticas de tipo `mimic`. Al intentar cargar el modelo, la terminal arroja un *warning* y el motor de física sufre un micro-congelamiento, lo que retrasa la inicialización de los controladores.
**Solución:** 
* En el URDF (dentro de la etiqueta `<ros2_control>`), se eliminó la `command_interface` de todas las articulaciones subordinadas del gripper.
* El gripper se controla exclusivamente enviando comandos a la articulación principal (`grip_joint`). Las demás articulaciones se visualizan, pero no son controladas activamente por el plugin de hardware.

### 2. Conflicto de etiquetas `<ros2_control>` (Articulaciones no registradas)
**Problema:** El controlador del brazo (`dofbot_trajectory_controller`) no podía activarse porque la interfaz `arm_joint_01/position` no estaba disponible. 
**Causa:** El URDF compilado contenía dos bloques `<ros2_control>` con el mismo nombre (`name="GazeboSystem"`): uno para la base omnidireccional y otro heredado de los archivos Xacro originales del Dofbot. Gazebo solo registraba el primer bloque (las ruedas) e ignoraba por completo el brazo.
**Solución:** 
* Se unificaron todas las articulaciones en un solo archivo (`omni_dofbot_controllers.xacro`).
* Se le asignó un nombre único al bloque (ej. `name="OmniDofbotSystem"`) para evitar colisiones con archivos heredados de Gazebo Classic.

### 3. Timeouts y Colapso de Controladores (Cross-talk)
**Problema:** El `joint_state_broadcaster` y el controlador del brazo fallaban por *timeout* al inicio. A veces, enviar comandos al brazo provocaba que se movieran las ruedas.
**Causa:** El micro-congelamiento causado por el *warning* de las articulaciones `<mimic>` bloqueaba el `controller_manager` durante los primeros segundos. Los controladores que intentaban arrancar en ese momento fallaban o quedaban en un estado inestable de lazo abierto.
**Solución:** 
* Se aumentó el `switch_timeout` a `30.0` segundos en el archivo YAML de configuración.
* Se configuró el archivo `launch` para instanciar (spawn) los controladores de manera **serializada** y encadenada, dándole tiempo al motor de física de estabilizarse antes de cargar el siguiente controlador.
