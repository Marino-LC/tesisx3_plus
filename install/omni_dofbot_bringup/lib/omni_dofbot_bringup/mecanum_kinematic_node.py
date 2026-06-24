#!/usr/bin/env python3
"""
mecanum_kinematic_node.py

Nodo ROS 2 que combina la cinemática inversa mecanum con un controlador
PID por rueda. Las ganancias son parámetros ROS 2 modificables en tiempo
de ejecución — el Algoritmo Genético las ajusta vía set_parameters().

NUEVO: incluye un modelo de "dinámica de motor" (filtro de primer orden)
entre la salida del PID y el comando publicado a las ruedas. Esto emula
el retraso electromecánico real del motor (identificado en MATLAB con
tfest a partir de pruebas de impulso/escalón con el encoder), que en
Gazebo no existe por defecto porque gz_ros2_control aplica el comando
de velocidad casi instantáneamente.

SUSCRIBE:
  /cmd_vel      (geometry_msgs/Twist)   — velocidad deseada del cuerpo
  /joint_states (sensor_msgs/JointState) — velocidad real de cada rueda

PUBLICA:
  /wheel_velocity_controller/commands (std_msgs/Float64MultiArray)

SERVICIOS:
  ~/reset_controller_state (std_srvs/Trigger)
      Reinicia los integradores PID y el estado del filtro de motor.
      Debe llamarse junto con ~/reset_pose (de mecanum_odometry_node)
      cada vez que se teleporta el robot, para que no queden residuos
      de velocidad/integral de la prueba anterior contaminando la
      siguiente medición del AG.

PARÁMETROS (modificables en tiempo de ejecución):
  wheel_radius   : radio de rueda en metros          (default: 0.040)
  lx             : semilongitud robot eje X en metros (default: 0.110)
  ly             : semiancho robot eje Y en metros    (default: 0.102)
  max_wheel_vel  : saturación de velocidad (rad/s)   (default: 20.0)
  control_rate   : frecuencia del lazo PID (Hz)      (default: 50.0)

  Ganancias PID — mismas para las 4 ruedas:
  kp             : ganancia proporcional              (default: 1.0)
  ki             : ganancia integral                  (default: 0.0)
  kd             : ganancia derivativa                (default: 0.0)
  i_clamp        : límite anti-windup del integrador  (default: 5.0)

  Dinámica de motor (NUEVO):
  motor_tau      : constante de tiempo del motor (s)  (default: 0.0)
                   0.0 = filtro desactivado (comportamiento idéntico
                   al nodo original, sin modelo de motor).
                   Calcúlalo como 1/polo a partir de la función de
                   transferencia identificada con tfest, YA CONVERTIDA
                   a rad/s (ver conversación previa sobre CPR_rueda).

CONVENCIÓN DE RUEDAS (orden del controlador):
  índice 0 → front_left_joint
  índice 1 → front_right_joint
  índice 2 → back_right_joint
  índice 3 → back_left_joint
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
import numpy as np


WHEEL_JOINTS = [
    'front_left_joint',
    'front_right_joint',
    'back_right_joint',
    'back_left_joint',
]


class PIDController:
    """PID con anti-windup por saturación del integrador."""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, i_clamp=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_clamp = i_clamp
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        # Integral
        self._integral += error * dt
        self._integral = float(np.clip(self._integral, -self.i_clamp, self.i_clamp))

        # Derivada
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def update_gains(self, kp, ki, kd, i_clamp=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        if i_clamp is not None:
            self.i_clamp = i_clamp
        self.reset()


class FirstOrderLag:
    """
    Filtro de primer orden discreto que emula la dinámica electromecánica
    real del motor (constante de tiempo tau, identificada con tfest).

    Discretización exacta para retención de orden cero (ZOH) de un polo
    simple en -1/tau:
        alpha = dt / (tau + dt)
        y[n]  = y[n-1] + alpha * (u[n] - y[n-1])

    tau = 0.0 desactiva el filtro (paso directo u -> y, sin retraso).
    Esto reproduce el comportamiento ORIGINAL del nodo si no se toca
    el parámetro motor_tau.
    """

    def __init__(self, tau: float = 0.0):
        self.tau = max(0.0, tau)
        self._y = 0.0

    def reset(self, y0: float = 0.0):
        self._y = y0

    def update_tau(self, tau: float):
        self.tau = max(0.0, tau)
        

    def filter(self, u: float, dt: float) -> float:
        if self.tau <= 0.0 or dt <= 0.0:
            self._y = u
            return self._y
        alpha = dt / (self.tau + dt)
        self._y += alpha * (u - self._y)
        return self._y


class MecanumKinematicNode(Node):

    def __init__(self):
        super().__init__('mecanum_kinematic_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('wheel_radius',  0.040)
        self.declare_parameter('lx',            0.110)
        self.declare_parameter('ly',            0.102)
        self.declare_parameter('max_wheel_vel', 20.0)
        self.declare_parameter('control_rate',  50.0)
        self.declare_parameter('kp',            1.0)
        self.declare_parameter('ki',            0.0)
        self.declare_parameter('kd',            0.0)
        self.declare_parameter('i_clamp',       5.0)
        self.declare_parameter('motor_tau',     0.46)   # NUEVO

        # Callback para actualizar ganancias en tiempo de ejecución
        self.add_on_set_parameters_callback(self._on_params_change)

        # ── PID por rueda ─────────────────────────────────────────────────────
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        ic = self.get_parameter('i_clamp').value
        self._pids = [PIDController(kp, ki, kd, ic) for _ in range(4)]

        # ── Modelo de motor por rueda (NUEVO) ────────────────────────────────
        tau = self.get_parameter('motor_tau').value
        self._motor_lags = [FirstOrderLag(tau) for _ in range(4)]

        # ── Estado ────────────────────────────────────────────────────────────
        self._omega_ref  = np.zeros(4)   # velocidades de referencia (rad/s)
        self._omega_real = np.zeros(4)   # velocidades reales leídas de joint_states
        self._last_time  = self.get_clock().now()
        # Esto te permite invertir la lógica de cada motor individualmente:
        self.wheel_polarity = [-1, -1, -1, -1] # [fl, fr, br, bl]
        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_cb, 10
        )
        self._joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_states_cb, 10
        )
        self._wheel_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/wheel_velocity_controller/commands',
            10
        )

        # ── Servicio de reset de estado (NUEVO) ──────────────────────────────
        # Llamar junto con ~/reset_pose de mecanum_odometry_node cada vez que
        # se teleporta el robot entre evaluaciones del AG, para que el
        # integrador del PID y el estado del filtro de motor no arrastren
        # residuos de la prueba anterior.
        self._reset_srv = self.create_service(
            Trigger, '~/reset_controller_state', self._on_reset_state
        )

        # ── Timer del lazo de control ─────────────────────────────────────────
        rate = self.get_parameter('control_rate').value
        self._timer = self.create_timer(1.0 / rate, self._control_loop)

        self.get_logger().info(
            f'MecanumKinematicNode iniciado — '
            f'Kp={kp:.3f} Ki={ki:.3f} Kd={kd:.3f} '
            f'motor_tau={tau:.4f}s '
            f'rate={rate:.0f}Hz'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cmd_vel_cb(self, msg: Twist):
        """Convierte Twist a velocidades de referencia por rueda."""
        r  = self.get_parameter('wheel_radius').value
        lx = self.get_parameter('lx').value
        ly = self.get_parameter('ly').value
        k  = lx + ly

        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

      # Cinemática inversa mecanum corregida (Estándar ROS 2)
        # Asumiendo: vx=adelante, vy=izquierda, wz=antihorario
        omega_raw = np.array([
        (vx - vy - k * wz) / r,   # front_left
        (vx + vy + k * wz) / r,   # front_right
        (vx - vy + k * wz) / r,   # back_right
        (vx + vy - k * wz) / r,   # back_left
        ])

        # Aplicar polaridad para corregir motores invertidos
        self._omega_ref = omega_raw * self.wheel_polarity

    def _joint_states_cb(self, msg: JointState):
        """Lee las velocidades reales de las ruedas desde /joint_states."""
        name_to_idx = {name: i for i, name in enumerate(WHEEL_JOINTS)}
        for i, name in enumerate(msg.name):
            if name in name_to_idx and i < len(msg.velocity):
                self._omega_real[name_to_idx[name]] = msg.velocity[i]

    def _control_loop(self):
        """Lazo PID ejecutado a control_rate Hz, seguido del modelo de motor."""
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        max_vel = self.get_parameter('max_wheel_vel').value

        commands = []
        for i, pid in enumerate(self._pids):
            error = self._omega_ref[i] - self._omega_real[i]
            u = self._omega_ref[i] + pid.compute(error, dt)
            u = float(np.clip(u, -max_vel, max_vel))

            # NUEVO: filtro de primer orden — emula la dinámica real del
            # motor antes de que el comando llegue a la rueda simulada.
            # Con motor_tau=0.0 esto es un paso directo (u_real == u).
            u_real = self._motor_lags[i].filter(u, dt)

            commands.append(u_real)

        msg = Float64MultiArray()
        msg.data = commands
        self._wheel_cmd_pub.publish(msg)

    def _on_reset_state(self, request, response):
        """Reinicia integradores PID y estado del filtro de motor.

        Llamar después de cada teleport del robot (junto con el
        ~/reset_pose de mecanum_odometry_node) para que la siguiente
        prueba del AG no arranque con residuos de la anterior.
        """
        for pid in self._pids:
            pid.reset()
        for lag in self._motor_lags:
            lag.reset(0.0)

        self.get_logger().info('Estado del controlador reiniciado (PID + modelo de motor).')
        response.success = True
        response.message = 'PID integrators and motor lag filters reset.'
        return response

    def _on_params_change(self, params):
        """Actualiza ganancias PID y tau del motor cuando el AG llama a set_parameters()."""
        kp  = self.get_parameter('kp').value
        ki  = self.get_parameter('ki').value
        kd  = self.get_parameter('kd').value
        ic  = self.get_parameter('i_clamp').value
        tau = self.get_parameter('motor_tau').value

        for param in params:
            if param.name == 'kp':   kp = param.value
            elif param.name == 'ki': ki = param.value
            elif param.name == 'kd': kd = param.value
            elif param.name == 'i_clamp': ic = param.value
            elif param.name == 'motor_tau': tau = param.value

        for pid in self._pids:
            pid.update_gains(kp, ki, kd, ic)

        for lag in self._motor_lags:
            lag.update_tau(tau)

        self.get_logger().info(
            f'Ganancias PID actualizadas — Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f} '
            f'motor_tau={tau:.4f}s'
        )
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = MecanumKinematicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()