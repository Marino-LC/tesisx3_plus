#!/usr/bin/env python3
"""
mecanum_kinematic_node.py

Nodo ROS 2 que combina la cinemática inversa mecanum con un controlador
PID por rueda. Las ganancias son parámetros ROS 2 modificables en tiempo
de ejecución — el Algoritmo Genético las ajusta vía set_parameters().

SUSCRIBE:
  /cmd_vel      (geometry_msgs/Twist)   — velocidad deseada del cuerpo
  /joint_states (sensor_msgs/JointState) — velocidad real de cada rueda

PUBLICA:
  /wheel_velocity_controller/commands (std_msgs/Float64MultiArray)

PARÁMETROS (modificables en tiempo de ejecución):
  wheel_radius   : radio de rueda en metros          (default: 0.040)
  lx             : semilongitud robot eje X en metros (default: 0.110)
  ly             : semiancho robot eje Y en metros    (default: 0.102)
  max_wheel_vel  : saturación de velocidad (rad/s)   (default: 10.0)
  control_rate   : frecuencia del lazo PID (Hz)      (default: 50.0)

  Ganancias PID — mismas para las 4 ruedas:
  kp             : ganancia proporcional              (default: 1.0)
  ki             : ganancia integral                  (default: 0.0)
  kd             : ganancia derivativa                (default: 0.0)
  i_clamp        : límite anti-windup del integrador  (default: 5.0)

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

        # Integral con anti-windup
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


class MecanumKinematicNode(Node):

    def __init__(self):
        super().__init__('mecanum_kinematic_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('wheel_radius',  0.040)
        self.declare_parameter('lx',            0.110)
        self.declare_parameter('ly',            0.102)
        self.declare_parameter('max_wheel_vel', 10.0)
        self.declare_parameter('control_rate',  50.0)
        self.declare_parameter('kp',            1.0)
        self.declare_parameter('ki',            0.0)
        self.declare_parameter('kd',            0.0)
        self.declare_parameter('i_clamp',       5.0)

        # Callback para actualizar ganancias en tiempo de ejecución
        self.add_on_set_parameters_callback(self._on_params_change)

        # ── PID por rueda ─────────────────────────────────────────────────────
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        ic = self.get_parameter('i_clamp').value
        self._pids = [PIDController(kp, ki, kd, ic) for _ in range(4)]

        # ── Estado ────────────────────────────────────────────────────────────
        self._omega_ref  = np.zeros(4)   # velocidades de referencia (rad/s)
        self._omega_real = np.zeros(4)   # velocidades reales leídas de joint_states
        self._last_time  = self.get_clock().now()

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

        # ── Timer del lazo de control ─────────────────────────────────────────
        rate = self.get_parameter('control_rate').value
        self._timer = self.create_timer(1.0 / rate, self._control_loop)

        self.get_logger().info(
            f'MecanumKinematicNode iniciado — '
            f'Kp={kp:.3f} Ki={ki:.3f} Kd={kd:.3f} '
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

        # Cinemática inversa mecanum (rodillos a 45°)
        # Orden: [fl, fr, br, bl]
        self._omega_ref = np.array([
            (vx - vy - k * wz) / r,   # front_left
            (vx + vy + k * wz) / r,   # front_right
            (vx - vy + k * wz) / r,   # back_right
            (vx + vy - k * wz) / r,   # back_left
        ])

    def _joint_states_cb(self, msg: JointState):
        """Lee las velocidades reales de las ruedas desde /joint_states."""
        name_to_idx = {name: i for i, name in enumerate(WHEEL_JOINTS)}
        for i, name in enumerate(msg.name):
            if name in name_to_idx and i < len(msg.velocity):
                self._omega_real[name_to_idx[name]] = msg.velocity[i]

    def _control_loop(self):
        """Lazo PID ejecutado a control_rate Hz."""
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        max_vel = self.get_parameter('max_wheel_vel').value

        commands = []
        for i, pid in enumerate(self._pids):
            error = self._omega_ref[i] - self._omega_real[i]
            u = self._omega_ref[i] + pid.compute(error, dt)
            u = float(np.clip(u, -max_vel, max_vel))
            commands.append(u)

        msg = Float64MultiArray()
        msg.data = commands
        self._wheel_cmd_pub.publish(msg)

    def _on_params_change(self, params):
        """Actualiza ganancias PID cuando el AG llama a set_parameters()."""
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        ic = self.get_parameter('i_clamp').value

        for param in params:
            if param.name == 'kp':   kp = param.value
            elif param.name == 'ki': ki = param.value
            elif param.name == 'kd': kd = param.value
            elif param.name == 'i_clamp': ic = param.value

        for pid in self._pids:
            pid.update_gains(kp, ki, kd, ic)

        self.get_logger().info(
            f'Ganancias PID actualizadas — Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}'
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