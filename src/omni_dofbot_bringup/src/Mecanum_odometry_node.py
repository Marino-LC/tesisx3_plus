#!/usr/bin/env python3
"""
mecanum_odometry_node.py
========================
Nodo puente: lee /joint_states → cinemática directa mecanum →
publica nav_msgs/Odometry en /odom y el TF odom→base_footprint.

Parámetros ROS 2 (coinciden con mecanum_kinematic_node.py):
  wheel_radius  : radio de rueda en metros        (default: 0.040)
  lx            : semilongitud robot eje X        (default: 0.110)
  ly            : semiancho robot eje Y           (default: 0.102)
  odom_frame    : frame del origen de odometría  (default: "odom")
  base_frame    : frame de la base del robot      (default: "base_footprint")
  publish_tf    : publicar TF odom→base_footprint (default: True)

Convención de ruedas (orden en /joint_states del controlador):
  front_left_joint   → w0
  front_right_joint  → w1
  back_right_joint   → w2
  back_left_joint    → w3

Cinemática directa mecanum estándar (ROS 2):
  vx  = r/4  * ( w0 + w1 + w2 + w3)
  vy  = r/4  * (-w0 + w1 - w2 + w3)   ← vy positivo = izquierda del robot
  wz  = r/(4*(lx+ly)) * (-w0 + w1 + w2 - w3)

Servicio de reset:
  ~/reset_pose  (std_srvs/Empty) — reinicia la pose a (0, 0, π/2),
  que coincide con el yaw del teleport que usa el AG (z=0.7071, w=0.7071).
  El AG lo llama automáticamente tras cada _teleport().

NOTA SOBRE LA POLARIDAD:
  Este nodo lee las velocidades de /joint_states TAL CUAL las reporta
  ros2_control. Si mecanum_kinematic_node aplica wheel_polarity=[-1,-1,-1,-1]
  a sus comandos, Gazebo invierte físicamente el giro, y /joint_states
  reflejará esas velocidades con su signo real. No se aplica corrección
  adicional aquí; si la odometría sale al revés, ajusta WHEEL_SIGN abajo.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Empty
import tf2_ros

# Signo por rueda: [FL, FR, BR, BL]
# Cambia a -1 las que estén físicamente invertidas para que la cinemática
# directa dé velocidades coherentes con el movimiento real del robot.
WHEEL_SIGN = [-1.0, -1.0, -1.0, -1.0]   # igual que wheel_polarity del nodo PID

# Pose de inicio fija del AG: teleport a (0,0, z=0.12) con yaw = π/2
# (quaternion: z=0.7071068, w=0.7071068)
RESET_YAW = math.pi / 2.0

WHEEL_JOINTS = [
    'front_left_joint',
    'front_right_joint',
    'back_right_joint',
    'back_left_joint',
]


class MecanumOdometryNode(Node):

    def __init__(self):
        super().__init__('mecanum_odometry_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('wheel_radius', 0.040)
        self.declare_parameter('lx',           0.110)
        self.declare_parameter('ly',           0.102)
        self.declare_parameter('odom_frame',   'odom')
        self.declare_parameter('base_frame',   'base_footprint')
        self.declare_parameter('publish_tf',   True)

        # ── Estado interno ────────────────────────────────────────────────────
        self._x          = 0.0
        self._y          = 0.0
        self._yaw        = RESET_YAW   # arranca ya alineado con el teleport del AG
        self._last_stamp = None        # se inicializa en el primer callback

        # ── TF broadcaster ────────────────────────────────────────────────────
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Publishers / Subscribers ──────────────────────────────────────────
        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)

        self.create_subscription(
            JointState, '/joint_states', self._joint_states_cb, 10
        )

        # ── Servicio de reset de pose ─────────────────────────────────────────
        # Llamado por el AG tras cada _teleport() para que la odometría
        # acumulada vuelva a coincidir con la posición real en Gazebo.
        self._reset_srv = self.create_service(
            Empty, '~/reset_pose', self._reset_cb
        )

        self.get_logger().info(
            f'mecanum_odometry_node iniciado — '
            f'r={self.get_parameter("wheel_radius").value:.3f} m  '
            f'lx={self.get_parameter("lx").value:.3f}  '
            f'ly={self.get_parameter("ly").value:.3f}  '
            f'reset_yaw={math.degrees(RESET_YAW):.1f}°'
        )

    # ── Callback del servicio de reset ────────────────────────────────────────
    def _reset_cb(self, request, response):
        """Reinicia la odometría a la pose fija del teleport del AG (0, 0, π/2)."""
        self.reset_pose(x=0.0, y=0.0, yaw=RESET_YAW)
        return response

    # ── Cinemática directa + integración ─────────────────────────────────────
    def _joint_states_cb(self, msg: JointState):
        """Calcula odometría y la publica en /odom."""

        # Mapear nombre de joint → velocidad
        name_to_vel = {name: 0.0 for name in WHEEL_JOINTS}
        for i, name in enumerate(msg.name):
            if name in name_to_vel and i < len(msg.velocity):
                name_to_vel[name] = msg.velocity[i]

        # Velocidades angulares con corrección de signo
        w0 = name_to_vel['front_left_joint']  * WHEEL_SIGN[0]
        w1 = name_to_vel['front_right_joint'] * WHEEL_SIGN[1]
        w2 = name_to_vel['back_right_joint']  * WHEEL_SIGN[2]
        w3 = name_to_vel['back_left_joint']   * WHEEL_SIGN[3]

        r  = self.get_parameter('wheel_radius').value
        lx = self.get_parameter('lx').value
        ly = self.get_parameter('ly').value
        k  = lx + ly

        # Cinemática directa mecanum
        vx = r / 4.0 * ( w0 + w1 + w2 + w3)
        vy = r / 4.0 * (-w0 + w1 - w2 + w3)
        wz = r / (4.0 * k) * (-w0 + w1 + w2 - w3)

        # ── Integración temporal ──────────────────────────────────────────────
        now = self.get_clock().now()
        if self._last_stamp is None:
            self._last_stamp = now
            return

        dt = (now - self._last_stamp).nanoseconds * 1e-9
        self._last_stamp = now

        if dt <= 0.0 or dt > 0.5:   # frame perdido o simulación pausada
            return

        # Integración en el marco del mundo
        dx = (vx * math.cos(self._yaw) - vy * math.sin(self._yaw)) * dt
        dy = (vx * math.sin(self._yaw) + vy * math.cos(self._yaw)) * dt
        self._x   += dx
        self._y   += dy
        self._yaw += wz * dt
        self._yaw  = (self._yaw + math.pi) % (2 * math.pi) - math.pi

        # ── Cuaternión desde yaw ──────────────────────────────────────────────
        qz = math.sin(self._yaw / 2.0)
        qw = math.cos(self._yaw / 2.0)

        odom_frame = self.get_parameter('odom_frame').value
        base_frame = self.get_parameter('base_frame').value
        stamp      = now.to_msg()

        # ── Publicar TF ───────────────────────────────────────────────────────
        if self.get_parameter('publish_tf').value:
            t = TransformStamped()
            t.header.stamp    = stamp
            t.header.frame_id = odom_frame
            t.child_frame_id  = base_frame
            t.transform.translation.x = self._x
            t.transform.translation.y = self._y
            t.transform.translation.z = 0.0
            t.transform.rotation.z    = qz
            t.transform.rotation.w    = qw
            self._tf_broadcaster.sendTransform(t)

        # ── Publicar Odometry ─────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = odom_frame
        odom.child_frame_id  = base_frame

        odom.pose.pose.position.x    = self._x
        odom.pose.pose.position.y    = self._y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[35] = 0.05   # yaw

        odom.twist.twist.linear.x  = vx
        odom.twist.twist.linear.y  = vy
        odom.twist.twist.angular.z = wz

        odom.twist.covariance[0]  = 0.01
        odom.twist.covariance[7]  = 0.01
        odom.twist.covariance[35] = 0.05

        self._odom_pub.publish(odom)

    # ── Reset público ─────────────────────────────────────────────────────────
    def reset_pose(self, x=0.0, y=0.0, yaw=RESET_YAW):
        """Resetea la pose acumulada. Llamar tras teleport en el AG."""
        self._x, self._y, self._yaw = x, y, yaw
        self._last_stamp = None
        self.get_logger().info(
            f'Odometría reseteada → ({x:.3f}, {y:.3f}, '
            f'yaw={math.degrees(yaw):.1f}°)'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MecanumOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()