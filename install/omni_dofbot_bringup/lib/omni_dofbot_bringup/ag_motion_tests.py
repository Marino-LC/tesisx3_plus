#!/usr/bin/env python3
"""
ag_motion_tests.py
==================
Algoritmo Genético (DEAP) para sintonizar las ganancias PID (Kp, Ki, Kd)
del nodo mecanum_kinematic_node.py mediante 3 pruebas de movimiento puro.

Sin Nav2. Sin AMCL. Sin mapa.
El robot recibe /cmd_vel directo y el fitness se mide con /odom.

La odometría es RELATIVA: al inicio de cada prueba se registra la pose
actual como origen y todos los errores se calculan contra ese punto.

══════════════════════════════════════════════════════════════════════════
PRUEBA 1 — Línea recta (eje X del cuerpo)
──────────────────────────────────────────
  Spawn  ──[+vx 0.80m]──▶  A  ──[-vx 0.80m]──▶  Spawn
  Fitness: ITAE del error de posición en X durante ambos tramos.

PRUEBA 2 — Traslación lateral pura (eje Y del cuerpo)
──────────────────────────────────────────────────────
  Spawn  ──[+vy hasta tope derecho]──▶  B  ──[-vy hasta tope izquierdo]──▶  C
         ──[+vy hasta centro]──▶  Spawn
  El robot NUNCA rota. Evalúa el modo lateral del mecanum (el más exigente).
  Fitness: ITAE del error de posición en Y durante los 3 tramos.

PRUEBA 3 — Combinada con rotaciones intermedias
────────────────────────────────────────────────
  Igual que Prueba 1 (adelante y atrás), PERO:
    • En vez de regresar en línea recta hacia atrás →
      rota 90° y avanza de frente hasta el origen (por el costado).
  Secuencia:
    Spawn  ──[+vx 0.80m]──▶  A  ──[rota 90°]──▶  A'
           ──[+vx 0.80m lateral]──▶  Spawn'  (≈ spawn)
  Evalúa la combinación de control lineal + rotación + alineación final.
  Fitness: ITAE posición en cada tramo + error de posición final.

══════════════════════════════════════════════════════════════════════════
EJECUCIÓN
---------
  # Terminal 1 — simulación
  ros2 launch omni_dofbot_bringup omni_dofbot_controller.launch.py

  # Terminal 2 — cinemática PID (recibe /cmd_vel, publica a ruedas)
  ros2 run omni_dofbot_bringup mecanum_kinematic_node.py

  # Terminal 3 — este nodo
  ros2 run omni_dofbot_bringup ag_motion_tests.py

DEPENDENCIAS:
  pip install deap --break-system-packages
"""

import math
import time
import threading
import subprocess
import random
import json

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry

from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from deap import base, creator, tools, algorithms


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# ── Arena (arena_world.world) ─────────────────────────────────────────────────
ARENA_HALF_X = 0.48   # semiancho usable en X  (0.61 m pared - 0.03 grosor - 0.10 footprint)
ARENA_HALF_Y = 1.10   # semilargo  usable en Y  (±1.10 m)

# ── Distancias de las pruebas ─────────────────────────────────────────────────
# Prueba 1 y 3: distancia recta en X
DIST_X = 0.40   # m  (margen ≈ 0.08 m respecto a la pared con footprint del robot)

# Prueba 2: distancias laterales.
# El robot spawns en el centro; va al tope derecho, cruza al izquierdo, regresa.
DIST_RIGHT  =  0.40   # m a la derecha  (+Y cuerpo = -Y mundo con yaw=0)
DIST_LEFT   =  0.40   # m a la izquierda desde el centro (-Y cuerpo)
# total lateral recorrido: 0.40 + 0.80 + 0.40 = 1.60 m dentro de 2.20 m usables

# ── Velocidades de referencia enviadas por cmd_vel ────────────────────────────
VX_REF  = 0.20   # m/s adelante/atrás
VY_REF  = 0.20   # m/s lateral
WZ_REF  = 0.50   # rad/s rotación

# ── Parámetros de lazo ────────────────────────────────────────────────────────
CTRL_DT       = 0.05    # s   — período del lazo de espera (20 Hz)
SETTLE_TIME   = 0.6     # s   — pausa entre segmentos (el robot se detiene)
TIMEOUT_MOVE  = 12.0    # s   — timeout por segmento de traslación
TIMEOUT_ROT   = 6.0     # s   — timeout por segmento de rotación
POS_TOL       = 0.04    # m   — umbral para declarar que llegó al setpoint
YAW_TOL       = 0.05    # rad — umbral para declarar rotación completa

# ── AG ────────────────────────────────────────────────────────────────────────
POP_SIZE  = 16
N_GEN     = 8
CX_PROB   = 0.50
MUT_PROB  = 0.20
KP_RANGE  = (0.0, 15.0)
KI_RANGE  = (0.0,  5.0)
KD_RANGE  = (0.0,  3.0)

# Pesos de fitness combinado (P1 + P2 + P3 se suman ponderados)
W1 = 0.30   # línea recta
W2 = 0.40   # lateral  (más difícil → más peso)
W3 = 0.30   # combinada

# Penalización si el robot toca timeout o no llega
PENALTY_TIMEOUT = 50.0

# ── Infraestructura ───────────────────────────────────────────────────────────
ROBOT_NAME = "omni_dofbot"
WORLD_NAME = "arena_pid_tuning"


# ══════════════════════════════════════════════════════════════════════════════
# DEAP  (se define fuera de main para no llamar creator.create dos veces)
# ══════════════════════════════════════════════════════════════════════════════
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# ══════════════════════════════════════════════════════════════════════════════
# Clase de pose 2D
# ══════════════════════════════════════════════════════════════════════════════
class Pose2D:
    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self.x   = x
        self.y   = y
        self.yaw = yaw

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)

    def dist(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return (f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, "
                f"yaw={math.degrees(self.yaw):.1f}°)")


# ══════════════════════════════════════════════════════════════════════════════
# Nodo ROS 2
# ══════════════════════════════════════════════════════════════════════════════
class AGMotionEvaluator(Node):
    """
    Nodo que:
      1. Inyecta ganancias PID en mecanum_kinematic_node via set_parameters.
      2. Ejecuta las 3 pruebas publicando en /cmd_vel.
      3. Lee /odom para calcular el fitness (ITAE relativo).
      4. Teleporta el robot al origen entre evaluaciones.
    """

    def __init__(self):
        super().__init__("ag_motion_evaluator")
        cbg = ReentrantCallbackGroup()

        # ── Publishers / clients ──────────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._pid_client = self.create_client(
            SetParameters,
            "/mecanum_kinematic_node/set_parameters",
            callback_group=cbg,
        )
        self._teleport_client = self.create_client(
            SetEntityPose,
            f"/world/{WORLD_NAME}/set_entity_pose",
            callback_group=cbg,
        )

        # ── Odometría ─────────────────────────────────────────────────────────
        # _pose siempre contiene la pose más reciente publicada por /odom.
        # _origin se fija al inicio de cada prueba; todos los errores son
        # relativos a ese punto.
        self._lock       = threading.Lock()
        self._pose       = Pose2D()
        self._origin     = Pose2D()

        # Estado del integrador de ITAE
        self._itae_accum = 0.0
        self._itae_t     = 0.0
        self._itae_target = Pose2D()
        self._measuring  = False

        self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10, callback_group=cbg
        )

        self.get_logger().info("AGMotionEvaluator iniciado.")

    # ── Callback de odometría ─────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        yaw  = math.atan2(siny, cosy)

        with self._lock:
            self._pose = Pose2D(
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                yaw,
            )
            if self._measuring:
                ex = self._itae_target.x - self._pose.x
                ey = self._itae_target.y - self._pose.y
                err = math.hypot(ex, ey)
                # ITAE: tiempo × error acumulado
                self._itae_t     += CTRL_DT
                self._itae_accum += self._itae_t * err * CTRL_DT

    # ── Helpers de odometría ──────────────────────────────────────────────────
    def _current_pose(self) -> Pose2D:
        with self._lock:
            return self._pose.copy()

    def _pose_relative(self) -> Pose2D:
        """Pose actual relativa al origen fijado al inicio de la prueba."""
        with self._lock:
            return Pose2D(
                self._pose.x - self._origin.x,
                self._pose.y - self._origin.y,
                self._pose.yaw,
            )

    def _fix_origin(self):
        """Registra la pose actual como punto de referencia (origen relativo)."""
        with self._lock:
            self._origin = self._pose.copy()

    def _start_itae(self, target: Pose2D):
        """Activa el integrador ITAE hacia el target dado (coords absolutas)."""
        with self._lock:
            self._itae_target = target.copy()
            self._itae_accum  = 0.0
            self._itae_t      = 0.0
            self._measuring   = True

    def _stop_itae(self) -> float:
        with self._lock:
            self._measuring = False
            return self._itae_accum

    # ── Publicar / detener cmd_vel ────────────────────────────────────────────
    def _send(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(wz)
        self._cmd_vel_pub.publish(msg)

    def _stop(self):
        self._send()   # todos ceros

    # ── Teleport al origen de la arena ────────────────────────────────────────
    def _teleport_to_origin(self):
        """
        Mueve el robot al centro de la arena (0, 0) con yaw = 0.
        Intenta primero el servicio ROS 2; si falla, usa gz service CLI.
        """
        self._stop()
        time.sleep(0.15)

        done = False
        if self._teleport_client.wait_for_service(timeout_sec=2.0):
            req = SetEntityPose.Request()
            req.entity.name = ROBOT_NAME
            req.entity.type = Entity.MODEL
            req.pose = Pose()
            req.pose.position    = Point(x=0.0, y=0.0, z=0.12)
            req.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            fut = self._teleport_client.call_async(req)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 3.0:
                time.sleep(0.05)
            done = fut.done() and fut.result() is not None

        if not done:
            req_str = (
                f'name: "{ROBOT_NAME}", '
                f'position: {{x: 0.0, y: 0.0, z: 0.12}}, '
                f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
            )
            try:
                subprocess.run(
                    ["gz", "service", "-s", f"/world/{WORLD_NAME}/set_pose",
                     "--reqtype", "gz.msgs.Pose",
                     "--reptype", "gz.msgs.Boolean",
                     "--timeout", "2000", "--req", req_str],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f"Teleport gz falló: {e.stderr.decode()}")

        # Esperar a que el robot se asiente físicamente
        time.sleep(0.80)
        self._fix_origin()

    # ── Inyectar ganancias PID ────────────────────────────────────────────────
    def _set_pid(self, kp: float, ki: float, kd: float):
        if not self._pid_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("mecanum_kinematic_node no disponible.")
            return

        def _p(name: str, val: float) -> Parameter:
            return Parameter(
                name=name,
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(val),
                ),
            )

        req = SetParameters.Request()
        req.parameters = [_p("kp", kp), _p("ki", ki), _p("kd", kd)]
        fut = self._pid_client.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 3.0:
            time.sleep(0.05)

    # ── Primitiva: mover hasta que se recorra 'dist' metros en un eje ─────────
    def _drive_until_dist(
        self,
        dist_m: float,
        axis: str,               # "x" o "y"
        vx: float = 0.0,
        vy: float = 0.0,
        timeout: float = TIMEOUT_MOVE,
    ) -> tuple[float, float, bool]:
        """
        Publica velocidad constante hasta que el desplazamiento relativo al
        punto de inicio del segmento alcance dist_m en el eje indicado.

        Retorna (itae, tiempo_s, llegó_a_tiempo).
        El target para ITAE es la posición final esperada (absoluta).
        """
        p0 = self._current_pose()

        # Calcular target absoluto del segmento
        if axis == "x":
            target = Pose2D(
                p0.x + math.cos(p0.yaw) * dist_m,
                p0.y + math.sin(p0.yaw) * dist_m,
                p0.yaw,
            )
        else:  # "y"
            # +vy en cmd_vel mueve a la izquierda del robot (convención ROS)
            sign_y = math.copysign(1.0, dist_m)
            target = Pose2D(
                p0.x - math.sin(p0.yaw) * dist_m,
                p0.y + math.cos(p0.yaw) * dist_m,
                p0.yaw,
            )

        self._start_itae(target)
        t0   = time.time()
        ok   = False

        while time.time() - t0 < timeout:
            p       = self._current_pose()
            dx      = p.x - p0.x
            dy      = p.y - p0.y
            traveled = (
                dx * math.cos(p0.yaw) + dy * math.sin(p0.yaw)
                if axis == "x"
                else -dx * math.sin(p0.yaw) + dy * math.cos(p0.yaw)
            )
            if abs(traveled) >= abs(dist_m) - POS_TOL:
                ok = True
                break
            self._send(vx=vx, vy=vy)
            time.sleep(CTRL_DT)

        self._stop()
        itae = self._stop_itae()
        elapsed = time.time() - t0
        time.sleep(SETTLE_TIME)
        return itae, elapsed, ok

    # ── Primitiva: rotar 'angle_rad' radianes ─────────────────────────────────
    def _rotate(
        self,
        angle_rad: float,
        timeout: float = TIMEOUT_ROT,
    ) -> tuple[float, bool]:
        """
        Gira el robot angle_rad radianes (+ = antihorario).
        Retorna (error_yaw_residual_rad, llegó_a_tiempo).
        """
        p0       = self._current_pose()
        goal_yaw = p0.yaw + angle_rad
        sign     = math.copysign(1.0, angle_rad)
        t0       = time.time()
        ok       = False

        while time.time() - t0 < timeout:
            p    = self._current_pose()
            diff = goal_yaw - p.yaw
            # Normalizar a (−π, π]
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= YAW_TOL:
                ok = True
                break
            # Velocidad proporcional al error restante (con mínimo)
            wz = sign * max(0.15, min(WZ_REF, abs(diff) * 1.5))
            self._send(wz=wz)
            time.sleep(CTRL_DT)

        self._stop()
        p    = self._current_pose()
        diff = goal_yaw - p.yaw
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        time.sleep(SETTLE_TIME)
        return abs(diff), ok

    # ══════════════════════════════════════════════════════════════════════════
    # PRUEBA 1 — Línea recta en X: avanza y regresa
    # ══════════════════════════════════════════════════════════════════════════
    def _run_test1(self) -> float:
        """
        Desde el origen:
          1. Avanza DIST_X metros en +X (frente del robot).
          2. Regresa DIST_X metros en -X (atrás del robot, mismo eje).

        El robot NO rota en ningún momento.

        Fitness: ITAE de los 2 tramos + penalización por error final.
        """
        self.get_logger().info("── Prueba 1: línea recta X ──")
        self._teleport_to_origin()

        # Tramo 1: adelante
        itae1, t1, ok1 = self._drive_until_dist(
            DIST_X, axis="x", vx=+VX_REF
        )
        # Tramo 2: atrás (mismo eje, dirección opuesta)
        itae2, t2, ok2 = self._drive_until_dist(
            -DIST_X, axis="x", vx=-VX_REF
        )

        # Error de posición final relativo al origen
        rel = self._pose_relative()
        err_final = math.hypot(rel.x, rel.y)

        # Normalización de ITAE (valor esperado con buen control ≈ 0.05 m·s²)
        ITAE_REF = 0.05
        TIME_REF = 2 * DIST_X / VX_REF   # tiempo ideal sin PID
        total_itae = itae1 + itae2
        total_time = t1 + t2

        cost = (0.60 * total_itae / ITAE_REF
              + 0.30 * total_time  / TIME_REF
              + 0.10 * err_final   / POS_TOL)

        if not ok1 or not ok2:
            cost += PENALTY_TIMEOUT

        self.get_logger().info(
            f"   ITAE={total_itae:.4f}  t={total_time:.1f}s  "
            f"err_final={err_final:.3f}m  cost={cost:.4f}"
        )
        return cost

    # ══════════════════════════════════════════════════════════════════════════
    # PRUEBA 2 — Traslación lateral pura: derecha → izquierda → centro
    # ══════════════════════════════════════════════════════════════════════════
    def _run_test2(self) -> float:
        """
        Desde el origen con yaw = 0 (frente al norte):
          1. Desplaza DIST_RIGHT metros a la DERECHA  (-vy en marco del cuerpo,
             porque +vy ROS = izquierda del robot).
          2. Desplaza (DIST_RIGHT + DIST_LEFT) metros a la IZQUIERDA (+vy).
          3. Desplaza DIST_LEFT metros a la DERECHA (-vy) → regresa al centro.

        El robot NO rota en ningún momento.

        Fitness: ITAE de los 3 tramos + penalización por error de posición
                 final (debería estar de vuelta en el centro).
        """
        self.get_logger().info("── Prueba 2: lateral Y (D → I → C) ──")
        self._teleport_to_origin()

        total_lateral = DIST_RIGHT + DIST_LEFT   # cruce completo izq→der

        # Tramo 1: al tope derecho
        itae1, t1, ok1 = self._drive_until_dist(
            -DIST_RIGHT, axis="y", vy=-VY_REF
        )
        # Tramo 2: al tope izquierdo  (cruza el centro + DIST_LEFT)
        itae2, t2, ok2 = self._drive_until_dist(
            +total_lateral, axis="y", vy=+VY_REF
        )
        # Tramo 3: de vuelta al centro
        itae3, t3, ok3 = self._drive_until_dist(
            -DIST_LEFT, axis="y", vy=-VY_REF
        )

        rel       = self._pose_relative()
        err_final = math.hypot(rel.x, rel.y)

        ITAE_REF = 0.08
        TIME_REF = (2 * DIST_RIGHT + 2 * DIST_LEFT) / VY_REF
        total_itae = itae1 + itae2 + itae3
        total_time = t1 + t2 + t3

        cost = (0.60 * total_itae / ITAE_REF
              + 0.25 * total_time  / TIME_REF
              + 0.15 * err_final   / POS_TOL)

        if not ok1 or not ok2 or not ok3:
            cost += PENALTY_TIMEOUT

        self.get_logger().info(
            f"   ITAE={total_itae:.4f}  t={total_time:.1f}s  "
            f"err_final={err_final:.3f}m  cost={cost:.4f}"
        )
        return cost

    # ══════════════════════════════════════════════════════════════════════════
    # PRUEBA 3 — Combinada: avanza, rota, regresa de frente
    # ══════════════════════════════════════════════════════════════════════════
    def _run_test3(self) -> float:
        """
        Versión extendida de la Prueba 1 que usa rotación en lugar de
        retroceder en línea recta:

          Spawn (yaw=0)
            │  +vx   DIST_X metros
            ▼
            A  (yaw=0)
            │  rota −90°  (gira a la derecha, ahora el frente apunta al origen)
            ▼
            A' (yaw=−90°)
            │  +vx   DIST_X (0.40 m) ← el robot avanza de frente hacia el origen
            ▼
            Spawn' ≈ Spawn

        Evaluamos:
          • ITAE del tramo 1 (avance en X)
          • ITAE del tramo 2 (avance de frente hacia el origen)
          • Error angular residual de la rotación
          • Error de posición final respecto al origen

        Fitness combina los 4 términos.
        """
        self.get_logger().info("── Prueba 3: avance + giro + regreso de frente ──")
        self._teleport_to_origin()

        # Tramo 1: adelante en X
        itae1, t1, ok1 = self._drive_until_dist(
            DIST_X, axis="x", vx=+VX_REF
        )

        # Rotación: −90° (giro a la derecha para que el frente apunte al origen)
        # Con yaw inicial ≈ 0 y habiendo avanzado en +X, rotar −90°
        # hace que el frente (eje X del cuerpo) apunte en −Y del mundo,
        # que es justamente la dirección de vuelta al spawn.
        #
        # Nota: la arena tiene las paredes en Y; DIST_X = 0.80 m que es
        # exactamente el desplazamiento en X, así que avanzar DIST_X con
        # el nuevo yaw lleva de vuelta al origen.
        err_yaw, ok_rot = self._rotate(-math.pi / 2)

        # Tramo 2: avanza de frente (en la nueva dirección, hacia el origen)
        itae2, t2, ok2 = self._drive_until_dist(
            DIST_X, axis="x", vx=+VX_REF
        )

        # Error de posición final
        rel       = self._pose_relative()
        err_final = math.hypot(rel.x, rel.y)

        ITAE_REF  = 0.08
        TIME_REF  = 2 * DIST_X / VX_REF + math.pi / 2 / WZ_REF
        total_itae = itae1 + itae2
        total_time = t1 + t2

        cost = (0.45 * total_itae / ITAE_REF
              + 0.20 * total_time  / TIME_REF
              + 0.15 * err_yaw     / YAW_TOL
              + 0.20 * err_final   / POS_TOL)

        if not ok1 or not ok2 or not ok_rot:
            cost += PENALTY_TIMEOUT

        self.get_logger().info(
            f"   ITAE={total_itae:.4f}  t={total_time:.1f}s  "
            f"err_yaw={math.degrees(err_yaw):.1f}°  "
            f"err_final={err_final:.3f}m  cost={cost:.4f}"
        )
        return cost

    # ══════════════════════════════════════════════════════════════════════════
    # Función de evaluación que el AG llama
    # ══════════════════════════════════════════════════════════════════════════
    def evaluate(self, individual) -> tuple:
        kp, ki, kd = individual
        self.get_logger().info(
            f"[AG] Kp={kp:.3f}  Ki={ki:.3f}  Kd={kd:.3f}"
        )
        self._set_pid(kp, ki, kd)

        c1 = self._run_test1()
        c2 = self._run_test2()
        c3 = self._run_test3()

        fitness = W1 * c1 + W2 * c2 + W3 * c3
        self.get_logger().info(
            f"[AG] P1={c1:.4f}  P2={c2:.4f}  P3={c3:.4f}  "
            f"fitness={fitness:.5f}"
        )
        return (fitness,)


# ══════════════════════════════════════════════════════════════════════════════
# Decorador de bounds para operadores genéticos
# ══════════════════════════════════════════════════════════════════════════════
def _bounded(func):
    def wrapper(*args, **kwargs):
        offspring = func(*args, **kwargs)
        bounds = [KP_RANGE, KI_RANGE, KD_RANGE]
        for child in offspring:
            for i, (lo, hi) in enumerate(bounds):
                child[i] = float(max(lo, min(hi, child[i])))
        return offspring
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = AGMotionEvaluator()

    # El AG corre en el hilo principal; ROS 2 en un hilo daemon.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        # ── Configurar DEAP ───────────────────────────────────────────────────
        toolbox = base.Toolbox()
        toolbox.register("kp", random.uniform, *KP_RANGE)
        toolbox.register("ki", random.uniform, *KI_RANGE)
        toolbox.register("kd", random.uniform, *KD_RANGE)
        toolbox.register(
            "individual", tools.initCycle, creator.Individual,
            (toolbox.kp, toolbox.ki, toolbox.kd), n=1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", node.evaluate)
        toolbox.register("mate",     tools.cxBlend, alpha=0.5)
        toolbox.register("mutate",   tools.mutGaussian, mu=0, sigma=0.5, indpb=0.3)
        toolbox.register("select",   tools.selTournament, tournsize=3)
        toolbox.decorate("mate",   _bounded)
        toolbox.decorate("mutate", _bounded)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min",  min)
        stats.register("mean", lambda x: sum(x) / len(x))
        stats.register("max",  max)
        hof = tools.HallOfFame(5)

        # ── Evolución ─────────────────────────────────────────────────────────
        node.get_logger().info(
            f"Iniciando AG — pop={POP_SIZE}  gen={N_GEN}"
        )
        pop = toolbox.population(n=POP_SIZE)
        pop, log = algorithms.eaSimple(
            pop, toolbox,
            cxpb=CX_PROB, mutpb=MUT_PROB,
            ngen=N_GEN, stats=stats, halloffame=hof,
            verbose=True,
        )

        # ── Resultados ────────────────────────────────────────────────────────
        best = hof[0]
        node.get_logger().info("=" * 54)
        node.get_logger().info("MEJOR INDIVIDUO:")
        node.get_logger().info(f"  Kp = {best[0]:.6f}")
        node.get_logger().info(f"  Ki = {best[1]:.6f}")
        node.get_logger().info(f"  Kd = {best[2]:.6f}")
        node.get_logger().info(f"  Fitness = {best.fitness.values[0]:.6f}")
        node.get_logger().info("=" * 54)

        results = {
            "config": {
                "pop_size": POP_SIZE, "n_gen": N_GEN,
                "dist_x_m": DIST_X,
                "dist_right_m": DIST_RIGHT,
                "dist_left_m":  DIST_LEFT,
                "speed_vx": VX_REF, "speed_vy": VY_REF, "speed_wz": WZ_REF,
                "weights": {"P1": W1, "P2": W2, "P3": W3},
            },
            "best": {
                "kp": best[0], "ki": best[1], "kd": best[2],
                "fitness": best.fitness.values[0],
            },
            "hall_of_fame": [
                {"kp": ind[0], "ki": ind[1], "kd": ind[2],
                 "fitness": ind.fitness.values[0]}
                for ind in hof
            ],
            "log": str(log),
        }
        out_path = "/tmp/ag_motion_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        node.get_logger().info(f"Resultados en {out_path}")

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()