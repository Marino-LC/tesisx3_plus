#!/usr/bin/env python3
"""
ag_motion_tests.py
==================
Algoritmo Genético (DEAP) para sintonizar las ganancias PID (Kp, Ki, Kd)
del nodo mecanum_kinematic_node.py mediante pruebas de movimiento puro.

Sin Nav2. Sin AMCL. Sin mapa.
El robot recibe /cmd_vel directo y el fitness se mide con /odom publicado
por mecanum_odometry_node.py (cinemática directa desde /joint_states).

NOTA DE DISEÑO (limitación física conocida):
  Las mallas de las ruedas mecanum no modelan los rodillos individuales
  reales, por lo que el giro de cada rueda es correcto y la odometría
  interpreta bien el movimiento lateral esperado, pero Gazebo no traslada
  realmente al robot en el eje Y del cuerpo (no hay deslizamiento lateral
  físico). Mientras se corrige la física de las ruedas, las pruebas de
  movimiento se restringen a los dos modos que sí se validaron como
  correctos en simulación: traslación recta (eje X del cuerpo) y rotación
  pura (eje Z). La prueba lateral (Y) queda retirada de esta batería.

Pruebas vigentes:
  P1 — Línea recta: avanza y retrocede en X.
  P2 — Rotación pura: gira +90° y -90° sobre el mismo punto (ida y vuelta).
  P3 — Combinada: avanza en X, gira -90°, avanza en X (nueva orientación).

Tras cada teleport, se llama al servicio ~/reset_pose de mecanum_odometry_node
para que la odometría acumulada vuelva a (0, 0, π/2), coherente con Gazebo.

EJECUCIÓN
  # T1 — simulación
  ros2 launch omni_dofbot_bringup omni_dofbot_controller.launch.py
  # T2 — cinemática PID
  ros2 run omni_dofbot_bringup mecanum_kinematic_node.py
  # T3 — odometría (lanzar antes del AG)
  ros2 run omni_dofbot_bringup mecanum_odometry_node.py
  # T4 — este nodo
  ros2 run omni_dofbot_bringup ag_motion_tests.py

DEPENDENCIAS:
  pip install deap --break-system-packages

══════════════════════════════════════════════════════════════════════════════
CHANGELOG respecto a la versión anterior (documentado para la tesis)
══════════════════════════════════════════════════════════════════════════════
1. SINCRONIZACIÓN DE SegmentLog (bug de graficado):
   Antes, las listas de cada segmento (t, vx_ref/real, vy_ref/real,
   wz_ref/real, pos_err, x/y/yaw ref/real) se llenaban desde DOS lugares
   distintos y con distinta frecuencia: el bucle de control de _drive()/
   _rotate() (a CTRL_DT=20Hz) Y el callback _odom_cb() (a la tasa real de
   /odom). Esto producía listas de longitudes distintas y descoordinadas,
   por lo que los guards `len(seg.t) == len(seg.X)` en _build_plots()
   fallaban silenciosamente, dejando las gráficas de P1/P2/P3 y pose
   X/Y/Yaw completamente vacías. Ahora _odom_cb() es la ÚNICA fuente de
   verdad: todas las listas de un segmento se llenan juntas, una vez por
   mensaje de /odom recibido, garantizando longitudes siempre iguales.

2. ROBUSTEZ DEL HILO DEL BRAZO:
   _run_test1/2/3 ahora envuelven las primitivas de movimiento en
   try/finally, garantizando que _stop_arm() se ejecute incluso si
   _drive()/_rotate() lanzan una excepción a mitad de una prueba. Esto
   evita que el hilo daemon del brazo siga publicando después de que
   rclpy empiece a destruir el nodo (causa de un InvalidHandle observado
   en una corrida previa).

3. ELITISMO EN EL BUCLE PRINCIPAL DEL AG:
   La versión anterior seleccionaba con selTournament únicamente sobre
   `offspring`, sin reinyectar el HallOfFame a la población de trabajo.
   Análisis del log de una corrida de 50 generaciones x 100 individuos
   mostró DOS regresiones documentadas del mínimo histórico (gen 12→13 y
   gen 33→34, donde el mejor individuo encontrado se perdía de la
   población) y dos mesetas largas sin ninguna mejora (gen 18-33 y
   34-45, 28 de 50 generaciones sin cambio). Ahora los mejores `len(hof)`
   individuos se reinyectan a la población tras cada selección.

4. KD_RANGE AMPLIADO (0.0, 0.5) -> (0.0, 1.0):
   En casi todas las generaciones de la corrida analizada, el mejor
   individuo reportaba Kd exactamente en el límite superior del rango
   anterior (0.5), señal clásica de que el óptimo real podría estar más
   allá del límite explorable. Se amplía para verificar.

5. SIGMA DE MUTACIÓN POR PARÁMETRO:
   mutGaussian usaba sigma=0.5 fijo para los tres genes, pese a que sus
   rangos son muy distintos (Kp: 0-20, Ki: 0-50, Kd: 0-1). Esto hacía la
   exploración de Kp/Ki extremadamente lenta (~1-2% del rango) y la de Kd
   excesivamente brusca (~50-100% del rango). Ahora sigma se escala
   aproximadamente al 7-8% del rango de cada parámetro.
"""

import math, time, threading, subprocess, random, json, os, webbrowser
from dataclasses import dataclass, field
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from deap import base, creator, tools, algorithms

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

WORLD_NAME   = "arena_pid_tuning"
ROBOT_NAME   = "omni_dofbot"

# ── Distancias / ángulos de prueba ────────────────────────────────────────────
# El robot se teleporta con yaw=π/2 (frente apuntando al eje +Y largo de la
# arena). DIST_X en las primitivas corresponde al avance frontal del robot,
# que en el marco del mundo se mueve en la dirección Y.
DIST_X      = 0.80   # m — recta P1 / P3  (aprovecha el largo de 1.10 m)
DIST_RETURN = 0.35   # m — avance corto de regreso en P3 tras el giro
ROT_ANGLE   = math.pi / 2   # rad — ángulo de giro usado en P2 y P3 (90°)

# ── Velocidades de referencia cmd_vel ─────────────────────────────────────────
VX_REF = 0.40   # m/s
VY_REF = 0.40   # m/s  (no usado por las pruebas vigentes, se conserva por compatibilidad)
WZ_REF = 1.00   # rad/s

# ── Lazo de control ───────────────────────────────────────────────────────────
CTRL_DT      = 0.05   # s  (20 Hz)
SETTLE_TIME  = 0.60   # s  pausa entre segmentos
TIMEOUT_MOVE = 5.0    # s  timeout traslación
TIMEOUT_ROT  = 5.0    # s  timeout rotación
POS_TOL      = 0.04   # m  umbral "llegó"
YAW_TOL      = 0.05   # rad umbral "rotó"

# ── AG ──────────────────────────────────────────────────────────────────────
POP_SIZE    = 25
N_GEN       = 100
CX_PROB     = 0.50
MUT_PROB    = 0.20
# Kd casi anulado para un control de velocidad; Kp acotado para evitar
# inestabilidades severas.
KP_RANGE    = (0.0, 20.0)
KI_RANGE    = (0.0, 50.0)
KD_RANGE    = (0.0, 1.0)   # CHANGELOG #4 — antes (0.0, 0.5); el mejor individuo
                           # quedaba pegado al límite superior en casi toda la
                           # corrida analizada, así que se amplía para verificar
                           # si el óptimo real está más allá de 0.5.

# CHANGELOG #5 — sigma por parámetro, ~7-8% del rango de cada gen, en vez de
# un sigma=0.5 fijo que hacía la exploración de Kp/Ki muy lenta y la de Kd
# demasiado brusca.
MUT_SIGMA   = [1.5, 3.5, 0.08]   # [Kp, Ki, Kd]

W1, W2, W3  = 0.35, 0.30, 0.35   # pesos P1 (recta), P2 (giro), P3 (combinada)
PENALTY_TO  = 50.0

# ── Brazo Dofbot — coreografía determinista ───────────────────────────────────
ARM_JOINTS = ["arm_joint_01","arm_joint_02","arm_joint_03",
              "arm_joint_04","arm_joint_05"]
GRIP_JOINTS = ["grip_joint"]
ARM_TOPIC   = "/dofbot_trajectory_controller/joint_trajectory"
GRIP_TOPIC  = "/dofbot_gripper_controller/joint_trajectory"
ARM_MOVE_DUR = 1     # s
ARM_HOLD_DUR = 1.0   # s

# ── Brazo Dofbot — pick & place lado a lado ───────────────────────────────────
ARM_HOME       = [ 0.00,  0.00,  0.00,  0.00, 1.57]
ARM_PICK_LEFT  = [-1.20, -1.25, -0.7, -0.3, 1.57]
ARM_PICK_RIGHT = [ 1.20, -1.25, -0.7, -0.3, 1.57]

ARM_CHOREOGRAPHY = [(1,4), (3,2), (5,1)]
GRIP_OPEN   =  0.00
GRIP_CLOSED = -1.54

# ── Salidas ───────────────────────────────────────────────────────────────────
OUT_PNG  = "ag_results.png"
OUT_HTML = "ag_results.html"
OUT_JSON = "ag_results.json"

# ══════════════════════════════════════════════════════════════════════════════
# Estructuras de datos para logging
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SegmentLog:
    name:    str
    t:       List[float] = field(default_factory=list)
    vx_ref:  List[float] = field(default_factory=list)
    vy_ref:  List[float] = field(default_factory=list)
    wz_ref:  List[float] = field(default_factory=list)
    vx_real: List[float] = field(default_factory=list)
    vy_real: List[float] = field(default_factory=list)
    wz_real: List[float] = field(default_factory=list)
    pos_err: List[float] = field(default_factory=list)
    # ── Pose deseada vs obtenida (series de tiempo) ─────────────────────────
    x_ref:    List[float] = field(default_factory=list)
    y_ref:    List[float] = field(default_factory=list)
    yaw_ref:  List[float] = field(default_factory=list)
    x_real:   List[float] = field(default_factory=list)
    y_real:   List[float] = field(default_factory=list)
    yaw_real: List[float] = field(default_factory=list)

@dataclass
class IndividualLog:
    gen: int; idx: int
    kp: float; ki: float; kd: float
    cost_p1: float = 0.0; cost_p2: float = 0.0; cost_p3: float = 0.0
    fitness: float = 0.0
    segments_p1: List[SegmentLog] = field(default_factory=list)
    segments_p2: List[SegmentLog] = field(default_factory=list)
    segments_p3: List[SegmentLog] = field(default_factory=list)

@dataclass
class GenLog:
    gen: int
    min_fit: float; mean_fit: float; max_fit: float
    best_kp: float; best_ki: float; best_kd: float

# ══════════════════════════════════════════════════════════════════════════════
# DEAP
# ══════════════════════════════════════════════════════════════════════════════
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# ══════════════════════════════════════════════════════════════════════════════
# Pose 2D
# ══════════════════════════════════════════════════════════════════════════════
class Pose2D:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw
    def copy(self): return Pose2D(self.x, self.y, self.yaw)
    def dist(self, o): return math.hypot(self.x-o.x, self.y-o.y)

# ══════════════════════════════════════════════════════════════════════════════
# Nodo principal
# ══════════════════════════════════════════════════════════════════════════════
class AGMotionEvaluator(Node):

    def __init__(self):
        super().__init__("ag_motion_evaluator")
        cbg = ReentrantCallbackGroup()

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_pub  = self.create_publisher(Twist, "/cmd_vel", 10)
        self._arm_pub  = self.create_publisher(JointTrajectory, ARM_TOPIC,  10)
        self._grip_pub = self.create_publisher(JointTrajectory, GRIP_TOPIC, 10)

        self._pub_err   = self.create_publisher(Float64, "ag_live_error",    qos_profile_sensor_data)
        self._pub_vref  = self.create_publisher(Float64, "ag_live_vel_ref",  qos_profile_sensor_data)
        self._pub_vreal = self.create_publisher(Float64, "ag_live_vel_real", qos_profile_sensor_data)

        # ── Pose deseada vs obtenida — tópicos en vivo ──────────────────────────
        self._pub_x_ref   = self.create_publisher(Float64, "ag_live_x_ref",   qos_profile_sensor_data)
        self._pub_y_ref   = self.create_publisher(Float64, "ag_live_y_ref",   qos_profile_sensor_data)
        self._pub_yaw_ref = self.create_publisher(Float64, "ag_live_yaw_ref", qos_profile_sensor_data)
        self._pub_x_real   = self.create_publisher(Float64, "ag_live_x_real",   qos_profile_sensor_data)
        self._pub_y_real   = self.create_publisher(Float64, "ag_live_y_real",   qos_profile_sensor_data)
        self._pub_yaw_real = self.create_publisher(Float64, "ag_live_yaw_real", qos_profile_sensor_data)

        # ── Clientes de servicio ──────────────────────────────────────────────
        self._pid_cli = self.create_client(
            SetParameters, "/mecanum_kinematic_node/set_parameters",
            callback_group=cbg)

        self._tp_cli = self.create_client(
            SetEntityPose, f"/world/{WORLD_NAME}/set_entity_pose",
            callback_group=cbg)

        # Cliente para resetear la odometría de mecanum_odometry_node
        self._reset_odom_cli = self.create_client(
            Empty, '/mecanum_odometry_node/reset_pose',
            callback_group=cbg)

        # Cliente para resetear PID + filtro de motor de mecanum_kinematic_node
        self._reset_ctrl_cli = self.create_client(
            Empty, '/mecanum_kinematic_node/reset_controller_state',
            callback_group=cbg)

        # ── Estado de odometría ───────────────────────────────────────────────
        self._lock         = threading.Lock()
        self._pose         = Pose2D()
        self._origin       = Pose2D()
        self._vel_real     = (0.0, 0.0, 0.0)
        self._current_vref = (0.0, 0.0, 0.0)
        self._itae_accum   = 0.0

        # Tiempos reales para cálculo matemático riguroso de ITAE
        self._start_eval_t = 0.0
        self._last_odom_t  = 0.0

        self._itae_target  = Pose2D()
        self._yaw_ref_live = 0.0
        self._measuring    = False
        self._seg_log: SegmentLog = None

        self.create_subscription(
            Odometry, "/odom", self._odom_cb, qos_profile_sensor_data, callback_group=cbg)

        # ── Logging AG ────────────────────────────────────────────────────────
        self._all_individuals: List[IndividualLog] = []
        self._gen_logs:        List[GenLog]        = []
        self._current_gen = 0
        self._current_idx = 0
        self._arm_active  = False

        # CHANGELOG #6 — timestamp de arranque, usado para reportar tiempo
        # transcurrido junto con el progreso de generación/individuo en cada
        # evaluación (ver evaluate()), así se puede seguir el avance real de
        # una corrida larga sin adivinar en qué punto va.
        self._eval_start_time = time.time()

        self.get_logger().info("AGMotionEvaluator listo.")

    # ── Odometría ─────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        """
        ÚNICA fuente de verdad para el llenado de SegmentLog (ver CHANGELOG #1).
        Todas las listas de un segmento (t, vx/vy/wz ref y real, pos_err,
        x/y/yaw ref y real) se añaden juntas aquí, una vez por mensaje de
        /odom recibido, garantizando que todas queden siempre con la misma
        longitud. _drive()/_rotate() ya NO escriben directamente en slog.
        """
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        vx  = msg.twist.twist.linear.x
        vy  = msg.twist.twist.linear.y
        wz  = msg.twist.twist.angular.z

        with self._lock:
            self._pose     = Pose2D(msg.pose.pose.position.x,
                                    msg.pose.pose.position.y, yaw)
            self._vel_real = (vx, vy, wz)

            if self._measuring:
                now = time.time()
                dt = now - self._last_odom_t
                if dt <= 0.0: dt = 0.001
                self._last_odom_t = now

                ex  = self._itae_target.x - self._pose.x
                ey  = self._itae_target.y - self._pose.y
                err = math.hypot(ex, ey)

                t_real = now - self._start_eval_t
                self._itae_accum += t_real * err * dt

                x_ref   = self._itae_target.x
                y_ref   = self._itae_target.y
                yaw_ref = self._yaw_ref_live

                if self._seg_log is not None:
                    vref_x, vref_y, vref_wz = self._current_vref
                    # ── ÚNICA fuente de verdad: todo se añade aquí, junto y sincronizado ──
                    self._seg_log.t.append(t_real)
                    self._seg_log.vx_ref.append(vref_x)
                    self._seg_log.vy_ref.append(vref_y)
                    self._seg_log.wz_ref.append(vref_wz)
                    self._seg_log.vx_real.append(vx)
                    self._seg_log.vy_real.append(vy)
                    self._seg_log.wz_real.append(wz)
                    self._seg_log.pos_err.append(err)
                    self._seg_log.x_ref.append(x_ref)
                    self._seg_log.y_ref.append(y_ref)
                    self._seg_log.yaw_ref.append(yaw_ref)
                    self._seg_log.x_real.append(self._pose.x)
                    self._seg_log.y_real.append(self._pose.y)
                    self._seg_log.yaw_real.append(self._pose.yaw)

                msg_e = Float64(); msg_e.data = float(err)
                self._pub_err.publish(msg_e)

                vref_x2, vref_y2, _ = self._current_vref
                msg_vref  = Float64()
                msg_vreal = Float64()
                if abs(vref_x2) >= abs(vref_y2):
                    msg_vref.data, msg_vreal.data = float(vref_x2), float(vx)
                else:
                    msg_vref.data, msg_vreal.data = float(vref_y2), float(vy)
                self._pub_vref.publish(msg_vref)
                self._pub_vreal.publish(msg_vreal)

                m = Float64(); m.data = float(x_ref);        self._pub_x_ref.publish(m)
                m = Float64(); m.data = float(y_ref);        self._pub_y_ref.publish(m)
                m = Float64(); m.data = float(yaw_ref);      self._pub_yaw_ref.publish(m)
                m = Float64(); m.data = float(self._pose.x); self._pub_x_real.publish(m)
                m = Float64(); m.data = float(self._pose.y); self._pub_y_real.publish(m)
                m = Float64(); m.data = float(self._pose.yaw); self._pub_yaw_real.publish(m)

    def _get_pose(self) -> Pose2D:
        with self._lock: return self._pose.copy()

    def _get_vel_real(self):
        with self._lock: return self._vel_real

    def _fix_origin(self):
        with self._lock: self._origin = self._pose.copy()

    def _pose_rel(self) -> Pose2D:
        with self._lock:
            return Pose2D(self._pose.x - self._origin.x,
                          self._pose.y - self._origin.y,
                          self._pose.yaw)

    def _start_itae(self, target: Pose2D, seg_log: SegmentLog = None):
        with self._lock:
            self._itae_target  = target.copy()
            self._itae_accum   = 0.0

            # Iniciar relojes
            self._start_eval_t = time.time()
            self._last_odom_t  = time.time()

            self._measuring    = True
            self._seg_log      = seg_log
            self._yaw_ref_live = target.yaw

    def _stop_itae(self) -> float:
        with self._lock:
            self._measuring = False
            self._seg_log   = None
            return self._itae_accum

    # ── Reset de odometría ────────────────────────────────────────────────────
    def _reset_odometry(self):
        if not self._reset_odom_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "Servicio /mecanum_odometry_node/reset_pose no disponible.")
            return

        fut = self._reset_odom_cli.call_async(Empty.Request())
        t0  = time.time()
        while not fut.done() and time.time() - t0 < 2.0:
            time.sleep(0.05)

        if fut.done():
            self.get_logger().info("Odometría reseteada a (0, 0, 90°).")
        else:
            self.get_logger().warn("Timeout esperando reset de odometría.")

    def _reset_controller(self):
        """Resetea integradores PID y filtro de motor de mecanum_kinematic_node."""
        if self._reset_ctrl_cli.wait_for_service(timeout_sec=2.0):
            fut = self._reset_ctrl_cli.call_async(Empty.Request())
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 2.0:
                time.sleep(0.05)
        else:
            self.get_logger().warn("reset_controller_state no disponible.")

    # ── Brazo Dofbot ──────────────────────────────────────────────────────────
    def _send_arm(self, positions: list, duration_sec: int = ARM_MOVE_DUR):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        pt  = JointTrajectoryPoint()
        pt.positions       = [float(v) for v in positions]
        pt.time_from_start = Duration(sec=int(duration_sec), nanosec=0)
        msg.points.append(pt)
        self._arm_pub.publish(msg)

    def _send_grip(self, position: float, duration_sec: int = ARM_MOVE_DUR):
        msg = JointTrajectory()
        msg.joint_names = GRIP_JOINTS
        pt  = JointTrajectoryPoint()
        pt.positions       = [float(position)]
        pt.time_from_start = Duration(sec=int(duration_sec), nanosec=0)
        msg.points.append(pt)
        self._grip_pub.publish(msg)

    def _wait_arm(self, duration_sec: float) -> bool:
        """Espera duration_sec mientras self._arm_active siga activo.
        Devuelve False si el hilo debe detenerse anticipadamente."""
        t0 = time.time()
        while self._arm_active and time.time() - t0 < duration_sec:
            time.sleep(0.05)
        return self._arm_active

    def _arm_loop(self):
        self.get_logger().info("[Brazo] hilo iniciado — pick & place lado a lado")
        side = "left"   # primer ciclo: recoge del lado izquierdo, deja a la derecha

        while self._arm_active:
            pick_pose  = ARM_PICK_LEFT  if side == "left" else ARM_PICK_RIGHT
            place_pose = ARM_PICK_RIGHT if side == "left" else ARM_PICK_LEFT

            # 1. Ir a recoger, pinza abierta
            self._send_arm(pick_pose)
            self._send_grip(GRIP_OPEN)
            if not self._wait_arm(ARM_MOVE_DUR + ARM_HOLD_DUR): break

            # 2. Cerrar pinza (tomar el objeto)
            self._send_grip(GRIP_CLOSED)
            if not self._wait_arm(ARM_HOLD_DUR): break

            # 3. Transitar por HOME (evita arrastrar el objeto al cruzar)
            self._send_arm(ARM_HOME)
            if not self._wait_arm(ARM_MOVE_DUR + ARM_HOLD_DUR): break

            # 4. Ir al lado opuesto a depositar
            self._send_arm(place_pose)
            if not self._wait_arm(ARM_MOVE_DUR + ARM_HOLD_DUR): break

            # 5. Soltar el objeto
            self._send_grip(GRIP_OPEN)
            if not self._wait_arm(ARM_HOLD_DUR): break

            # 6. Volver a HOME antes de alternar de lado
            self._send_arm(ARM_HOME)
            if not self._wait_arm(ARM_MOVE_DUR + ARM_HOLD_DUR): break

            side = "right" if side == "left" else "left"   # alterna — lado a lado

        self._send_arm(ARM_HOME)
        self._send_grip(GRIP_OPEN)
        self.get_logger().info("[Brazo] hilo detenido → HOME")

    def _start_arm(self):
        self._arm_active  = True
        self._arm_thread  = threading.Thread(target=self._arm_loop, daemon=True)
        self._arm_thread.start()

    def _stop_arm(self):
        self._arm_active = False
        if hasattr(self, '_arm_thread'):
            self._arm_thread.join(timeout=ARM_MOVE_DUR + 1.0)

    # ── cmd_vel ───────────────────────────────────────────────────────────────
    def _send(self, vx=0.0, vy=0.0, wz=0.0):
        with self._lock:
            self._current_vref = (vx, vy, wz)
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    def _stop(self): self._send()

    # ── Teleport ──────────────────────────────────────────────────────────────
    def _teleport(self):
        self._stop()
        time.sleep(0.15)
        done = False
        if self._tp_cli.wait_for_service(timeout_sec=2.0):
            req = SetEntityPose.Request()
            req.entity.name = ROBOT_NAME
            req.entity.type = Entity.MODEL
            req.pose = Pose()
            req.pose.position    = Point(x=0.0, y=0.0, z=0.12)
            # yaw = π/2  →  quaternion (z=sin(π/4), w=cos(π/4))
            req.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.7071068, w=0.7071068)
            fut = self._tp_cli.call_async(req)
            t0  = time.time()
            while not fut.done() and time.time() - t0 < 3.0:
                time.sleep(0.05)
            done = fut.done() and fut.result() is not None

        if not done:
            rs = (f'name: "{ROBOT_NAME}", '
                f'position: {{x:0,y:0,z:0.12}}, '
                f'orientation: {{x:0,y:0,z:0.7071068,w:0.7071068}}')
            try:
                subprocess.run(
                    ["gz", "service", "-s", f"/world/{WORLD_NAME}/set_pose",
                    "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                    "--timeout", "2000", "--req", rs],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f"Teleport gz falló: {e.stderr.decode()}")

        time.sleep(0.80)

        # ORDEN CORREGIDO: resetear ANTES de fijar el origen, no después.
        self._reset_odometry()
        self._reset_controller()
        time.sleep(0.10)            # deja llegar al menos un /odom ya en cero
        self._fix_origin()

    # ── Primitivas de movimiento ──────────────────────────────────────────────
    def _drive(self, dist_m: float, axis: str, vx=0.0, vy=0.0,
           timeout=TIMEOUT_MOVE, seg_name="") -> tuple:
        p0 = self._get_pose()

        if axis == "x":
            target = Pose2D(p0.x + math.cos(p0.yaw)*dist_m,
                            p0.y + math.sin(p0.yaw)*dist_m, p0.yaw)
        else:
            target = Pose2D(p0.x - math.sin(p0.yaw)*dist_m,
                            p0.y + math.cos(p0.yaw)*dist_m, p0.yaw)

        slog = SegmentLog(name=seg_name)
        self._start_itae(target, slog)
        t0, ok = time.time(), False

        while time.time()-t0 < timeout:
            p  = self._get_pose()
            dx = p.x - p0.x
            dy = p.y - p0.y
            traveled = (
                dx*math.cos(p0.yaw) + dy*math.sin(p0.yaw)
                if axis == "x"
                else -dx*math.sin(p0.yaw) + dy*math.cos(p0.yaw)
            )

            if abs(traveled) >= abs(dist_m) - POS_TOL:
                ok = True; break

            if abs(traveled) > abs(dist_m) + 0.12:
                self.get_logger().warn(f"Overshoot ({traveled:.2f}m) — freno emergencia.")
                break

            # CHANGELOG #1 — ya NO se toca slog aquí, lo llena _odom_cb
            self._send(vx=vx, vy=vy)
            time.sleep(CTRL_DT)

        self._stop()
        itae    = self._stop_itae()
        elapsed = time.time() - t0
        time.sleep(SETTLE_TIME)
        return itae, elapsed, ok, slog


    def _rotate(self, angle_rad: float, timeout=TIMEOUT_ROT, seg_name="") -> tuple:
        p0       = self._get_pose()
        goal_yaw = p0.yaw + angle_rad
        sign     = math.copysign(1.0, angle_rad)
        t0, ok   = time.time(), False

        target = Pose2D(p0.x, p0.y, goal_yaw)
        slog   = SegmentLog(name=seg_name) if seg_name else None
        self._start_itae(target, slog)

        while time.time()-t0 < timeout:
            p    = self._get_pose()
            diff = (goal_yaw - p.yaw + math.pi) % (2*math.pi) - math.pi
            if abs(diff) <= YAW_TOL:
                ok = True; break

            wz = sign * max(0.15, min(WZ_REF, abs(diff)*1.5))

            t_rel    = time.time() - t0
            est_total = max(abs(angle_rad) / WZ_REF, 1e-3)
            frac      = min(1.0, t_rel / est_total)
            with self._lock:
                self._yaw_ref_live = p0.yaw + angle_rad * frac
                self._current_vref = (0.0, 0.0, wz)

            # CHANGELOG #1 — ya NO se toca slog aquí, lo llena _odom_cb
            self._send(wz=wz)
            time.sleep(CTRL_DT)

        self._stop()
        with self._lock:
            self._yaw_ref_live = goal_yaw
        self._stop_itae()

        p       = self._get_pose()
        diff    = (goal_yaw - p.yaw + math.pi) % (2*math.pi) - math.pi
        elapsed = time.time() - t0
        time.sleep(SETTLE_TIME)
        return abs(diff), ok, slog, elapsed

    # ── PID ───────────────────────────────────────────────────────────────────
    def _set_pid(self, kp, ki, kd):
        if not self._pid_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("mecanum_kinematic_node no disponible")
            return
        def _p(n, v):
            return Parameter(name=n,
                value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                     double_value=float(v)))
        req = SetParameters.Request()
        req.parameters = [_p("kp",kp), _p("ki",ki), _p("kd",kd)]
        fut = self._pid_cli.call_async(req)
        t0  = time.time()
        while not fut.done() and time.time()-t0 < 3.0:
            time.sleep(0.05)

    # ══════════════════════════════════════════════════════════════════════════
    # PRUEBA 1 — Línea recta frontal: adelante y atrás
    # ══════════════════════════════════════════════════════════════════════════
    def _run_test1(self, record=False):
        self.get_logger().info("── P1: línea recta adelante-atrás ──")
        self._teleport()
        self._start_arm()
        # CHANGELOG #2 — try/finally: garantiza _stop_arm() aunque _drive()
        # lance una excepción a mitad de la prueba.
        try:
            i1, t1, ok1, s1 = self._drive( DIST_X, "x", vx=+VX_REF, seg_name="P1_adelante")
            i2, t2, ok2, s2 = self._drive(-DIST_X, "x", vx=-VX_REF, seg_name="P1_atras")
        finally:
            self._stop_arm()

        rel   = self._pose_rel()
        err_f = math.hypot(rel.x, rel.y)

        ITAE_REF = 0.05
        TIME_REF = 2 * DIST_X / VX_REF
        cost = (0.60*(i1+i2)/ITAE_REF + 0.30*(t1+t2)/TIME_REF
              + 0.10*err_f/POS_TOL)
        if not ok1 or not ok2: cost += PENALTY_TO

        self.get_logger().info(
            f"   ITAE={i1+i2:.4f} t={t1+t2:.1f}s err_f={err_f:.3f}m cost={cost:.4f}")
        return cost, ([s1, s2] if record else [])

    # ══════════════════════════════════════════════════════════════════════════
    # PRUEBA 2 — Rotación pura: gira +90° y luego -90° (ida y vuelta)
    # ══════════════════════════════════════════════════════════════════════════
    # Sustituye a la prueba lateral original (retirada por limitación física
    # de las mallas de las ruedas mecanum: no hay deslizamiento lateral real
    # en Gazebo aunque el giro de cada rueda y la odometría sí son correctos).
    # Esta prueba valida exclusivamente el modo de rotación pura del PID,
    # que sí se comporta de forma fiable en la simulación actual.
    # ══════════════════════════════════════════════════════════════════════════
    def _run_test2(self, record=False):
        self.get_logger().info("── P2: rotación pura +90°/-90° ──")
        self._teleport()
        self._start_arm()
        try:
            err1, ok1, s1, t1 = self._rotate(+ROT_ANGLE, seg_name="P2_giro_horario")
            err2, ok2, s2, t2 = self._rotate(-ROT_ANGLE, seg_name="P2_giro_antihorario")
        finally:
            self._stop_arm()

        rel   = self._pose_rel()
        err_f = math.hypot(rel.x, rel.y)

        TIME_REF = 2 * ROT_ANGLE / WZ_REF

        cost = (0.55*(err1+err2)/(2*YAW_TOL) + 0.25*(t1+t2)/TIME_REF
              + 0.20*err_f/POS_TOL)
        if not ok1 or not ok2: cost += PENALTY_TO

        self.get_logger().info(
            f"   err_yaw1={math.degrees(err1):.1f}° err_yaw2={math.degrees(err2):.1f}° "
            f"t={t1+t2:.1f}s err_f={err_f:.3f}m cost={cost:.4f}")
        return cost, ([s1, s2] if record else [])

    # ══════════════════════════════════════════════════════════════════════════
    # PRUEBA 3 — Combinada: avance + giro −90° + avance de frente
    # ══════════════════════════════════════════════════════════════════════════
    def _run_test3(self, record=False):
        self.get_logger().info("── P3: avance + giro + regreso de frente ──")
        self._teleport()
        self._start_arm()
        try:
            i1, t1, ok1, s1   = self._drive(DIST_X, "x", vx=+VX_REF, seg_name="P3_adelante")
            err_yaw, ok_rot, s_rot, t_rot = self._rotate(-ROT_ANGLE, seg_name="P3_giro")
            i2, t2, ok2, s2   = self._drive(DIST_RETURN, "x", vx=+VX_REF, seg_name="P3_regreso")
        finally:
            self._stop_arm()

        rel   = self._pose_rel()
        err_f = math.hypot(rel.x, rel.y)

        ITAE_REF = 0.08
        TIME_REF = (DIST_X/VX_REF) + (ROT_ANGLE/WZ_REF) + (DIST_RETURN/VX_REF)

        cost = (0.45*(i1+i2)/ITAE_REF + 0.20*(t1+t2)/TIME_REF
              + 0.15*err_yaw/YAW_TOL + 0.20*err_f/POS_TOL)
        if not ok1 or not ok2 or not ok_rot: cost += PENALTY_TO

        self.get_logger().info(
            f"   ITAE={i1+i2:.4f} t={t1+t2:.1f}s "
            f"err_yaw={math.degrees(err_yaw):.1f}° err_f={err_f:.3f}m cost={cost:.4f}")

        segs = [s1, s_rot, s2] if record else []
        return cost, segs

    # ── Evaluación ────────────────────────────────────────────────────────────
    def evaluate(self, individual) -> tuple:
        kp, ki, kd = individual

        # --- IMPLEMENTACIÓN DEL CHANGELOG 6 ---
        elapsed = time.time() - self._eval_start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        
        self.get_logger().info(
            f"=== [Progreso AG] Tiempo: {h:02d}:{m:02d}:{s:02d} | "
            f"Gen: {self._current_gen}/{N_GEN} | "
            f"Ind: {self._current_idx + 1}/{POP_SIZE} ==="
        )
        # --------------------------------------

        self.get_logger().info(f"[AG] Kp={kp:.3f} Ki={ki:.3f} Kd={kd:.3f}")
        self._set_pid(kp, ki, kd)

        c1, _ = self._run_test1()
        c2, _ = self._run_test2()
        c3, _ = self._run_test3()
        fitness = W1*c1 + W2*c2 + W3*c3

        ilog = IndividualLog(
            gen=self._current_gen, idx=self._current_idx,
            kp=kp, ki=ki, kd=kd,
            cost_p1=c1, cost_p2=c2, cost_p3=c3, fitness=fitness)
        self._all_individuals.append(ilog)
        self._current_idx += 1

        self.get_logger().info(
            f"[AG] P1={c1:.4f} P2={c2:.4f} P3={c3:.4f} fit={fitness:.5f}")
        return (fitness,)

    def record_best(self, best_ind):
        kp, ki, kd = best_ind
        self.get_logger().info(
            f"[AG] Grabando mejor: Kp={kp:.3f} Ki={ki:.3f} Kd={kd:.3f}")
        self._set_pid(kp, ki, kd)
        c1, segs1 = self._run_test1(record=True)
        c2, segs2 = self._run_test2(record=True)
        c3, segs3 = self._run_test3(record=True)
        return segs1, segs2, segs3


# ══════════════════════════════════════════════════════════════════════════════
# Gráficas (matplotlib + plotly)
# ══════════════════════════════════════════════════════════════════════════════
def _build_plots(gen_logs, all_inds, segs1, segs2, segs3,
                 best_kp, best_ki, best_kd):

    gens     = [g.gen      for g in gen_logs]
    min_fit  = [g.min_fit  for g in gen_logs]
    mean_fit = [g.mean_fit for g in gen_logs]
    max_fit  = [g.max_fit  for g in gen_logs]
    best_kps = [g.best_kp  for g in gen_logs]
    best_kis = [g.best_ki  for g in gen_logs]
    best_kds = [g.best_kd  for g in gen_logs]
    all_kp   = [i.kp       for i in all_inds]
    all_ki   = [i.ki       for i in all_inds]
    all_fit  = [i.fitness  for i in all_inds]

    # ── PNG ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(18, 22))
        fig.suptitle("AG — Sintonización PID base mecanum\n"
                     f"Mejor: Kp={best_kp:.4f}  Ki={best_ki:.4f}  Kd={best_kd:.4f}",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(gens, min_fit,  "g-o", ms=5, lw=1.8, label="min")
        ax1.plot(gens, mean_fit, "b-s", ms=5, lw=1.8, label="media")
        ax1.plot(gens, max_fit,  "r-^", ms=5, lw=1.8, label="max")
        ax1.fill_between(gens, min_fit, max_fit, alpha=0.12, color="blue")
        ax1.set_title("Evolución del fitness"); ax1.set_xlabel("Generación")
        ax1.set_ylabel("Fitness"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(gens, best_kps, "r-o", ms=5, lw=1.8, label="Kp")
        ax2.plot(gens, best_kis, "g-s", ms=5, lw=1.8, label="Ki")
        ax2.plot(gens, best_kds, "b-^", ms=5, lw=1.8, label="Kd")
        ax2.set_title("Ganancias del mejor por generación")
        ax2.set_xlabel("Generación"); ax2.set_ylabel("Ganancia")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3_has_data = False
        for seg in segs1:
            if seg.t and len(seg.t) == len(seg.vx_real):
                ax3.plot(seg.t, seg.vx_ref,  "--", lw=1.5, label=f"{seg.name} ref",  alpha=0.8)
                ax3.plot(seg.t, seg.vx_real, "-",  lw=1.5, label=f"{seg.name} real")
                ax3_has_data = True
        ax3.set_title("P1 — vx"); ax3.set_xlabel("t (s)"); ax3.set_ylabel("vx (m/s)")
        if ax3_has_data: ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4_has_data = False
        for seg in segs2:
            if seg.t and len(seg.t) == len(seg.wz_real):
                ax4.plot(seg.t, seg.wz_ref,  "--", lw=1.5, label=f"{seg.name} ref",  alpha=0.8)
                ax4.plot(seg.t, seg.wz_real, "-",  lw=1.5, label=f"{seg.name} real")
                ax4_has_data = True
        ax4.set_title("P2 — wz (rotación pura)"); ax4.set_xlabel("t (s)"); ax4.set_ylabel("wz (rad/s)")
        if ax4_has_data: ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 0])
        for seg in segs3:
            if seg.t and len(seg.t) == len(seg.pos_err):
                ax5.plot(seg.t, seg.pos_err, "-", lw=1.5, label=seg.name)
        ax5.axhline(POS_TOL, color="r", ls=":", lw=1.2, label=f"tol={POS_TOL}m")
        ax5.set_title("P3 — error de posición"); ax5.set_xlabel("t (s)")
        ax5.set_ylabel("|err| (m)"); ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

        ax6  = fig.add_subplot(gs[2, 1])
        scat = ax6.scatter(all_kp, all_ki, c=all_fit, cmap="viridis_r",
                           s=35, alpha=0.75, edgecolors="none")
        plt.colorbar(scat, ax=ax6, label="Fitness")
        ax6.scatter([best_kp], [best_ki], marker="*", s=200, c="red",
                    zorder=5, label="mejor")
        ax6.set_title("Distribución Kp–Ki"); ax6.set_xlabel("Kp")
        ax6.set_ylabel("Ki"); ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

        # ── Pose deseada vs obtenida: x(t), y(t), yaw(t) ────────────────────────
        # Se concatenan las pruebas P1+P2+P3 para tener una sola línea de
        # tiempo continua del mejor individuo (cada prueba parte de t=0,
        # así que se suma un offset acumulado para que no se sobrepongan).
        all_segs = (segs1 or []) + (segs2 or []) + (segs3 or [])

        ax7 = fig.add_subplot(gs[3, 0])
        ax8 = fig.add_subplot(gs[3, 1])
        ax9 = fig.add_subplot(gs[4, 0])

        t_offset  = 0.0
        pose_drawn = False   # flag: indica si se pintó al menos un segmento
        for seg in all_segs:
            if not seg.t or len(seg.t) != len(seg.x_real):
                continue
            t_shifted = [t + t_offset for t in seg.t]
            # Solo el primer segmento lleva label para no duplicarlos en la leyenda
            lbl_ref  = "deseada"  if not pose_drawn else "_nolegend_"
            lbl_real = "obtenida" if not pose_drawn else "_nolegend_"
            ax7.plot(t_shifted, seg.x_ref,  "--", lw=1.3, alpha=0.8,
                     color="tab:orange", label=lbl_ref)
            ax7.plot(t_shifted, seg.x_real, "-",  lw=1.3,
                     color="tab:blue",   label=lbl_real)
            ax8.plot(t_shifted, seg.y_ref,  "--", lw=1.3, alpha=0.8,
                     color="tab:orange", label=lbl_ref)
            ax8.plot(t_shifted, seg.y_real, "-",  lw=1.3,
                     color="tab:blue",   label=lbl_real)
            ax9.plot(t_shifted, [math.degrees(v) for v in seg.yaw_ref],  "--",
                     lw=1.3, alpha=0.8, color="tab:orange", label=lbl_ref)
            ax9.plot(t_shifted, [math.degrees(v) for v in seg.yaw_real], "-",
                     lw=1.3, color="tab:blue", label=lbl_real)
            if seg.t:
                t_offset += seg.t[-1] + 0.1
            pose_drawn = True

        for ax, title, ylabel in [
            (ax7, "Pose X — deseada vs obtenida (P1→P2→P3)", "x (m)"),
            (ax8, "Pose Y — deseada vs obtenida (P1→P2→P3)", "y (m)"),
            (ax9, "Pose Yaw — deseada vs obtenida (P1→P2→P3)", "yaw (°)"),
        ]:
            ax.set_title(title); ax.set_xlabel("t (s, concatenado)"); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if pose_drawn:
                ax.legend(fontsize=7)

        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] PNG guardado en {os.path.abspath(OUT_PNG)}")
    except ImportError:
        print("[plot] matplotlib no disponible — omitiendo PNG")

    # ── HTML interactivo ─────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[
                "Evolución del fitness", "Ganancias del mejor individuo",
                "P1 — vel X (ref vs real)", "P2 — vel angular (ref vs real)",
                "P3 — error de posición",  "Distribución Kp–Ki",
                "Pose X — deseada vs obtenida", "Pose Y — deseada vs obtenida",
                "Pose Yaw — deseada vs obtenida", "",
            ],
            vertical_spacing=0.07, horizontal_spacing=0.10,
        )

        fig.add_trace(go.Scatter(x=gens, y=min_fit,  name="min",
                      mode="lines+markers", line=dict(color="green")),  row=1, col=1)
        fig.add_trace(go.Scatter(x=gens, y=mean_fit, name="media",
                      mode="lines+markers", line=dict(color="royalblue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=gens, y=max_fit,  name="max",
                      mode="lines+markers", line=dict(color="red")),    row=1, col=1)

        fig.add_trace(go.Scatter(x=gens, y=best_kps, name="Kp",
                      mode="lines+markers", line=dict(color="red")),   row=1, col=2)
        fig.add_trace(go.Scatter(x=gens, y=best_kis, name="Ki",
                      mode="lines+markers", line=dict(color="green")), row=1, col=2)
        fig.add_trace(go.Scatter(x=gens, y=best_kds, name="Kd",
                      mode="lines+markers", line=dict(color="blue")),  row=1, col=2)

        COLORS = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]
        for ci, seg in enumerate(segs1 or []):
            if not seg.t or len(seg.t) != len(seg.vx_real): continue
            c = COLORS[ci % len(COLORS)]
            fig.add_trace(go.Scatter(x=seg.t, y=seg.vx_ref,  name=f"{seg.name} ref",
                          line=dict(dash="dash", color=c)), row=2, col=1)
            fig.add_trace(go.Scatter(x=seg.t, y=seg.vx_real, name=f"{seg.name} real",
                          line=dict(color=c)), row=2, col=1)

        for ci, seg in enumerate(segs2 or []):
            if not seg.t or len(seg.t) != len(seg.wz_real): continue
            c = COLORS[ci % len(COLORS)]
            fig.add_trace(go.Scatter(x=seg.t, y=seg.wz_ref,  name=f"{seg.name} ref",
                          line=dict(dash="dash", color=c)), row=2, col=2)
            fig.add_trace(go.Scatter(x=seg.t, y=seg.wz_real, name=f"{seg.name} real",
                          line=dict(color=c)), row=2, col=2)

        for ci, seg in enumerate(segs3 or []):
            if not seg.t or len(seg.t) != len(seg.pos_err): continue
            fig.add_trace(go.Scatter(x=seg.t, y=seg.pos_err, name=seg.name,
                          line=dict(color=COLORS[ci % len(COLORS)])), row=3, col=1)
        fig.add_hline(y=POS_TOL, line_dash="dot", line_color="red",
                      annotation_text=f"tol {POS_TOL}m", row=3, col=1)

        fig.add_trace(
            go.Scatter(x=all_kp, y=all_ki, mode="markers",
                       marker=dict(color=all_fit, colorscale="Viridis_r", size=8,
                                   showscale=True,
                                   colorbar=dict(title="Fitness", x=1.02)),
                       text=[f"gen={i.gen} fit={i.fitness:.4f}" for i in all_inds],
                       hoverinfo="text+x+y", name="individuos"),
            row=3, col=2)
        fig.add_trace(
            go.Scatter(x=[best_kp], y=[best_ki], mode="markers",
                       marker=dict(symbol="star", size=18, color="red"), name="mejor"),
            row=3, col=2)

        # ── Pose deseada vs obtenida — series de tiempo concatenadas ──────────
        all_segs = (segs1 or []) + (segs2 or []) + (segs3 or [])
        t_offset = 0.0
        first_pose_trace = True
        for seg in all_segs:
            if not seg.t or len(seg.t) != len(seg.x_real):
                continue
            t_shifted = [t + t_offset for t in seg.t]
            show_leg  = first_pose_trace

            fig.add_trace(go.Scatter(
                x=t_shifted, y=seg.x_ref, name="deseada",
                legendgroup="pose_ref", showlegend=show_leg,
                line=dict(dash="dash", color="orange")), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=t_shifted, y=seg.x_real, name="obtenida",
                legendgroup="pose_real", showlegend=show_leg,
                line=dict(color="royalblue")), row=4, col=1)

            fig.add_trace(go.Scatter(
                x=t_shifted, y=seg.y_ref, name="deseada",
                legendgroup="pose_ref", showlegend=False,
                line=dict(dash="dash", color="orange")), row=4, col=2)
            fig.add_trace(go.Scatter(
                x=t_shifted, y=seg.y_real, name="obtenida",
                legendgroup="pose_real", showlegend=False,
                line=dict(color="royalblue")), row=4, col=2)

            yaw_ref_deg  = [math.degrees(v) for v in seg.yaw_ref]
            yaw_real_deg = [math.degrees(v) for v in seg.yaw_real]
            fig.add_trace(go.Scatter(
                x=t_shifted, y=yaw_ref_deg, name="deseada",
                legendgroup="pose_ref", showlegend=False,
                line=dict(dash="dash", color="orange")), row=5, col=1)
            fig.add_trace(go.Scatter(
                x=t_shifted, y=yaw_real_deg, name="obtenida",
                legendgroup="pose_real", showlegend=False,
                line=dict(color="royalblue")), row=5, col=1)

            first_pose_trace = False
            if seg.t:
                t_offset += seg.t[-1] + 0.1

        fig.update_xaxes(title_text="t (s, concatenado P1→P2→P3)", row=4, col=1)
        fig.update_xaxes(title_text="t (s, concatenado P1→P2→P3)", row=4, col=2)
        fig.update_xaxes(title_text="t (s, concatenado P1→P2→P3)", row=5, col=1)
        fig.update_yaxes(title_text="x (m)",   row=4, col=1)
        fig.update_yaxes(title_text="y (m)",   row=4, col=2)
        fig.update_yaxes(title_text="yaw (°)", row=5, col=1)

        fig.update_layout(
            height=1700, width=1300,
            title_text=(f"AG — Sintonización PID base mecanum<br>"
                        f"<sub>Mejor: Kp={best_kp:.4f} Ki={best_ki:.4f} Kd={best_kd:.4f}</sub>"),
            template="plotly_white",
        )

        html_path = os.path.abspath(OUT_HTML)
        fig.write_html(html_path)
        print(f"[plot] HTML guardado en {html_path}")
        webbrowser.open('file://' + html_path)

    except ImportError:
        print("[plot] plotly no disponible — omitiendo HTML")


# ══════════════════════════════════════════════════════════════════════════════
# Decorador de bounds para operadores genéticos
# ══════════════════════════════════════════════════════════════════════════════
def _bounded(func):
    def wrapper(*args, **kwargs):
        off = func(*args, **kwargs)
        for child in off:
            for i, (lo, hi) in enumerate([KP_RANGE, KI_RANGE, KD_RANGE]):
                child[i] = float(max(lo, min(hi, child[i])))
        return off
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Callback de estadísticas por generación
# ══════════════════════════════════════════════════════════════════════════════
def _make_gen_callback(node: AGMotionEvaluator):
    def _cb(pop, gen, **kwargs):
        fits = [ind.fitness.values[0] for ind in pop]
        best = min(pop, key=lambda i: i.fitness.values[0])
        glog = GenLog(
            gen=gen,
            min_fit=min(fits), mean_fit=sum(fits)/len(fits), max_fit=max(fits),
            best_kp=best[0], best_ki=best[1], best_kd=best[2])
        node._gen_logs.append(glog)
        node._current_gen = gen + 1
        node._current_idx = 0
        node.get_logger().info(
            f"[GEN {gen}] min={glog.min_fit:.5f} mean={glog.mean_fit:.5f} "
            f"best=[{best[0]:.3f},{best[1]:.3f},{best[2]:.3f}]")
    return _cb


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = AGMotionEvaluator()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        toolbox = base.Toolbox()
        toolbox.register("kp", random.uniform, *KP_RANGE)
        toolbox.register("ki", random.uniform, *KI_RANGE)
        toolbox.register("kd", random.uniform, *KD_RANGE)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.kp, toolbox.ki, toolbox.kd), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", node.evaluate)
        toolbox.register("mate",     tools.cxBlend, alpha=0.5)
        # CHANGELOG #5 — sigma por parámetro en vez de un valor fijo único,
        # escalado aproximadamente al 7-8% del rango de cada gen (antes,
        # sigma=0.5 fijo hacía la exploración de Kp/Ki muy lenta y la de Kd
        # demasiado brusca, al ser rangos de magnitudes muy distintas).
        toolbox.register("mutate",   tools.mutGaussian, mu=0, sigma=MUT_SIGMA, indpb=0.3)
        toolbox.register("select",   tools.selTournament, tournsize=3)
        toolbox.decorate("mate",   _bounded)
        toolbox.decorate("mutate", _bounded)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min",  min)
        stats.register("mean", lambda x: sum(x)/len(x))
        stats.register("max",  max)
        hof = tools.HallOfFame(5)

        gen_cb = _make_gen_callback(node)
        node.get_logger().info(f"Iniciando AG — pop={POP_SIZE}  gen={N_GEN}")

        # Evaluación inicial
        pop = toolbox.population(n=POP_SIZE)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        hof.update(pop)

        # Bucle evolutivo
        for gen in range(N_GEN):
            offspring = algorithms.varAnd(pop, toolbox, CX_PROB, MUT_PROB)
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            # CHANGELOG #3 — ELITISMO: se reservan len(hof) espacios en la
            # nueva población para los mejores individuos históricos, en vez
            # de seleccionar únicamente sobre `offspring`. Un análisis de una
            # corrida de 50x100 sin este fix mostró DOS regresiones del
            # mínimo histórico (el mejor individuo se perdía de la
            # población al no ganar el torneo) y 28 de 50 generaciones sin
            # ninguna mejora. clone() evita que los individuos del HOF y los
            # de `pop` compartan el mismo objeto (lo que rompería mate/mutate
            # en la siguiente generación por aliasing).
            n_elite = len(hof)
            elites  = [toolbox.clone(ind) for ind in hof]
            pop[:]  = toolbox.select(offspring, len(pop) - n_elite) + elites

            hof.update(pop)
            gen_cb(pop, gen)

        best = hof[0]
        best_kp, best_ki, best_kd = best[0], best[1], best[2]
        node.get_logger().info("=" * 54)
        node.get_logger().info("MEJOR INDIVIDUO:")
        node.get_logger().info(f"  Kp = {best_kp:.6f}")
        node.get_logger().info(f"  Ki = {best_ki:.6f}")
        node.get_logger().info(f"  Kd = {best_kd:.6f}")
        node.get_logger().info(f"  Fitness = {best.fitness.values[0]:.6f}")
        node.get_logger().info("=" * 54)

        node.get_logger().info("Grabando corrida final del mejor individuo...")
        segs1, segs2, segs3 = node.record_best(best)

        def _seg_to_dict(s: SegmentLog):
            return {"name": s.name, "t": s.t,
                    "vx_ref": s.vx_ref, "vy_ref": s.vy_ref, "wz_ref": s.wz_ref,
                    "vx_real": s.vx_real, "vy_real": s.vy_real, "wz_real": s.wz_real,
                    "pos_err": s.pos_err,
                    "x_ref": s.x_ref, "y_ref": s.y_ref, "yaw_ref": s.yaw_ref,
                    "x_real": s.x_real, "y_real": s.y_real, "yaw_real": s.yaw_real}

        results = {
            "config": {
                "pop_size": POP_SIZE, "n_gen": N_GEN,
                "dist_x": DIST_X, "dist_return": DIST_RETURN,
                "rot_angle_deg": math.degrees(ROT_ANGLE),
                "vx_ref": VX_REF, "vy_ref": VY_REF, "wz_ref": WZ_REF,
                "weights": {"P1": W1, "P2": W2, "P3": W3},
                "kp_range": KP_RANGE, "ki_range": KI_RANGE, "kd_range": KD_RANGE,
                "mut_sigma": MUT_SIGMA, "elitism": True,
                "note": "P2 (lateral) retirada por limitacion fisica de mallas "
                        "mecanum; sustituida por rotacion pura.",
            },
            "best": {"kp": best_kp, "ki": best_ki, "kd": best_kd,
                     "fitness": best.fitness.values[0]},
            "generations": [
                {"gen": g.gen, "min": g.min_fit, "mean": g.mean_fit,
                 "max": g.max_fit, "best_kp": g.best_kp,
                 "best_ki": g.best_ki, "best_kd": g.best_kd}
                for g in node._gen_logs],
            "best_run": {
                "test1": [_seg_to_dict(s) for s in segs1],
                "test2": [_seg_to_dict(s) for s in segs2],
                "test3": [_seg_to_dict(s) for s in segs3],
            },
        }

        json_path = os.path.abspath(OUT_JSON)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        node.get_logger().info(f"JSON guardado en {json_path}")

        _build_plots(node._gen_logs, node._all_individuals,
                     segs1, segs2, segs3,
                     best_kp, best_ki, best_kd)

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()