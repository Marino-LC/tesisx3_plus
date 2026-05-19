#!/usr/bin/env python3
"""
genetic_algorithm_node.py

Algoritmo Genético (DEAP) para sintonización de ganancias PID
de la base móvil mecanum. Evalúa cada individuo haciendo que el
robot recorra la trayectoria en zigzag de la arena de pruebas
usando Nav2, y mide el fitness como ITAE + penalización por fallo.

DEPENDENCIAS:
  pip install deap --break-system-packages

EJECUCIÓN:
  # Terminal 1 — sistema completo
  ros2 launch omni_dofbot_bringup omni_dofbot_controller.launch.py

  # Terminal 2 — Nav2 (cuando esté configurado)
  ros2 launch nav2_bringup navigation_launch.py ...

  # Terminal 3 — AG
  ros2 run omni_dofbot_bringup genetic_algorithm_node.py
"""

import os
import math
import time
import random
import threading
import subprocess
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus

from nav2_msgs.action import NavigateThroughPoses

from deap import base, creator, tools, algorithms


# ── Configuración del AG ───────────────────────────────────────────────────────
POP_SIZE    = 10      # individuos por generación
N_GEN       = 1       # generaciones (aumentar para tesis final)
CX_PROB     = 0.5     # probabilidad de cruce
MUT_PROB    = 0.2     # probabilidad de mutación
KP_RANGE    = (0.0, 8.0)
KI_RANGE    = (0.0, 2.0)
KD_RANGE    = (0.0, 3.0)
PENALTY     = 300.0   # costo si Nav2 falla o hay colisión
WORLD_NAME  = "arena_world"
ROBOT_NAME  = "omni_dofbot"

# Ruta al yaml de waypoints — instalada junto con el paquete
WAYPOINTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'share', 'omni_dofbot_bringup', 'config', 'waypoints.yaml'
)


# ── Definición DEAP (fuera de main para que creator no se llame dos veces) ────
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


class AGArenaEvaluator(Node):
    """
    Nodo ROS 2 que actúa como entorno de evaluación para el AG.
    Corre en un MultiThreadedExecutor en un hilo separado para que
    los futuros de ROS 2 no bloqueen el hilo del AG (DEAP).
    """

    def __init__(self):
        super().__init__('ag_arena_evaluator')

        # Callback group reentrant: permite llamadas concurrentes al mismo nodo
        self._cbg = ReentrantCallbackGroup()

        # ── Clientes de servicio y acción ─────────────────────────────────────
        self._pid_client = self.create_client(
            SetParameters,
            '/mecanum_kinematic_node/set_parameters',
            callback_group=self._cbg
        )
        self._nav_client = ActionClient(
            self,
            NavigateThroughPoses,
            'navigate_through_poses',
            callback_group=self._cbg
        )

        # ── Suscripción a odometría para calcular ITAE ────────────────────────
        self._odom_lock   = threading.Lock()
        self._current_x   = 0.0
        self._current_y   = 0.0
        self._itae_accum  = 0.0
        self._ref_x       = 0.0
        self._ref_y       = 0.0
        self._measuring   = False

        self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10,
            callback_group=self._cbg
        )

        # ── Cargar waypoints ──────────────────────────────────────────────────
        self._waypoints = self._load_waypoints()
        self.get_logger().info(
            f'AG listo — {len(self._waypoints)} waypoints cargados'
        )

    # ── Odometría / ITAE ──────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            self._current_x = msg.pose.pose.position.x
            self._current_y = msg.pose.pose.position.y
            if self._measuring:
                ex = self._ref_x - self._current_x
                ey = self._ref_y - self._current_y
                # ITAE: Integral of Absolute Error ponderado por tiempo
                # Aquí usamos IAE simple; el tiempo lo lleva el caller
                self._itae_accum += math.sqrt(ex**2 + ey**2) * 0.05  # dt≈50Hz

    def _start_measuring(self, ref_x: float, ref_y: float):
        with self._odom_lock:
            self._ref_x      = ref_x
            self._ref_y      = ref_y
            self._itae_accum = 0.0
            self._measuring  = True

    def _stop_measuring(self) -> float:
        with self._odom_lock:
            self._measuring = False
            return self._itae_accum

    # ── Waypoints ─────────────────────────────────────────────────────────────

    def _load_waypoints(self) -> list:
        path = WAYPOINTS_PATH
        if not os.path.exists(path):
            # Fallback: buscar en directorio de trabajo
            path = 'waypoints.yaml'
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data['waypoints']

    @staticmethod
    def _yaw_to_quat(yaw: float):
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    # ── Reset del robot ───────────────────────────────────────────────────────

    def reset_robot(self):
        """Teletransporta el robot a la posición de inicio via gz service."""
        spawn = self._waypoints[0]
        req_str = (
            f'name: "{ROBOT_NAME}", '
            f'position: {{x: {spawn["x"]}, y: {spawn["y"]}, z: 0.05}}, '
            f'orientation: {{w: 1.0, x: 0.0, y: 0.0, z: 0.0}}'
        )
        cmd = [
            'gz', 'service',
            '-s', f'/world/{WORLD_NAME}/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req_str
        ]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.get_logger().info('Robot reseteado al inicio')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(
                f'Error al resetear: {e.stderr.decode()}'
            )

    # ── PID ───────────────────────────────────────────────────────────────────

    def set_pid(self, kp: float, ki: float, kd: float):
        """Inyecta las ganancias PID en mecanum_kinematic_node."""
        while not self._pid_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Esperando /mecanum_kinematic_node/set_parameters...')

        def _make_param(name, value):
            return Parameter(
                name=name,
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(value)
                )
            )

        req = SetParameters.Request()
        req.parameters = [
            _make_param('kp', kp),
            _make_param('ki', ki),
            _make_param('kd', kd),
        ]
        future = self._pid_client.call_async(req)
        # Esperar sin bloquear el executor — usar timeout
        deadline = time.time() + 3.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if future.done() and future.result() is not None:
            ok = all(r.successful for r in future.result().results)
            if not ok:
                self.get_logger().error('set_parameters rechazó las ganancias')
        else:
            self.get_logger().error('set_parameters timeout')

    # ── Navegación ────────────────────────────────────────────────────────────

    def navigate_zigzag(self) -> tuple[bool, float]:
        """
        Envía los waypoints 2-4 a Nav2 y espera el resultado.
        Retorna (éxito, tiempo_segundos).
        """
        while not self._nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Esperando Nav2...')

        goal = NavigateThroughPoses.Goal()

        # WP1 es el inicio (spawn) — Nav2 recibe WP2, WP3, WP4
        for wp in self._waypoints[1:]:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(wp['x'])
            pose.pose.position.y = float(wp['y'])
            qx, qy, qz, qw = self._yaw_to_quat(float(wp['yaw']))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            goal.poses.append(pose)

        # Activar medición ITAE hacia el último waypoint
        last_wp = self._waypoints[-1]
        self._start_measuring(float(last_wp['x']), float(last_wp['y']))

        t_start = time.time()
        send_future = self._nav_client.send_goal_async(goal)

        # Esperar aceptación sin bloquear executor
        deadline = time.time() + 5.0
        while not send_future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not send_future.done() or send_future.result() is None:
            self._stop_measuring()
            return False, 0.0

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rechazó la ruta')
            self._stop_measuring()
            return False, 0.0

        # Esperar resultado (timeout generoso: 120s)
        result_future = goal_handle.get_result_async()
        deadline = time.time() + 120.0
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.1)

        elapsed = time.time() - t_start
        itae = self._stop_measuring()

        if not result_future.done():
            self.get_logger().warn('Nav2 timeout')
            return False, elapsed

        status = result_future.result().status
        success = (status == GoalStatus.STATUS_SUCCEEDED)
        if success:
            self.get_logger().info(f'Ruta completada — t={elapsed:.1f}s ITAE={itae:.3f}')
        else:
            self.get_logger().warn(f'Ruta fallida — status={status}')

        return success, elapsed

    # ── Función de fitness ────────────────────────────────────────────────────

    def evaluate(self, individual) -> tuple:
        """
        Función de fitness para DEAP.
        Cromosoma: [Kp, Ki, Kd]
        Retorna: (costo,)  — menor es mejor.

        Costo = w1 * tiempo_normalizado + w2 * ITAE + penalización_colisión
        """
        kp, ki, kd = individual
        self.get_logger().info(
            f'Evaluando — Kp={kp:.3f} Ki={ki:.3f} Kd={kd:.3f}'
        )

        # 1. Reset
        self.reset_robot()
        time.sleep(1.5)  # esperar que Gazebo aplique el teleport

        # 2. Inyectar ganancias
        self.set_pid(kp, ki, kd)
        time.sleep(0.3)

        # 3. Navegar y medir
        success, elapsed = self.navigate_zigzag()

        # 4. Calcular costo
        with self._odom_lock:
            itae = self._itae_accum

        if success:
            # Normalización: tiempo_ref=30s, itae_ref=1.0
            w_time = 0.4
            w_itae = 0.6
            cost = w_time * (elapsed / 30.0) + w_itae * itae
        else:
            cost = PENALTY

        self.get_logger().info(f'Costo = {cost:.4f}')
        return (cost,)


# ── Bounds para operadores DEAP ────────────────────────────────────────────────

def check_bounds(func):
    """Decorador que mantiene Kp, Ki, Kd dentro de sus rangos."""
    def wrapper(*args, **kwargs):
        offspring = func(*args, **kwargs)
        bounds = [KP_RANGE, KI_RANGE, KD_RANGE]
        for child in offspring:
            for i, (lo, hi) in enumerate(bounds):
                child[i] = float(max(lo, min(hi, child[i])))
        return offspring
    return wrapper


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AGArenaEvaluator()

    # Executor en hilo separado — el AG corre en el hilo principal
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        # ── Configurar DEAP ───────────────────────────────────────────────────
        toolbox = base.Toolbox()

        toolbox.register('kp',  random.uniform, *KP_RANGE)
        toolbox.register('ki',  random.uniform, *KI_RANGE)
        toolbox.register('kd',  random.uniform, *KD_RANGE)

        toolbox.register(
            'individual', tools.initCycle, creator.Individual,
            (toolbox.kp, toolbox.ki, toolbox.kd), n=1
        )
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('evaluate', node.evaluate)
        toolbox.register('mate',     tools.cxBlend, alpha=0.5)
        toolbox.register('mutate',   tools.mutGaussian, mu=0, sigma=0.5, indpb=0.3)
        toolbox.register('select',   tools.selTournament, tournsize=3)

        toolbox.decorate('mate',   check_bounds)
        toolbox.decorate('mutate', check_bounds)

        # Estadísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register('min',  min)
        stats.register('mean', lambda x: sum(x) / len(x))
        stats.register('max',  max)

        hof = tools.HallOfFame(3)  # guardar los 3 mejores

        # ── Evolución ─────────────────────────────────────────────────────────
        node.get_logger().info(
            f'Iniciando AG — pop={POP_SIZE} gen={N_GEN}'
        )
        pop = toolbox.population(n=POP_SIZE)

        pop, log = algorithms.eaSimple(
            pop, toolbox,
            cxpb=CX_PROB, mutpb=MUT_PROB,
            ngen=N_GEN,
            stats=stats, halloffame=hof,
            verbose=True
        )

        # ── Resultados ────────────────────────────────────────────────────────
        best = hof[0]
        node.get_logger().info('\n' + '='*50)
        node.get_logger().info('MEJOR INDIVIDUO ENCONTRADO:')
        node.get_logger().info(f'  Kp = {best[0]:.6f}')
        node.get_logger().info(f'  Ki = {best[1]:.6f}')
        node.get_logger().info(f'  Kd = {best[2]:.6f}')
        node.get_logger().info(f'  Costo = {best.fitness.values[0]:.6f}')
        node.get_logger().info('='*50)

        # Guardar resultados
        results = {
            'best': {'kp': best[0], 'ki': best[1], 'kd': best[2],
                     'cost': best.fitness.values[0]},
            'hall_of_fame': [
                {'kp': ind[0], 'ki': ind[1], 'kd': ind[2],
                 'cost': ind.fitness.values[0]}
                for ind in hof
            ],
            'log': str(log)
        }
        import json
        out_path = '/tmp/ga_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        node.get_logger().info(f'Resultados guardados en {out_path}')

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()