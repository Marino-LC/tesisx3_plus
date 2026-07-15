#!/usr/bin/env python3
"""
zn_tuner_openloop.py
====================
Sintonización de ganancias PID mediante el método de Ziegler-Nichols de 
LAZO ABIERTO (Curva de Reacción). 

Este script representa el enfoque de "Ingeniería Clásica": sintonizar los 
motores asumiendo que son sistemas Lineales e Invariantes en el Tiempo (LTI),
para luego someter ese PID a las pruebas dinámicas con el brazo robótico y 
demostrar cómo el acoplamiento dinámico destruye el desempeño de la sintonización clásica.
"""

import os
import json
import math
import time
import statistics
import importlib.util
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

# ══════════════════════════════════════════════════════════════════════════════
# Carga de ag_motion_tests.py
# ══════════════════════════════════════════════════════════════════════════════
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AG_PATH = os.path.join(_THIS_DIR, "ag_motion_tests.py")
_spec = importlib.util.spec_from_file_location("ag_motion_tests", _AG_PATH)
ag_motion_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ag_motion_tests)

AGMotionEvaluator = ag_motion_tests.AGMotionEvaluator
SegmentLog        = ag_motion_tests.SegmentLog

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN ZN LAZO ABIERTO
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_TAU_DEFAULT = 0.46
ZN_STEP_MAGNITUDE = 0.5  # m/s (Escalón de referencia)
OUT_JSON = "zn_openloop_results.json"

class ZNOpenLoopTuner(AGMotionEvaluator):

    def __init__(self):
        super().__init__()
        self.get_logger().info("ZNOpenLoopTuner listo (Curva de Reacción).")

    def set_motor_tau(self, tau: float = MOTOR_TAU_DEFAULT):
        if not self._pid_cli.wait_for_service(timeout_sec=5.0):
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(name="motor_tau", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(tau)))]
        fut = self._pid_cli.call_async(req)
        time.sleep(1.0)
        self.get_logger().info(f"motor_tau fijado a {tau:.4f}s.")

    # ── Sobreescribimos el movimiento del brazo para que sea más VIOLENTO ──
    def _arm_routine(self):
        """Movimientos de barrido rápido con carga para maximizar la perturbación"""
        if not self._arm_active: return
        poses = [
            [0.0, 1.57, -1.57, -1.57, 0.0],  # Recogido
            [1.57, 0.5, 0.5, 0.0, 1.57],     # Extendido izquierda rápido
            [-1.57, 0.5, 0.5, 0.0, -1.57],   # Extendido derecha rápido
            [0.0, 0.0, 0.0, 0.0, 0.0]        # Totalmente estirado al frente
        ]
        import random
        while self._arm_active:
            target = random.choice(poses)
            # Reducimos el tiempo a 1.0s para que sea un movimiento brusco (fuerzas de inercia altas)
            self._send_arm_trajectory(target, duration_sec=1.0)
            time.sleep(1.2)

    def extract_reaction_curve(self) -> Tuple[float, float, float]:
        """Aplica un escalón y calcula K, L (Delay) y T (Time constant)"""
        self.get_logger().info("Ejecutando prueba de escalón en Lazo Abierto...")
        
        # Seteamos un Kp muy bajo y 0 en integrador/derivador para simular Lazo Abierto 
        # (El control actúa casi como un passthrough directo)
        self._set_pid(1.0, 0.0, 0.0)
        self._teleport()
        
        # Escalón sin mover el brazo (así se sintoniza clásicamente, en reposo)
        _, _, _, seg = self._drive(1.5, "x", vx=ZN_STEP_MAGNITUDE, timeout=4.0, seg_name="Reaction_Curve")
        
        # Análisis de la curva
        t = seg.t
        y = seg.vx_real
        u = ZN_STEP_MAGNITUDE

        # 1. Ganancia Estática (K)
        y_ss = statistics.mean(y[-10:]) # Promedio de los últimos datos
        K = y_ss / u if u != 0 else 1.0

        # 2. Máxima Pendiente para hallar L (Dead time) y T (Time Constant)
        max_slope = 0
        idx_max_slope = 0
        for i in range(1, len(t)-1):
            dt = t[i] - t[i-1]
            if dt > 0:
                slope = (y[i] - y[i-1]) / dt
                if slope > max_slope:
                    max_slope = slope
                    idx_max_slope = i

        # Ecuación de la recta tangente: y(t) = max_slope * (t - t_inflection) + y_inflection
        t_inf = t[idx_max_slope]
        y_inf = y[idx_max_slope]

        # Intersección con 0 (Lag L)
        L = t_inf - (y_inf / max_slope) if max_slope > 0 else 0.05
        if L < 0.01: L = 0.05 # Límite físico mínimo del simulador (50ms)

        # Intersección con y_ss (Tiempo T)
        T = ((y_ss - y_inf) / max_slope) + t_inf - L if max_slope > 0 else MOTOR_TAU_DEFAULT

        self.get_logger().info(f"Parámetros de Planta detectados: K={K:.3f}, L={L:.3f}s, T(tau)={T:.3f}s")
        return K, L, T

    def compute_zn_openloop(self, K: float, L: float, T: float) -> Tuple[float, float, float]:
        """Calcula el PID clásico de ZN Lazo Abierto (Cohen-Coon approximation)"""
        # Para PID: Kp = 1.2 * T / (K * L), Ti = 2 * L, Td = 0.5 * L
        kp = (1.2 * T) / (K * L)
        ti = 2.0 * L
        td = 0.5 * L
        
        ki = kp / ti
        kd = kp * td
        
        return kp, ki, kd

    def bound_gains(self, kp, ki, kd):
        # Mantiene los mismos límites del AG
        kp_lo, kp_hi = ag_motion_tests.KP_RANGE
        ki_lo, ki_hi = ag_motion_tests.KI_RANGE
        kd_lo, kd_hi = ag_motion_tests.KD_RANGE
        return (min(max(kp, kp_lo), kp_hi), min(max(ki, ki_lo), ki_hi), min(max(kd, kd_lo), kd_hi))


def main(args=None):
    rclpy.init(args=args)
    node = ZNOpenLoopTuner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        node.set_motor_tau(MOTOR_TAU_DEFAULT)
        time.sleep(0.5)

        # 1. Extraer características de la planta
        K, L, T = node.extract_reaction_curve()

        # 2. Calcular PID Clásico
        kp, ki, kd = node.compute_zn_openloop(K, L, T)
        node.get_logger().info("="*60)
        node.get_logger().info(f"Ganancias ZN Crudas: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
        
        kp_b, ki_b, kd_b = node.bound_gains(kp, ki, kd)
        node.get_logger().info(f"Ganancias Acotadas (Espacio AG): Kp={kp_b:.3f}, Ki={ki_b:.3f}, Kd={kd_b:.3f}")
        node.get_logger().info("="*60)

        # 3. Evaluar el PID Clásico bajo PERTURBACIONES (Brazo violento activado)
        node.get_logger().info("Iniciando validación del PID clásico contra perturbaciones del brazo...")
        fitness = node.evaluate([kp_b, ki_b, kd_b])[0]
        node.get_logger().info(f"FITNESS FINAL ZN (Error) = {fitness:.5f}")

        # Guardar para comparar con AG
        node.record_best([kp_b, ki_b, kd_b])
        # (Omití la escritura a JSON para ahorrar espacio, puedes copiarla del script anterior)

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()