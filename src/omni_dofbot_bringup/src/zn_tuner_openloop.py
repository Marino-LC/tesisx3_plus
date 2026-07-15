#!/usr/bin/env python3
"""
zn_tuner_openloop.py
====================
Sintonización de ganancias PID mediante el método de Ziegler-Nichols de
LAZO ABIERTO (curva de reacción, Ziegler & Nichols 1942, "método 2"),
como contraparte de la variante de lazo cerrado implementada en zn_tuner.py.

Este script representa el enfoque de "ingeniería clásica": sintonizar los
motores asumiendo que son sistemas Lineales e Invariantes en el Tiempo (LTI),
para luego someter ese PID a las pruebas dinámicas con el brazo robótico y
demostrar cómo el acoplamiento dinámico degrada el desempeño de la
sintonización clásica frente al Algoritmo Genético.

══════════════════════════════════════════════════════════════════════════════
POR QUÉ SE REESCRIBIÓ LA IDENTIFICACIÓN DE K, L, T (bitácora para la tesis)
══════════════════════════════════════════════════════════════════════════════
La primera versión de este script estimaba L (retardo) y T (constante de
tiempo) con el método clásico de la "tangente en el punto de máxima
pendiente" aplicado directamente a la señal de vx_real simulada. Ese método
deriva la señal punto a punto (diferencia finita), lo que amplifica muchísimo
el ruido de muestreo de la odometría: una sola muestra ruidosa entre dos
lecturas puede generar una "pendiente máxima" artificialmente alta, y de ahí
una recta tangente casi vertical que colapsa L y T a valores absurdamente
pequeños. En una corrida real esto produjo T≈0.017s, sido que la constante de
tiempo del motor ya identificada y validada con MATLAB `tfest` (Sistema de
Identificación, ver Capítulo de identificación de la tesis) es τ≈0.46s — casi
30 veces mayor. Esa T ruidosa disparaba Ki a valores sin sentido físico.

SOLUCIÓN — Opción A (método principal, usado para las ganancias):
  No se re-estima T a partir de un escalón ruidoso: se usa directamente la
  τ ya validada con `tfest`. Tampoco se estima L con una tangente ruidosa: se
  usa el periodo de muestreo del lazo de control discreto (Ts = 1/control_rate,
  el mismo que ya sustenta la discretización ZOH documentada en la tesis) como
  el retardo efectivo L. Ninguno de los dos valores depende de una derivada
  numérica sobre una señal ruidosa; K sigue siendo el promedio de las últimas
  muestras del escalón (una media es mucho más robusta al ruido que una
  derivada).

VALIDACIÓN CRUZADA — Opción B (solo informativa, no se usa para las ganancias):
  Adicionalmente se calcula L y T con el método de los dos puntos de
  Sundaresan-Krishnaswamy (cruces al 28.3% y 63.2% de la respuesta final),
  que tampoco deriva la señal y es mucho más robusto al ruido que la tangente
  de máxima pendiente. Se imprime junto a los valores de la Opción A para que
  quede documentado en la tesis qué tan cerca (o no) coinciden ambos métodos.

NOTA IMPORTANTE incluso con T correcta:
  Con T≈0.46s y L≈Ts≈0.02s, la razón R=L/T≈0.04 es muchísimo menor que el
  rango en el que las reglas de Ziegler-Nichols (y también Cohen-Coon) son
  válidas (típicamente 0.1 ≲ L/T ≲ 1). Esto es estructural, no un error de
  medición: la planta identificada es esencialmente un retraso de primer
  orden puro, casi sin tiempo muerto real, y NINGUNA regla de sintonización
  basada en curva de reacción (ZN o Cohen-Coon) está bien planteada para ese
  caso — ambas dividen por L o por una potencia de R, así que R→0 dispara Kp
  y/o Ki sin límite. El script imprime una advertencia explícita cuando esto
  ocurre. La ganancia final igual se acota al mismo espacio admisible
  (KP_RANGE/KI_RANGE/KD_RANGE) que usa el AG, para que la comparación sea
  justa; ver `gains_raw` vs `gains_bounded` en el JSON de salida.

══════════════════════════════════════════════════════════════════════════════
OTRA CORRECCIÓN: perturbación de brazo "violenta" que nunca se ejecutaba
══════════════════════════════════════════════════════════════════════════════
La versión anterior sobreescribía un método `_arm_routine()` pensado para dar
un movimiento de brazo más brusco (mayor perturbación inercial) durante la
validación del PID. Pero `AGMotionEvaluator._start_arm()` lanza un hilo sobre
`_arm_loop`, no sobre `_arm_routine` — ese método nunca se llamaba, así que la
validación en realidad corría con la coreografía heredada por defecto
("pick & place lado a lado"), no con el barrido agresivo que se documentaba.
Esto no invalida la comparación contra el AG (ambos usan la misma
perturbación, vía el mismo `evaluate()` heredado), pero si en la tesis se
describe la perturbación como "un barrido violento" hay que corregir esa
afirmación o corregir el código. Aquí se corrige el código: se sobreescribe
`_arm_loop` (el hook correcto) con el movimiento agresivo.

EJECUCIÓN
  ros2 launch omni_dofbot_bringup omni_dofbot_controller.launch.py
  ros2 run omni_dofbot_bringup mecanum_kinematic_node.py
  ros2 run omni_dofbot_bringup mecanum_odometry_node.py
  ros2 run omni_dofbot_bringup zn_tuner_openloop.py
"""

import os
import json
import math
import time
import random
import statistics
import importlib.util
from typing import List, Optional, Tuple, Dict

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

# ══════════════════════════════════════════════════════════════════════════════
# Carga de ag_motion_tests.py (mismo mecanismo que zn_tuner.py, sin depender de
# que sea importable como paquete Python — ambos se instalan como ejecutables
# en lib/omni_dofbot_bringup/).
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
MOTOR_TAU_DEFAULT   = 0.46   # s — identificado y validado con MATLAB tfest
CONTROL_LOOP_RATE_HZ = 50.0  # Hz — control_rate por defecto de mecanum_kinematic_node.py
                             # (mismo periodo de muestreo que sustenta la
                             # discretización ZOH ya documentada en la tesis)

ZN_STEP_MAGNITUDE = 0.5      # m/s — escalón de referencia
ZN_STEP_DIST      = 1.5      # m   — distancia de seguridad del ensayo de escalón
ZN_STEP_TIMEOUT   = 4.0      # s

ZN_MIN_VALID_RATIO = 0.10    # R=L/T por debajo de esto: fuera del rango de
                             # validez habitual de las reglas ZN/Cohen-Coon
                             # (ver docstring del módulo)

SMOOTH_WINDOW = 5            # ventana de media móvil para suavizar vx_real
                             # antes de cualquier análisis (K, y validación B)

OUT_JSON = "zn_openloop_results.json"
OUT_PNG  = "zn_openloop_identification.png"


def _seg_to_dict(s: SegmentLog) -> dict:
    """Mismas llaves que usa ag_motion_tests.py / zn_tuner.py en su JSON de
    salida (best_run.test1/2/3), para postprocesar los tres JSON con el mismo
    script de MATLAB sin cambiar el parser."""
    return {"name": s.name, "t": s.t,
            "vx_ref": s.vx_ref, "vy_ref": s.vy_ref, "wz_ref": s.wz_ref,
            "vx_real": s.vx_real, "vy_real": s.vy_real, "wz_real": s.wz_real,
            "pos_err": s.pos_err,
            "x_ref": s.x_ref, "y_ref": s.y_ref, "yaw_ref": s.yaw_ref,
            "x_real": s.x_real, "y_real": s.y_real, "yaw_real": s.yaw_real}


class ZNOpenLoopTuner(AGMotionEvaluator):

    def __init__(self):
        super().__init__()
        self._last_identification: Dict = {}
        self.get_logger().info("ZNOpenLoopTuner listo (curva de reacción, tau validado + Ts=ZOH).")

    def set_motor_tau(self, tau: float = MOTOR_TAU_DEFAULT):
        if not self._pid_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("mecanum_kinematic_node no disponible (motor_tau).")
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name="motor_tau",
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                  double_value=float(tau)))]
        fut = self._pid_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 3.0:
            time.sleep(0.05)
        self.get_logger().info(f"motor_tau fijado a {tau:.4f}s.")

    # ── Perturbación de brazo agresiva — CORREGIDO ───────────────────────────
    # AGMotionEvaluator._start_arm() lanza un hilo sobre `_arm_loop`, no sobre
    # `_arm_routine` (nombre usado en la versión anterior, nunca invocado en
    # la práctica). Se sobreescribe aquí el hook correcto.
    def _arm_loop(self):
        """Barrido rápido y brusco del brazo (con carga), pensado para
        maximizar la perturbación inercial sobre la base durante la
        validación del PID clásico ZN — más agresivo que la coreografía
        "pick & place" heredada por defecto del AG."""
        self.get_logger().info(
            "[Brazo-ZN-openloop] hilo iniciado — barrido violento (perturbación inercial)")
        poses = [
            [0.0, 1.57, -1.57, -1.57, 0.0],   # recogido
            [1.57, 0.5, 0.5, 0.0, 1.57],       # extendido izquierda, rápido
            [-1.57, 0.5, 0.5, 0.0, -1.57],     # extendido derecha, rápido
            [0.0, 0.0, 0.0, 0.0, 0.0],         # totalmente estirado al frente
        ]
        while self._arm_active:
            target = random.choice(poses)
            self._send_arm(target, duration_sec=1)   # movimiento brusco: 1s
            if not self._wait_arm(1.2):
                break
        self._send_arm(ag_motion_tests.ARM_HOME)
        self.get_logger().info("[Brazo-ZN-openloop] hilo detenido → HOME")

    # ── Utilidades de procesamiento de señal ─────────────────────────────────
    @staticmethod
    def _smooth(x: List[float], window: int = SMOOTH_WINDOW) -> List[float]:
        """Media móvil simple, sin dependencias externas. Se usa para
        atenuar el ruido de muestreo antes de calcular K o los cruces de la
        validación B — nunca antes de una derivada (eso es justamente lo que
        se quiere evitar)."""
        if window <= 1 or len(x) < window:
            return list(x)
        half = window // 2
        n = len(x)
        out = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out.append(sum(x[lo:hi]) / (hi - lo))
        return out

    @staticmethod
    def _interp_time(t: List[float], y: List[float], i: int, target: float) -> float:
        """Interpola linealmente el instante en que la señal cruza `target`
        entre las muestras i-1 e i (evita el escalón discreto del muestreo)."""
        if i == 0:
            return t[0]
        t0, t1 = t[i - 1], t[i]
        y0, y1 = y[i - 1], y[i]
        if y1 == y0:
            return t1
        frac = (target - y0) / (y1 - y0)
        return t0 + frac * (t1 - t0)

    @staticmethod
    def _extract_via_max_slope(t: List[float], y: List[float], u: float) -> Tuple[float, float, float]:
        """Método LEGADO — tangente en el punto de máxima pendiente. Se deja
        disponible solo por trazabilidad/comparación; NO es el método por
        defecto porque amplifica el ruido de muestreo de la odometría (ver
        docstring del módulo)."""
        y_ss = statistics.mean(y[-10:])
        K = y_ss / u if u != 0 else 1.0

        max_slope = 0.0
        idx_max_slope = 0
        for i in range(1, len(t) - 1):
            dt = t[i] - t[i - 1]
            if dt > 0:
                slope = (y[i] - y[i - 1]) / dt
                if slope > max_slope:
                    max_slope = slope
                    idx_max_slope = i

        t_inf = t[idx_max_slope]
        y_inf = y[idx_max_slope]
        L = t_inf - (y_inf / max_slope) if max_slope > 0 else 0.05
        if L < 0.01:
            L = 0.05
        T = ((y_ss - y_inf) / max_slope) + t_inf - L if max_slope > 0 else MOTOR_TAU_DEFAULT
        return K, L, T

    def _sundaresan_krishnaswamy(self, t: List[float], y_smooth: List[float],
                                  y_ss: float) -> Tuple[Optional[float], Optional[float]]:
        """Validación cruzada (Opción B): método de los dos puntos de
        Sundaresan-Krishnaswamy. Usa los tiempos en que la respuesta cruza
        28.3% y 63.2% de su valor final — sin derivar la señal — y por eso
        es mucho más robusto al ruido que la tangente de máxima pendiente.
        Solo informativo: NO se usa para calcular las ganancias finales."""
        y28_target = 0.283 * y_ss
        y63_target = 0.632 * y_ss
        t28 = t63 = None
        for i in range(1, len(t)):
            if t28 is None and y_smooth[i] >= y28_target:
                t28 = self._interp_time(t, y_smooth, i, y28_target)
            if t63 is None and y_smooth[i] >= y63_target:
                t63 = self._interp_time(t, y_smooth, i, y63_target)
                break

        if t28 is None or t63 is None or t63 <= t28:
            return None, None

        T_b = 1.5 * (t63 - t28)
        L_b = max(t63 - T_b, 0.0)
        return L_b, T_b

    # ── Extracción de K, L, T ─────────────────────────────────────────────────
    def extract_reaction_curve(self, use_identified_tau: bool = True,
                                tau_identified: float = MOTOR_TAU_DEFAULT
                                ) -> Tuple[float, float, float]:
        """Aplica un escalón en lazo abierto y determina K, L, T.

        Opción A (use_identified_tau=True, RECOMENDADO — usada por defecto):
          T = tau_identified (ya validado con MATLAB tfest, ver Capítulo de
              identificación de la tesis; no se re-deriva de una señal
              ruidosa).
          L = 1/CONTROL_LOOP_RATE_HZ (el mismo periodo de muestreo que ya
              sustenta la discretización ZOH del lazo de control, no una
              tangente estimada sobre ruido).
          K = ganancia estática medida (promedio robusto de las últimas
              muestras del escalón, suavizado con media móvil).

        Opción B (informativa, siempre calculada y registrada, nunca usada
        para las ganancias): método de los dos puntos de
        Sundaresan-Krishnaswamy sobre la misma curva, para verificar qué tan
        cerca cae de la Opción A.

        Si use_identified_tau=False, se usa en su lugar el método legado de
        la tangente de máxima pendiente (no recomendado, se deja solo por
        completitud/comparación histórica).
        """
        self.get_logger().info("Ejecutando prueba de escalón en Lazo Abierto...")

        # Kp=1 puro (Ki=Kd=0): el lazo actúa casi como un passthrough directo,
        # tal como exige la identificación clásica en reposo (sin perturbación
        # de brazo — se activa solo después, en la validación con evaluate()).
        self._set_pid(1.0, 0.0, 0.0)
        self._teleport()

        _, _, _, seg = self._drive(
            ZN_STEP_DIST, "x", vx=ZN_STEP_MAGNITUDE,
            timeout=ZN_STEP_TIMEOUT, seg_name="Reaction_Curve")

        t = seg.t
        y_raw = seg.vx_real
        u = ZN_STEP_MAGNITUDE

        if len(t) < 10:
            self.get_logger().error(
                "Muy pocas muestras en la curva de reacción — no se puede "
                "identificar la planta con confianza. Usando valores por defecto.")
            self._last_identification = {"insufficient_data": True}
            return 1.0, 1.0 / CONTROL_LOOP_RATE_HZ, tau_identified

        y_smooth = self._smooth(y_raw, window=SMOOTH_WINDOW)
        y_ss = statistics.mean(y_smooth[-10:])
        K = y_ss / u if u != 0 else 1.0

        # ── Opción B — validación cruzada (no se usa para las ganancias) ────
        L_b, T_b = self._sundaresan_krishnaswamy(t, y_smooth, y_ss)
        if T_b is not None:
            self.get_logger().info(
                f"[Validación B — Sundaresan-Krishnaswamy] L={L_b:.4f}s  T={T_b:.4f}s")
        else:
            self.get_logger().warn(
                "[Validación B] No se pudieron determinar los cruces 28.3%/63.2% "
                "(señal insuficiente o demasiado ruidosa).")

        # ── Opción A — método principal (usado para las ganancias) ─────────
        if use_identified_tau:
            T = tau_identified
            L = 1.0 / CONTROL_LOOP_RATE_HZ
            self.get_logger().info(
                f"[Opción A — planta ya validada] T=tau_tfest={T:.4f}s, "
                f"L=Ts_control={L:.4f}s (no se re-estiman con la tangente de "
                f"máxima pendiente).")
        else:
            K, L, T = self._extract_via_max_slope(t, y_smooth, u)
            self.get_logger().warn(
                "Usando método LEGADO de tangente de máxima pendiente "
                "(sensible a ruido) por solicitud explícita.")

        ratio_R = L / T if T > 0 else float("inf")
        if ratio_R < ZN_MIN_VALID_RATIO:
            self.get_logger().warn(
                f"R=L/T={ratio_R:.3f} está muy por debajo del rango habitual "
                f"de validez de las reglas ZN/Cohen-Coon (≈0.1–1). La planta "
                f"identificada es casi un retraso de primer orden puro, sin "
                f"tiempo muerto significativo: NINGUNA regla de curva de "
                f"reacción está bien planteada aquí — es esperable que Kp y/o "
                f"Ki salgan desproporcionados y deban acotarse. Ver docstring "
                f"del módulo.")

        self.get_logger().info(
            f"Parámetros de planta usados: K={K:.3f}, L={L:.4f}s, T(tau)={T:.4f}s "
            f"(R=L/T={ratio_R:.3f})")

        self._last_identification = {
            "K": K, "L": L, "T": T, "R_ratio": ratio_R,
            "method": "tau_validado_tfest" if use_identified_tau else "tangente_max_pendiente_legado",
            "validation_b": {"L": L_b, "T": T_b},
            "step_response": {"t": t, "vx_real_raw": y_raw, "vx_real_smooth": y_smooth,
                               "vx_ref": u},
        }
        return K, L, T

    # ── Ganancias ZN de curva de reacción (Ziegler-Nichols 1942, método 2) ──
    def compute_zn_openloop(self, K: float, L: float, T: float) -> Tuple[float, float, float]:
        """Regla clásica de Ziegler-Nichols para curva de reacción (PID):
        Kp = 1.2*T/(K*L) ; Ti = 2*L ; Td = 0.5*L
        (Ziegler & Nichols, 1942 — NO es Cohen-Coon; Cohen-Coon (1953) usa
        coeficientes adicionales dependientes de L/T y, para R=L/T muy
        pequeño como el de esta planta, también diverge por la misma razón
        estructural — no se gana nada usándolo aquí)."""
        kp = (1.2 * T) / (K * L)
        ti = 2.0 * L
        td = 0.5 * L
        ki = kp / ti
        kd = kp * td
        return kp, ki, kd

    # ── Acotar ganancias al mismo espacio admisible que usa el AG ───────────
    def bound_gains(self, kp: float, ki: float, kd: float
                     ) -> Tuple[Tuple[float, float, float], Dict[str, bool]]:
        kp_lo, kp_hi = ag_motion_tests.KP_RANGE
        ki_lo, ki_hi = ag_motion_tests.KI_RANGE
        kd_lo, kd_hi = ag_motion_tests.KD_RANGE

        kp_b = float(max(kp_lo, min(kp_hi, kp)))
        ki_b = float(max(ki_lo, min(ki_hi, ki)))
        kd_b = float(max(kd_lo, min(kd_hi, kd)))

        clamped = {"kp": kp_b != kp, "ki": ki_b != ki, "kd": kd_b != kd}
        return (kp_b, ki_b, kd_b), clamped

    # ── Gráfica de identificación (figura para el capítulo de resultados) ──
    def build_identification_plot(self):
        info = self._last_identification
        if not info or "step_response" not in info:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[plot] matplotlib no disponible — omitiendo PNG de identificación")
            return

        t = info["step_response"]["t"]
        y_raw = info["step_response"]["vx_real_raw"]
        y_smooth = info["step_response"]["vx_real_smooth"]
        u = info["step_response"]["vx_ref"]
        K, L, T = info["K"], info["L"], info["T"]
        L_b, T_b = info["validation_b"]["L"], info["validation_b"]["T"]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(t, y_raw, color="lightgray", lw=1.0, label="vx_real (crudo)")
        ax.plot(t, y_smooth, color="tab:blue", lw=1.8, label="vx_real (suavizado)")
        ax.axhline(u, color="black", ls=":", lw=1.0, label=f"escalón u={u:.2f} m/s")
        ax.axhline(K * u, color="tab:blue", ls="--", lw=1.0, alpha=0.6,
                   label=f"K·u={K*u:.3f} m/s")

        # Modelo FOPDT de la Opción A: y(t) = K*u*(1-exp(-(t-L)/T)) para t>=L
        t_model = [ti for ti in t]
        y_model = [K * u * (1 - math.exp(-(ti - L) / T)) if ti >= L else 0.0
                   for ti in t_model]
        ax.plot(t_model, y_model, color="tab:green", lw=1.6, ls="-.",
                label=f"modelo FOPDT (Opción A): T={T:.3f}s, L={L:.3f}s")

        if T_b is not None and L_b is not None:
            y_model_b = [K * u * (1 - math.exp(-(ti - L_b) / T_b)) if ti >= L_b else 0.0
                         for ti in t_model]
            ax.plot(t_model, y_model_b, color="tab:red", lw=1.3, ls=":",
                    label=f"modelo FOPDT (Opción B, validación): T={T_b:.3f}s, L={L_b:.3f}s")

        ax.set_title("Identificación de planta — curva de reacción (lazo abierto)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("vx (m/s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] PNG de identificación guardado en {os.path.abspath(OUT_PNG)}")


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

        # 1. Identificar la planta (Opción A por defecto: tau validado + Ts=ZOH;
        #    Opción B calculada automáticamente como validación cruzada).
        K, L, T = node.extract_reaction_curve(use_identified_tau=True,
                                               tau_identified=MOTOR_TAU_DEFAULT)

        # 2. Calcular PID clásico (Ziegler-Nichols, curva de reacción)
        kp, ki, kd = node.compute_zn_openloop(K, L, T)
        node.get_logger().info("=" * 60)
        node.get_logger().info(f"Ganancias ZN crudas (sin acotar): Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}")

        (kp_b, ki_b, kd_b), clamped = node.bound_gains(kp, ki, kd)
        if any(clamped.values()):
            recortadas = [g.upper() for g, was in clamped.items() if was]
            node.get_logger().warn(
                f"Ganancia(s) {recortadas} recortada(s) al espacio admisible del AG "
                f"→ Kp={kp_b:.4f} Ki={ki_b:.4f} Kd={kd_b:.4f} "
                f"(cruda: Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f})")
        else:
            node.get_logger().info(
                f"Ganancias dentro del espacio admisible del AG, sin recorte: "
                f"Kp={kp_b:.4f} Ki={ki_b:.4f} Kd={kd_b:.4f}")
        node.get_logger().info("=" * 60)

        # 3. Evaluar el PID clásico (acotado) bajo perturbación del brazo,
        #    con la misma métrica P1+P2+P3 que usa el AG.
        node.get_logger().info(
            "Iniciando validación del PID clásico ZN contra perturbaciones del brazo...")
        fitness = node.evaluate([kp_b, ki_b, kd_b])[0]
        node.get_logger().info(f"FITNESS FINAL ZN (comparable con AG) = {fitness:.5f}")

        # 4. Grabar la corrida final — MISMAS series que exporta el AG para su
        #    mejor individuo (record_best se hereda de AGMotionEvaluator).
        segs1, segs2, segs3 = node.record_best([kp_b, ki_b, kd_b])

        results = {
            "method": "ziegler_nichols_openloop",
            "motor_tau": MOTOR_TAU_DEFAULT,
            "control_loop_rate_hz": CONTROL_LOOP_RATE_HZ,
            "identification": {
                "K": K, "L": L, "T": T, "R_ratio": L / T if T > 0 else None,
                "method": node._last_identification.get("method"),
                "validation_b_sundaresan_krishnaswamy": node._last_identification.get("validation_b"),
                "valid_range_warning": (L / T < ZN_MIN_VALID_RATIO) if T > 0 else True,
            },
            "config": {
                # Mismos campos/unidades que "config" en ag_results.json y
                # zn_results.json, para un único parser de MATLAB.
                "dist_x": ag_motion_tests.DIST_X,
                "dist_return": ag_motion_tests.DIST_RETURN,
                "rot_angle_deg": math.degrees(ag_motion_tests.ROT_ANGLE),
                "vx_ref": ag_motion_tests.VX_REF,
                "vy_ref": ag_motion_tests.VY_REF,
                "wz_ref": ag_motion_tests.WZ_REF,
                "weights": {"P1": ag_motion_tests.W1, "P2": ag_motion_tests.W2,
                            "P3": ag_motion_tests.W3},
                "note": "Ziegler-Nichols de curva de reacción (lazo abierto). "
                        "T tomado del tau ya validado con tfest; L tomado del "
                        "periodo de muestreo del lazo de control (Ts=1/control_rate, "
                        "consistente con la discretización ZOH ya documentada). "
                        "Ganancias acotadas al mismo espacio KP/KI/KD_RANGE del AG.",
            },
            "gains_raw": {"kp": kp, "ki": ki, "kd": kd},
            "gains_bounded": {"kp": kp_b, "ki": ki_b, "kd": kd_b},
            "bounds_used": {
                "KP_RANGE": list(ag_motion_tests.KP_RANGE),
                "KI_RANGE": list(ag_motion_tests.KI_RANGE),
                "KD_RANGE": list(ag_motion_tests.KD_RANGE),
            },
            "best": {"kp": kp_b, "ki": ki_b, "kd": kd_b, "fitness": fitness},
            "best_run": {
                "test1": [_seg_to_dict(s) for s in segs1],
                "test2": [_seg_to_dict(s) for s in segs2],
                "test3": [_seg_to_dict(s) for s in segs3],
            },
        }

        json_path = os.path.abspath(OUT_JSON)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        node.get_logger().info(f"Resultados guardados en {json_path}")

        node.build_identification_plot()

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()