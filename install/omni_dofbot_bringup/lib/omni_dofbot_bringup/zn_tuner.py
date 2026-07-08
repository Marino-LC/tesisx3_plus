#!/usr/bin/env python3
"""
zn_tuner.py
===========
Sintonización de ganancias PID (Kp, Ki, Kd) del nodo mecanum_kinematic_node.py
mediante el método clásico de Ziegler-Nichols en lazo cerrado (método de la
ganancia última Ku / periodo último Tu, oscilación sostenida), como
contraparte metodológica clásica frente al Algoritmo Genético implementado
en ag_motion_tests.py.

DISEÑO
------
Este nodo NO reimplementa la batería de pruebas ni la infraestructura ROS 2:
hereda directamente de AGMotionEvaluator (definido en ag_motion_tests.py) y
reutiliza:
  - _teleport() / _reset_odometry() / _reset_controller()  (mismo protocolo
    de reseteo, con el orden ya corregido: reset ANTES de fijar origen).
  - _start_arm() / _stop_arm()  (misma coreografía de brazo "pick & place
    lado a lado" usada como perturbación dinámica de masa en el AG).
  - _drive() / _rotate()  (mismas primitivas de movimiento, mismo SegmentLog).
  - evaluate()  (misma función de costo P1+P2+P3, para poder comparar
    directamente el resultado de ZN contra el mejor individuo del AG).

De este modo, el modelo de motor de primer orden (FirstOrderLag, tau≈0.46 s,
identificado experimentalmente) queda incluido de forma transparente: es
parte de mecanum_kinematic_node.py y por lo tanto está presente en TODOS los
ensayos de escalón de este script, sin necesidad de duplicar el filtro aquí.
Antes de iniciar la búsqueda se fija explícitamente motor_tau para dejar
constancia de qué planta se está caracterizando (bridging teoría-implementación
señalado como pendiente en la tesis).

MÉTODO
------
1. Búsqueda gruesa: con Ki=Kd=0, se incrementa Kp multiplicativamente y se
   excita el lazo con un escalón de velocidad frontal (misma distancia seguridad
   que P1 del AG). Se analiza la señal de error de vx (vx_real - vx_ref) en
   busca de oscilaciones:
       - decreciente  -> lazo estable, subir Kp
       - sostenida    -> ¡encontrado! Ku = Kp actual, Tu = periodo medido
       - creciente / overshoot -> lazo inestable, se acotó el intervalo
2. Bisección entre el último Kp estable y el primer Kp inestable hasta
   converger a una oscilación sostenida dentro de una tolerancia.
3. Con (Ku, Tu) se calculan las ganancias de la tabla clásica de
   Ziegler-Nichols (P, PI, PID clásico, PID sin sobreimpulso).
4. Se valida el PID clásico resultante corriendo evaluate() (P1+P2+P3),
   exactamente la misma métrica que usa el AG, para comparación directa.

SEGURIDAD
---------
El Kp máximo está acotado a ag_motion_tests.KP_RANGE[1] (mismo límite físico
ya validado en el AG) y _drive() ya corta el movimiento por overshoot,
así que la búsqueda no puede sacar al robot de la arena de forma descontrolada.

EJECUCIÓN
---------
  # T1 — simulación (idéntico a ag_motion_tests.py)
  ros2 launch omni_dofbot_bringup omni_dofbot_controller.launch.py
  # T2 — cinemática + PID + modelo de motor
  ros2 run omni_dofbot_bringup mecanum_kinematic_node.py
  # T3 — odometría
  ros2 run omni_dofbot_bringup mecanum_odometry_node.py
  # T4 — este nodo
  ros2 run omni_dofbot_bringup zn_tuner.py

NOTA: no correr este nodo al mismo tiempo que ag_motion_tests.py — ambos
heredan el mismo nombre de nodo ROS 2 ("ag_motion_evaluator") porque ZNTuner
reutiliza el __init__ de AGMotionEvaluator tal cual. Se ejecutan de forma
secuencial (primero uno, luego el otro) para comparar resultados, tal como
está previsto en el capítulo de comparación AG vs ZN de la tesis.
"""

import os
import json
import math
import time
import statistics
import importlib.util
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

# ══════════════════════════════════════════════════════════════════════════════
# Carga de ag_motion_tests.py como módulo, sin depender de que sea importable
# como paquete Python (ambos scripts se instalan como ejecutables en
# lib/omni_dofbot_bringup/, no como módulos del paquete Python).
# ══════════════════════════════════════════════════════════════════════════════
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AG_PATH = os.path.join(_THIS_DIR, "ag_motion_tests.py")

_spec = importlib.util.spec_from_file_location("ag_motion_tests", _AG_PATH)
ag_motion_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ag_motion_tests)

AGMotionEvaluator = ag_motion_tests.AGMotionEvaluator
SegmentLog        = ag_motion_tests.SegmentLog

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA BÚSQUEDA ZN
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_TAU_DEFAULT = 0.46                       # s — identificado con MATLAB tfest

ZN_KP_START     = 0.5
ZN_KP_MAX       = ag_motion_tests.KP_RANGE[1]  # mismo límite físico que el AG (20.0)
ZN_KP_GROWTH    = 1.4                          # factor multiplicativo, búsqueda gruesa
ZN_BISECT_TOL   = 0.03                         # tolerancia relativa final (hi-lo)/hi
ZN_MAX_BISECT   = 12

ZN_STEP_DIST    = ag_motion_tests.DIST_X       # mismo tramo seguro que P1 (0.80 m)
ZN_STEP_TIMEOUT = 8.0                          # s — más largo que TIMEOUT_MOVE del AG
                                                # para capturar varios ciclos de oscilación

OSC_MIN_EXTREMA      = 5      # mínimo de extremos locales para confiar en el análisis
OSC_SKIP_FRACTION    = 0.25   # se descarta el primer 25% (transitorio de arranque)
OSC_RATIO_SUSTAINED  = (0.85, 1.15)   # razón amplitud[i+1]/amplitud[i] "sostenida"
OSC_RATIO_GROWING    = 1.15           # por encima de esto -> creciente/inestable

OUT_JSON = "zn_results.json"
OUT_PNG  = "zn_results.png"


# ══════════════════════════════════════════════════════════════════════════════
# Estructuras de resultado
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class OscillationResult:
    kp: float
    status: str                       # "decaying" | "sustained" | "growing" | "insufficient"
    ratio: Optional[float] = None
    tu: Optional[float] = None
    n_extrema: int = 0
    ok: bool = True

    def as_dict(self) -> dict:
        return {"kp": self.kp, "status": self.status, "ratio": self.ratio,
                "tu": self.tu, "n_extrema": self.n_extrema, "ok": self.ok}


# ══════════════════════════════════════════════════════════════════════════════
# Nodo ZN — hereda toda la infraestructura del AG
# ══════════════════════════════════════════════════════════════════════════════
class ZNTuner(AGMotionEvaluator):

    def __init__(self):
        super().__init__()   # reutiliza __init__ de AGMotionEvaluator tal cual
        self._search_history: List[OscillationResult] = []
        self.get_logger().info("ZNTuner listo (hereda de AGMotionEvaluator).")

    # ── Fijar motor_tau explícitamente antes de caracterizar la planta ───────
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
        self.get_logger().info(
            f"motor_tau fijado a {tau:.4f}s — la planta caracterizada por ZN "
            f"incluye el retraso electromecánico identificado del motor.")

    # ── Ensayo de escalón con Ki=Kd=0 ────────────────────────────────────────
    def _step_test(self, kp: float) -> Tuple[SegmentLog, bool]:
        """Aplica un escalón de velocidad frontal con ganancia proporcional
        pura (kp) y devuelve el SegmentLog (vx_ref/vx_real vs t) más el flag
        `ok` de _drive() (False si hubo timeout o frenado por overshoot)."""
        self._set_pid(kp, 0.0, 0.0)
        self._teleport()
        self._start_arm()   # misma perturbación dinámica de masa que el AG

        _itae, _elapsed, ok, seg = self._drive(
            ZN_STEP_DIST, "x",
            vx=+ag_motion_tests.VX_REF,
            timeout=ZN_STEP_TIMEOUT,
            seg_name=f"ZN_kp_{kp:.3f}")

        self._stop_arm()
        return seg, ok

    # ── Detección de picos (extremos locales) sin dependencias externas ─────
    @staticmethod
    def _find_extrema(t: List[float], x: List[float]) -> List[Tuple[float, float]]:
        extrema = []
        for i in range(1, len(x) - 1):
            if (x[i] - x[i - 1]) * (x[i + 1] - x[i]) < 0:
                extrema.append((t[i], x[i]))
        return extrema

    def _analyze_oscillation(self, seg: SegmentLog, kp: float, ok: bool) -> OscillationResult:
        if not seg.t or len(seg.t) != len(seg.vx_real) or len(seg.t) < 10:
            return OscillationResult(kp=kp, status="insufficient", ok=ok)

        n_skip = int(len(seg.t) * OSC_SKIP_FRACTION)
        t_trim  = seg.t[n_skip:]
        err_trim = [vr - vref for vr, vref in
                    zip(seg.vx_real[n_skip:], seg.vx_ref[n_skip:])]

        extrema = self._find_extrema(t_trim, err_trim)

        # Frenado por overshoot o timeout con poca oscilación -> tratar como
        # inestable directamente (evita confundir un corte de seguridad con
        # un lazo bien comportado).
        if not ok and len(extrema) < OSC_MIN_EXTREMA:
            return OscillationResult(kp=kp, status="growing", ok=ok,
                                      n_extrema=len(extrema))

        if len(extrema) < OSC_MIN_EXTREMA:
            return OscillationResult(kp=kp, status="insufficient", ok=ok,
                                      n_extrema=len(extrema))

        times = [e[0] for e in extrema]
        mags  = [abs(e[1]) for e in extrema]

        # Extremos del mismo tipo (mismo signo) están separados de a 2
        same_type_mags = mags[0::2]
        if len(same_type_mags) < 2:
            return OscillationResult(kp=kp, status="insufficient", ok=ok,
                                      n_extrema=len(extrema))

        ratios = []
        for i in range(len(same_type_mags) - 1):
            prev = same_type_mags[i]
            nxt  = same_type_mags[i + 1]
            if prev > 1e-6:
                ratios.append(nxt / prev)
        if not ratios:
            return OscillationResult(kp=kp, status="insufficient", ok=ok,
                                      n_extrema=len(extrema))
        mean_ratio = statistics.mean(ratios)

        # Periodo completo = tiempo entre dos extremos del mismo tipo
        periods = [times[i + 2] - times[i] for i in range(len(times) - 2)]
        tu = statistics.mean(periods) if periods else None

        if not ok:
            status = "growing"
        elif mean_ratio > OSC_RATIO_GROWING:
            status = "growing"
        elif OSC_RATIO_SUSTAINED[0] <= mean_ratio <= OSC_RATIO_SUSTAINED[1]:
            status = "sustained"
        else:
            status = "decaying"

        return OscillationResult(kp=kp, status=status, ratio=mean_ratio,
                                  tu=tu, n_extrema=len(extrema), ok=ok)

    # ── Búsqueda de Ku, Tu ────────────────────────────────────────────────────
    def find_ultimate_gain(self) -> Tuple[Optional[float], Optional[float]]:
        kp = ZN_KP_START
        last_stable_kp = 0.0
        last_unstable_kp = None

        # Fase 1 — búsqueda gruesa
        while kp <= ZN_KP_MAX:
            seg, ok = self._step_test(kp)
            res = self._analyze_oscillation(seg, kp, ok)
            self._search_history.append(res)
            self.get_logger().info(
                f"[ZN][gruesa] Kp={kp:.3f} -> {res.status} "
                f"(ratio={res.ratio}, Tu={res.tu}, extrema={res.n_extrema})")

            if res.status == "sustained":
                return kp, res.tu
            elif res.status == "growing":
                last_unstable_kp = kp
                break
            else:
                last_stable_kp = kp
                kp *= ZN_KP_GROWTH

        if last_unstable_kp is None:
            self.get_logger().error(
                f"No se alcanzó inestabilidad hasta Kp={ZN_KP_MAX:.2f}. "
                f"Aumenta ZN_KP_MAX o revisa la planta.")
            return None, None

        # Fase 2 — bisección
        lo, hi = last_stable_kp, last_unstable_kp
        for it in range(ZN_MAX_BISECT):
            mid = 0.5 * (lo + hi)
            seg, ok = self._step_test(mid)
            res = self._analyze_oscillation(seg, mid, ok)
            self._search_history.append(res)
            self.get_logger().info(
                f"[ZN][bisección {it}] Kp={mid:.3f} -> {res.status} "
                f"(ratio={res.ratio}, Tu={res.tu})")

            if res.status == "sustained":
                return mid, res.tu
            elif res.status == "growing":
                hi = mid
            else:
                lo = mid

            if hi > 0 and (hi - lo) / hi < ZN_BISECT_TOL:
                break

        # Última aproximación disponible (puede no ser una oscilación
        # perfectamente sostenida, pero es la mejor estimación alcanzada).
        seg, ok = self._step_test(hi)
        res = self._analyze_oscillation(seg, hi, ok)
        self._search_history.append(res)
        if res.tu is None:
            self.get_logger().warn(
                "No se pudo estimar Tu con precisión al cerrar la bisección; "
                "el resultado de Ku/Tu es aproximado.")
        return hi, res.tu

    # ── Tabla clásica de Ziegler-Nichols (lazo cerrado) ─────────────────────
    @staticmethod
    def compute_zn_table(ku: float, tu: float) -> Dict[str, Tuple[float, float, float]]:
        """Devuelve {nombre: (Kp, Ki, Kd)} según la tabla clásica de ZN.
        Ki = Kp/Ti ; Kd = Kp*Td (consistente con la forma paralela que
        implementa PIDController en mecanum_kinematic_node.py: u = Kp*e +
        Ki*integral(e) + Kd*derivative(e))."""
        table = {}

        # P
        table["P"] = (0.5 * ku, 0.0, 0.0)

        # PI
        kp_pi = 0.45 * ku
        ti_pi = tu / 1.2
        table["PI"] = (kp_pi, kp_pi / ti_pi, 0.0)

        # PID clásico
        kp_pid = 0.6 * ku
        ti_pid = tu / 2.0
        td_pid = tu / 8.0
        table["PID_clasico"] = (kp_pid, kp_pid / ti_pid, kp_pid * td_pid)

        # PID "sin sobreimpulso" (variante conservadora, útil para el
        # comparativo de robustez frente al AG)
        kp_no = 0.2 * ku
        ti_no = tu / 2.0
        td_no = tu / 3.0
        table["PID_sin_sobreimpulso"] = (kp_no, kp_no / ti_no, kp_no * td_no)

        return table

    # ── Gráfica simple de la búsqueda (Kp vs razón de amplitud) ─────────────
    def build_search_plot(self, ku: Optional[float], tu: Optional[float]):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[plot] matplotlib no disponible — omitiendo PNG")
            return

        kps    = [r.kp for r in self._search_history]
        ratios = [r.ratio if r.ratio is not None else float("nan")
                  for r in self._search_history]
        colors = {"decaying": "tab:blue", "sustained": "tab:green",
                  "growing": "tab:red", "insufficient": "tab:gray"}
        point_colors = [colors.get(r.status, "black") for r in self._search_history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(kps, ratios, c=point_colors, s=60, zorder=3)
        for r in self._search_history:
            ax.annotate(r.status[:3], (r.kp, r.ratio if r.ratio else 0),
                        fontsize=7, alpha=0.7)
        ax.axhspan(OSC_RATIO_SUSTAINED[0], OSC_RATIO_SUSTAINED[1],
                   color="green", alpha=0.10, label="banda sostenida")
        ax.axhline(1.0, color="black", ls=":", lw=1)
        if ku is not None:
            ax.axvline(ku, color="green", ls="--", lw=1.5,
                       label=f"Ku={ku:.3f}")
        title = "Búsqueda de ganancia última (Ziegler-Nichols)"
        if tu is not None:
            title += f"  —  Tu={tu:.3f}s"
        ax.set_title(title)
        ax.set_xlabel("Kp probado")
        ax.set_ylabel("Razón de amplitud entre picos consecutivos")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] PNG guardado en {os.path.abspath(OUT_PNG)}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = ZNTuner()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        node.set_motor_tau(MOTOR_TAU_DEFAULT)
        time.sleep(0.3)

        node.get_logger().info("Iniciando búsqueda de ganancia última (Ziegler-Nichols)...")
        ku, tu = node.find_ultimate_gain()

        if ku is None or tu is None:
            node.get_logger().error("No fue posible determinar Ku/Tu. Abortando.")
            return

        node.get_logger().info("=" * 54)
        node.get_logger().info(f"Ku = {ku:.4f}   Tu = {tu:.4f} s")
        node.get_logger().info("=" * 54)

        gains_table = node.compute_zn_table(ku, tu)
        for name, (kp, ki, kd) in gains_table.items():
            node.get_logger().info(f"[{name}] Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}")

        # Validar el PID clásico con la misma métrica del AG (P1+P2+P3)
        kp_c, ki_c, kd_c = gains_table["PID_clasico"]
        node.get_logger().info(
            "Validando ganancias ZN (PID clásico) con la batería P1/P2/P3 "
            "del AG para comparación directa...")
        fitness = node.evaluate([kp_c, ki_c, kd_c])[0]
        node.get_logger().info(f"Fitness ZN (comparable con AG) = {fitness:.5f}")

        results = {
            "motor_tau": MOTOR_TAU_DEFAULT,
            "ku": ku,
            "tu": tu,
            "gains_table": {
                name: {"kp": kp, "ki": ki, "kd": kd}
                for name, (kp, ki, kd) in gains_table.items()
            },
            "pid_clasico_fitness_ag_metric": fitness,
            "search_history": [r.as_dict() for r in node._search_history],
        }

        json_path = os.path.abspath(OUT_JSON)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        node.get_logger().info(f"Resultados guardados en {json_path}")

        node.build_search_plot(ku, tu)

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()