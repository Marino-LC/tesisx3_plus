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
import webbrowser
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
OUT_HTML = "zn_results.html"

# ── Acotamiento de ganancias ──────────────────────────────────────────────────
# La receta clásica de ZN calcula Ki = Kp/Ti con Ti = Tu/2 (PID clásico) o
# Ti = Tu/1.2 (PI). Cuando la planta tiene un retraso de fase dominante — aquí,
# el filtro de motor de primer orden con tau≈0.46 s — la oscilación sostenida
# aparece con un periodo Tu relativamente corto, así que Ti sale pequeño y Ki
# se dispara. Esto es un defecto documentado del método ZN clásico frente a
# plantas con retraso (ver Åström & Hägglund, "PID Controllers", cap. 2-3):
# la regla fue derivada para dar un decaimiento de cuarto de amplitud, no para
# respetar límites de actuador, y típicamente requiere "detuning" manual del
# 30-50%. Para que la comparación contra el AG sea justa (mismo espacio de
# búsqueda admisible) las ganancias finales de ZN se acotan a los mismos
# rangos KP_RANGE / KI_RANGE / KD_RANGE que usa ag_motion_tests.py. El valor
# crudo (sin acotar) se conserva en el JSON de resultados para dejar
# constancia, en la tesis, de cuánto se salía la receta clásica del espacio
# físicamente razonable.


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


def _seg_to_dict(s: SegmentLog) -> dict:
    """Serializa un SegmentLog exactamente con las mismas llaves que usa
    ag_motion_tests.py en su JSON de salida (best_run.test1/2/3), para que
    el mismo script de post-procesamiento en MATLAB funcione sin cambios
    tanto para los resultados del AG como para los de ZN."""
    return {"name": s.name, "t": s.t,
            "vx_ref": s.vx_ref, "vy_ref": s.vy_ref, "wz_ref": s.wz_ref,
            "vx_real": s.vx_real, "vy_real": s.vy_real, "wz_real": s.wz_real,
            "pos_err": s.pos_err,
            "x_ref": s.x_ref, "y_ref": s.y_ref, "yaw_ref": s.yaw_ref,
            "x_real": s.x_real, "y_real": s.y_real, "yaw_real": s.yaw_real}


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

    # ── Acotar ganancias al mismo espacio admisible que usa el AG ───────────
    @staticmethod
    def bound_gains(kp: float, ki: float, kd: float) -> Tuple[Tuple[float, float, float], Dict[str, bool]]:
        """Satura (Kp, Ki, Kd) a KP_RANGE / KI_RANGE / KD_RANGE (los mismos
        límites que ag_motion_tests.py usa para el AG). Devuelve la terna
        acotada y un diccionario indicando qué componente(s) se recortaron,
        para poder reportarlo explícitamente en el log y en el JSON."""
        kp_lo, kp_hi = ag_motion_tests.KP_RANGE
        ki_lo, ki_hi = ag_motion_tests.KI_RANGE
        kd_lo, kd_hi = ag_motion_tests.KD_RANGE

        kp_b = float(max(kp_lo, min(kp_hi, kp)))
        ki_b = float(max(ki_lo, min(ki_hi, ki)))
        kd_b = float(max(kd_lo, min(kd_hi, kd)))

        clamped = {
            "kp": kp_b != kp,
            "ki": ki_b != ki,
            "kd": kd_b != kd,
        }
        return (kp_b, ki_b, kd_b), clamped

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

        # ── Punto de oscilación sostenida (Ku, Tu) — el resultado del método ──
        sustained_pt = next(
            (r for r in reversed(self._search_history) if r.status == "sustained"),
            None
        )
        if ku is not None:
            ax.axvline(ku, color="green", ls="--", lw=1.5, label=f"Ku={ku:.3f}")
        if sustained_pt is not None:
            ax.scatter([sustained_pt.kp], [sustained_pt.ratio],
                       s=220, facecolors="none", edgecolors="darkgreen",
                       linewidths=2.2, zorder=4)
            ax.annotate(
                f"Ku={sustained_pt.kp:.3f}\nTu={tu:.3f}s\n(oscilación sostenida)",
                xy=(sustained_pt.kp, sustained_pt.ratio),
                xytext=(0.65, 0.85), textcoords="axes fraction",
                fontsize=9, color="darkgreen", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))

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
        if sustained_pt is not None:
            print(f"[plot] Punto de oscilación sostenida resaltado en el PNG: "
                  f"Kp={sustained_pt.kp:.3f}, razón={sustained_pt.ratio:.3f}, "
                  f"extremos={sustained_pt.n_extrema}")


    # ── Gráficas de comparación directa con el AG ────────────────────────────
    # Reutiliza exactamente las mismas series (P1 vx, P2 wz, P3 error de
    # posición, pose x/y/yaw deseada-vs-obtenida) que ag_motion_tests.py
    # exporta para el mejor individuo del AG (ver _build_plots / record_best).
    # Aquí las llenamos con el resultado de record_best() aplicado a las
    # ganancias ZN ya acotadas, así ambos métodos quedan graficados con el
    # mismo formato y las mismas unidades — comparación "manzanas con
    # manzanas" para el capítulo de resultados de la tesis.
    def build_comparison_plots(self, segs1, segs2, segs3,
                                kp: float, ki: float, kd: float,
                                ku: float, tu: float):
        POS_TOL = ag_motion_tests.POS_TOL

        # ── PNG (matplotlib) ─────────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(18, 18))
            fig.suptitle(
                "Ziegler-Nichols — Sintonización PID base mecanum\n"
                f"Ku={ku:.4f}  Tu={tu:.4f}s   →   "
                f"Kp={kp:.4f}  Ki={ki:.4f}  Kd={kd:.4f} (acotado)",
                fontsize=13, fontweight="bold")
            gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.32)

            # Fila 1 — resumen propio de ZN (equivalente a la fila de
            # "evolución del AG", pero con el contenido que sí tiene sentido
            # para un método de un solo punto de operación).
            ax0 = fig.add_subplot(gs[0, 0])
            kps_h = [r.kp for r in self._search_history]
            rat_h = [r.ratio if r.ratio is not None else float("nan")
                     for r in self._search_history]
            colors = {"decaying": "tab:blue", "sustained": "tab:green",
                      "growing": "tab:red", "insufficient": "tab:gray"}
            pt_colors = [colors.get(r.status, "black") for r in self._search_history]
            ax0.scatter(kps_h, rat_h, c=pt_colors, s=45, zorder=3)
            ax0.axhspan(OSC_RATIO_SUSTAINED[0], OSC_RATIO_SUSTAINED[1],
                        color="green", alpha=0.10)
            ax0.axvline(ku, color="green", ls="--", lw=1.5, label=f"Ku={ku:.3f}")
            ax0.set_title("Búsqueda de Ku (oscilación sostenida)")
            ax0.set_xlabel("Kp probado"); ax0.set_ylabel("razón de amplitud")
            ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

            ax1 = fig.add_subplot(gs[0, 1])
            bars = ax1.bar(["Kp", "Ki", "Kd"], [kp, ki, kd],
                            color=["tab:red", "tab:green", "tab:blue"])
            for b, v in zip(bars, [kp, ki, kd]):
                ax1.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                          ha="center", va="bottom", fontsize=9)
            ax1.set_title("Ganancias ZN finales (acotadas al espacio del AG)")
            ax1.grid(True, alpha=0.3, axis="y")

            ax3 = fig.add_subplot(gs[1, 0])
            for seg in segs1 or []:
                if seg.t and len(seg.t) == len(seg.vx_real):
                    ax3.plot(seg.t, seg.vx_ref,  "--", lw=1.5, label=f"{seg.name} ref", alpha=0.8)
                    ax3.plot(seg.t, seg.vx_real, "-",  lw=1.5, label=f"{seg.name} real")
            ax3.set_title("P1 — vx"); ax3.set_xlabel("t (s)"); ax3.set_ylabel("vx (m/s)")
            ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

            ax4 = fig.add_subplot(gs[1, 1])
            for seg in segs2 or []:
                if seg.t and len(seg.t) == len(seg.wz_real):
                    ax4.plot(seg.t, seg.wz_ref,  "--", lw=1.5, label=f"{seg.name} ref", alpha=0.8)
                    ax4.plot(seg.t, seg.wz_real, "-",  lw=1.5, label=f"{seg.name} real")
            ax4.set_title("P2 — wz (rotación pura)"); ax4.set_xlabel("t (s)"); ax4.set_ylabel("wz (rad/s)")
            ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

            ax5 = fig.add_subplot(gs[2, 0])
            for seg in segs3 or []:
                if seg.t and len(seg.t) == len(seg.pos_err):
                    ax5.plot(seg.t, seg.pos_err, "-", lw=1.5, label=seg.name)
            ax5.axhline(POS_TOL, color="r", ls=":", lw=1.2, label=f"tol={POS_TOL}m")
            ax5.set_title("P3 — error de posición"); ax5.set_xlabel("t (s)")
            ax5.set_ylabel("|err| (m)"); ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

            all_segs = (segs1 or []) + (segs2 or []) + (segs3 or [])
            ax6 = fig.add_subplot(gs[2, 1])
            ax7 = fig.add_subplot(gs[3, 0])
            ax8 = fig.add_subplot(gs[3, 1])
            t_offset, drawn = 0.0, False
            for seg in all_segs:
                if not seg.t or len(seg.t) != len(seg.x_real):
                    continue
                ts = [t + t_offset for t in seg.t]
                lbl_ref  = "deseada"  if not drawn else "_nolegend_"
                lbl_real = "obtenida" if not drawn else "_nolegend_"
                ax6.plot(ts, seg.x_ref, "--", lw=1.3, alpha=0.8, color="tab:orange", label=lbl_ref)
                ax6.plot(ts, seg.x_real, "-", lw=1.3, color="tab:blue", label=lbl_real)
                ax7.plot(ts, seg.y_ref, "--", lw=1.3, alpha=0.8, color="tab:orange", label=lbl_ref)
                ax7.plot(ts, seg.y_real, "-", lw=1.3, color="tab:blue", label=lbl_real)
                ax8.plot(ts, [math.degrees(v) for v in seg.yaw_ref], "--", lw=1.3,
                         alpha=0.8, color="tab:orange", label=lbl_ref)
                ax8.plot(ts, [math.degrees(v) for v in seg.yaw_real], "-", lw=1.3,
                         color="tab:blue", label=lbl_real)
                if seg.t:
                    t_offset += seg.t[-1] + 0.1
                drawn = True

            for ax, title, ylabel in [
                (ax6, "Pose X — deseada vs obtenida (P1→P2→P3)", "x (m)"),
                (ax7, "Pose Y — deseada vs obtenida (P1→P2→P3)", "y (m)"),
                (ax8, "Pose Yaw — deseada vs obtenida (P1→P2→P3)", "yaw (°)"),
            ]:
                ax.set_title(title); ax.set_xlabel("t (s, concatenado)"); ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                if drawn:
                    ax.legend(fontsize=7)

            out_png = "zn_vs_ag_comparison.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] PNG de comparación guardado en {os.path.abspath(out_png)}")
        except ImportError:
            print("[plot] matplotlib no disponible — omitiendo PNG de comparación")

        # ── HTML interactivo (plotly) ────────────────────────────────────────
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=[
                    "Búsqueda de Ku (oscilación sostenida)", "Ganancias ZN finales",
                    "P1 — vel X (ref vs real)", "P2 — vel angular (ref vs real)",
                    "P3 — error de posición", "Pose X — deseada vs obtenida",
                    "Pose Y — deseada vs obtenida", "Pose Yaw — deseada vs obtenida",
                ],
                vertical_spacing=0.08, horizontal_spacing=0.10,
            )

            kps_h = [r.kp for r in self._search_history]
            rat_h = [r.ratio if r.ratio is not None else None for r in self._search_history]
            fig.add_trace(go.Scatter(x=kps_h, y=rat_h, mode="markers",
                          marker=dict(size=9), name="ensayos ZN"), row=1, col=1)
            fig.add_vline(x=ku, line_dash="dash", line_color="green",
                          annotation_text=f"Ku={ku:.3f}", row=1, col=1)

            fig.add_trace(go.Bar(x=["Kp", "Ki", "Kd"], y=[kp, ki, kd],
                          marker_color=["red", "green", "blue"], name="ganancias"),
                          row=1, col=2)

            COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
            for ci, seg in enumerate(segs1 or []):
                if not seg.t or len(seg.t) != len(seg.vx_real): continue
                c = COLORS[ci % len(COLORS)]
                fig.add_trace(go.Scatter(x=seg.t, y=seg.vx_ref, name=f"{seg.name} ref",
                              line=dict(dash="dash", color=c)), row=2, col=1)
                fig.add_trace(go.Scatter(x=seg.t, y=seg.vx_real, name=f"{seg.name} real",
                              line=dict(color=c)), row=2, col=1)

            for ci, seg in enumerate(segs2 or []):
                if not seg.t or len(seg.t) != len(seg.wz_real): continue
                c = COLORS[ci % len(COLORS)]
                fig.add_trace(go.Scatter(x=seg.t, y=seg.wz_ref, name=f"{seg.name} ref",
                              line=dict(dash="dash", color=c)), row=2, col=2)
                fig.add_trace(go.Scatter(x=seg.t, y=seg.wz_real, name=f"{seg.name} real",
                              line=dict(color=c)), row=2, col=2)

            for ci, seg in enumerate(segs3 or []):
                if not seg.t or len(seg.t) != len(seg.pos_err): continue
                fig.add_trace(go.Scatter(x=seg.t, y=seg.pos_err, name=seg.name,
                              line=dict(color=COLORS[ci % len(COLORS)])), row=3, col=1)
            fig.add_hline(y=POS_TOL, line_dash="dot", line_color="red",
                          annotation_text=f"tol {POS_TOL}m", row=3, col=1)

            all_segs = (segs1 or []) + (segs2 or []) + (segs3 or [])
            t_offset, first = 0.0, True
            for seg in all_segs:
                if not seg.t or len(seg.t) != len(seg.x_real):
                    continue
                ts = [t + t_offset for t in seg.t]
                sl = first
                fig.add_trace(go.Scatter(x=ts, y=seg.x_ref, name="deseada",
                              legendgroup="ref", showlegend=sl,
                              line=dict(dash="dash", color="orange")), row=3, col=2)
                fig.add_trace(go.Scatter(x=ts, y=seg.x_real, name="obtenida",
                              legendgroup="real", showlegend=sl,
                              line=dict(color="royalblue")), row=3, col=2)
                fig.add_trace(go.Scatter(x=ts, y=seg.y_ref, name="deseada",
                              legendgroup="ref", showlegend=False,
                              line=dict(dash="dash", color="orange")), row=4, col=1)
                fig.add_trace(go.Scatter(x=ts, y=seg.y_real, name="obtenida",
                              legendgroup="real", showlegend=False,
                              line=dict(color="royalblue")), row=4, col=1)
                yaw_ref_deg  = [math.degrees(v) for v in seg.yaw_ref]
                yaw_real_deg = [math.degrees(v) for v in seg.yaw_real]
                fig.add_trace(go.Scatter(x=ts, y=yaw_ref_deg, name="deseada",
                              legendgroup="ref", showlegend=False,
                              line=dict(dash="dash", color="orange")), row=4, col=2)
                fig.add_trace(go.Scatter(x=ts, y=yaw_real_deg, name="obtenida",
                              legendgroup="real", showlegend=False,
                              line=dict(color="royalblue")), row=4, col=2)
                first = False
                if seg.t:
                    t_offset += seg.t[-1] + 0.1

            fig.update_layout(
                height=1500, width=1300,
                title_text=(f"Ziegler-Nichols — Ku={ku:.4f} Tu={tu:.4f}s<br>"
                            f"<sub>Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f} (acotado)</sub>"),
                template="plotly_white",
            )

            out_html = os.path.abspath("zn_vs_ag_comparison.html")
            fig.write_html(out_html)
            print(f"[plot] HTML de comparación guardado en {out_html}")
            webbrowser.open("file://" + out_html)
        except ImportError:
            print("[plot] plotly no disponible — omitiendo HTML de comparación")


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

        # Bloque explícito e incondicional (no depende de que matplotlib esté
        # instalado) — señala claramente la ganancia última encontrada por
        # oscilación sostenida, lista para citar en la tesis.
        node.get_logger().info("=" * 60)
        node.get_logger().info("GANANCIA ÚLTIMA (OSCILACIÓN SOSTENIDA) DETECTADA")
        node.get_logger().info(f"  Ku (Kp crítico)            = {ku:.4f}")
        node.get_logger().info(f"  Tu (periodo de oscilación) = {tu:.4f} s")
        node.get_logger().info("=" * 60)

        gains_table_raw = node.compute_zn_table(ku, tu)
        node.get_logger().info("Tabla ZN — ganancias CRUDAS (sin acotar):")
        for name, (kp, ki, kd) in gains_table_raw.items():
            node.get_logger().info(f"  [{name}] Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}")

        # ── Acotar cada entrada de la tabla al mismo espacio admisible del AG ──
        # Justificación: Ki = Kp/Ti con Ti ligado a Tu tiende a dispararse
        # cuando el retraso del motor (motor_tau) empuja la oscilación
        # sostenida a un periodo corto. Sin este acotamiento, la comparación
        # contra el AG no sería justa (el AG nunca explora fuera de
        # KP_RANGE/KI_RANGE/KD_RANGE) y el PID resultante podría saturar el
        # actuador o generar windup severo en la simulación.
        gains_table_bounded = {}
        for name, (kp, ki, kd) in gains_table_raw.items():
            (kp_b, ki_b, kd_b), clamped = node.bound_gains(kp, ki, kd)
            gains_table_bounded[name] = (kp_b, ki_b, kd_b)
            if any(clamped.values()):
                recortadas = [g.upper() for g, was in clamped.items() if was]
                node.get_logger().warn(
                    f"  [{name}] ganancia(s) {recortadas} recortada(s) al rango del AG "
                    f"→ Kp={kp_b:.4f} Ki={ki_b:.4f} Kd={kd_b:.4f} "
                    f"(cruda: Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f})")

        # Validar el PID clásico (acotado) con la misma métrica del AG (P1+P2+P3)
        kp_c, ki_c, kd_c = gains_table_bounded["PID_clasico"]
        node.get_logger().info(
            "Validando ganancias ZN (PID clásico, acotado) con la batería "
            "P1/P2/P3 del AG para comparación directa...")
        fitness = node.evaluate([kp_c, ki_c, kd_c])[0]
        node.get_logger().info(f"Fitness ZN (comparable con AG) = {fitness:.5f}")

        # ── Corrida final grabada — MISMAS series que exporta el AG para su
        # mejor individuo (record_best se hereda tal cual de AGMotionEvaluator) ──
        node.get_logger().info("Grabando corrida final del PID clásico ZN (acotado)...")
        segs1, segs2, segs3 = node.record_best([kp_c, ki_c, kd_c])

        results = {
            "method": "ziegler_nichols",
            "motor_tau": MOTOR_TAU_DEFAULT,
            "ku": ku,
            "tu": tu,
            "config": {
                # Mismos campos y mismas unidades que "config" en ag_results.json
                # (ag_motion_tests.py) para que un único script de MATLAB pueda
                # leer cualquiera de los dos JSON con el mismo parser.
                "dist_x": ag_motion_tests.DIST_X,
                "dist_return": ag_motion_tests.DIST_RETURN,
                "rot_angle_deg": math.degrees(ag_motion_tests.ROT_ANGLE),
                "vx_ref": ag_motion_tests.VX_REF,
                "vy_ref": ag_motion_tests.VY_REF,
                "wz_ref": ag_motion_tests.WZ_REF,
                "weights": {"P1": ag_motion_tests.W1, "P2": ag_motion_tests.W2,
                            "P3": ag_motion_tests.W3},
                "note": "Sintonización por Ziegler-Nichols (lazo cerrado, "
                        "oscilación sostenida). Ganancias acotadas a "
                        "KP_RANGE/KI_RANGE/KD_RANGE (mismo espacio del AG) "
                        "para comparación justa; ver gains_table_raw para "
                        "los valores sin acotar.",
            },
            "gains_table_raw": {
                name: {"kp": kp, "ki": ki, "kd": kd}
                for name, (kp, ki, kd) in gains_table_raw.items()
            },
            "gains_table_bounded": {
                name: {"kp": kp, "ki": ki, "kd": kd}
                for name, (kp, ki, kd) in gains_table_bounded.items()
            },
            "bounds_used": {
                "KP_RANGE": list(ag_motion_tests.KP_RANGE),
                "KI_RANGE": list(ag_motion_tests.KI_RANGE),
                "KD_RANGE": list(ag_motion_tests.KD_RANGE),
            },
            # "best" con el mismo nombre/forma que usa ag_results.json, para
            # que el post-procesamiento en MATLAB identifique el punto de
            # operación final de la misma manera en ambos archivos.
            "best": {"kp": kp_c, "ki": ki_c, "kd": kd_c, "fitness": fitness},
            "pid_clasico_fitness_ag_metric": fitness,
            "search_history": [r.as_dict() for r in node._search_history],
            # ── Series completas P1/P2/P3 del PID clásico ZN (acotado) ──────
            # Mismas llaves que ag_results.json → best_run.test1/test2/test3,
            # así el script de MATLAB que ya lees para el AG funciona igual
            # aquí sin modificar el parser.
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

        node.build_search_plot(ku, tu)
        node.build_comparison_plots(segs1, segs2, segs3, kp_c, ki_c, kd_c, ku, tu)

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()