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

[ACTUALIZACIÓN]: 
Se agregaron filtros para evadir ruido de alta frecuencia en la odometría y
falsos positivos en ganancias iniciales, garantizando que el algoritmo detecte 
la dinámica real de la planta (tau=0.46s) y no artefactos de muestreo a 20Hz.
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
# CONFIGURACIÓN DE LA BÚSQUEDA ZN
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_TAU_DEFAULT = 0.46                       # s — identificado experimentalmente

ZN_KP_START     = 0.5
ZN_KP_MAX       = ag_motion_tests.KP_RANGE[1]  # Límite físico del AG
ZN_KP_GROWTH    = 1.4                          # Búsqueda gruesa
ZN_BISECT_TOL   = 0.03                         # Tolerancia relativa final
ZN_MAX_BISECT   = 12

ZN_STEP_DIST    = ag_motion_tests.DIST_X
ZN_STEP_TIMEOUT = 8.0                          # s — capturar varios ciclos

# --- FILTROS DE RUIDO Y ROBUSTEZ ---
OSC_MIN_EXTREMA      = 5      # Mínimo de extremos locales para análisis
OSC_SKIP_FRACTION    = 0.25   # Descartar transitorio de arranque
OSC_RATIO_SUSTAINED  = (0.85, 1.15) 
OSC_RATIO_GROWING    = 1.15   

ZN_MIN_AMPLITUDE     = 0.02   # (m/s) Umbral mínimo de error para ignorar ruido de odometría
ZN_MIN_TU            = 0.5    # (s) Periodo mínimo aceptable (10x el lazo de 20Hz)

OUT_JSON = "zn_results.json"
OUT_PNG  = "zn_results.png"
OUT_HTML = "zn_results.html"

# ══════════════════════════════════════════════════════════════════════════════
# Estructuras de resultado
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class OscillationResult:
    kp: float
    status: str
    ratio: Optional[float] = None
    tu: Optional[float] = None
    n_extrema: int = 0
    ok: bool = True

    def as_dict(self) -> dict:
        return {"kp": self.kp, "status": self.status, "ratio": self.ratio,
                "tu": self.tu, "n_extrema": self.n_extrema, "ok": self.ok}


def _seg_to_dict(s: SegmentLog) -> dict:
    return {"name": s.name, "t": s.t,
            "vx_ref": s.vx_ref, "vy_ref": s.vy_ref, "wz_ref": s.wz_ref,
            "vx_real": s.vx_real, "vy_real": s.vy_real, "wz_real": s.wz_real,
            "pos_err": s.pos_err,
            "x_ref": s.x_ref, "y_ref": s.y_ref, "yaw_ref": s.yaw_ref,
            "x_real": s.x_real, "y_real": s.y_real, "yaw_real": s.yaw_real}


# ══════════════════════════════════════════════════════════════════════════════
# Nodo ZN
# ══════════════════════════════════════════════════════════════════════════════
class ZNTuner(AGMotionEvaluator):

    def __init__(self):
        super().__init__()
        self._search_history: List[OscillationResult] = []
        self.get_logger().info("ZNTuner listo (versión filtrada contra ruido).")

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
            f"incluye el retraso electromecánico.")

    def _step_test(self, kp: float) -> Tuple[SegmentLog, bool]:
        self._set_pid(kp, 0.0, 0.0)
        self._teleport()
        self._start_arm()

        _itae, _elapsed, ok, seg = self._drive(
            ZN_STEP_DIST, "x",
            vx=+ag_motion_tests.VX_REF,
            timeout=ZN_STEP_TIMEOUT,
            seg_name=f"ZN_kp_{kp:.3f}")

        self._stop_arm()
        return seg, ok

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

        if not ok and len(extrema) < OSC_MIN_EXTREMA:
            return OscillationResult(kp=kp, status="growing", ok=ok, n_extrema=len(extrema))

        if len(extrema) < OSC_MIN_EXTREMA:
            return OscillationResult(kp=kp, status="insufficient", ok=ok, n_extrema=len(extrema))

        times = [e[0] for e in extrema]
        mags  = [abs(e[1]) for e in extrema]

        # --- DEFENSA 1: Umbral de Amplitud de Ruido ---
        max_amp = max(mags)
        if max_amp < ZN_MIN_AMPLITUDE:
            self.get_logger().debug(f"Picos muy pequeños (max={max_amp:.4f} < {ZN_MIN_AMPLITUDE}). Ignorando como ruido.")
            return OscillationResult(kp=kp, status="decaying", ok=ok, n_extrema=len(extrema))

        same_type_mags = mags[0::2]
        if len(same_type_mags) < 2:
            return OscillationResult(kp=kp, status="insufficient", ok=ok, n_extrema=len(extrema))

        ratios = []
        for i in range(len(same_type_mags) - 1):
            prev = same_type_mags[i]
            nxt  = same_type_mags[i + 1]
            if prev > 1e-6:
                ratios.append(nxt / prev)
        
        if not ratios:
            return OscillationResult(kp=kp, status="insufficient", ok=ok, n_extrema=len(extrema))
            
        mean_ratio = statistics.mean(ratios)

        periods = [times[i + 2] - times[i] for i in range(len(times) - 2)]
        tu = statistics.mean(periods) if periods else None

        # --- DEFENSA 2: Filtro de Periodo Mínimo (Muestreo Nyquist) ---
        if tu is not None and tu < ZN_MIN_TU:
            self.get_logger().debug(f"Periodo Tu={tu:.3f}s demasiado corto. Ignorando ruido de alta frecuencia.")
            status = "decaying"
        elif not ok:
            status = "growing"
        elif mean_ratio > OSC_RATIO_GROWING:
            status = "growing"
        elif OSC_RATIO_SUSTAINED[0] <= mean_ratio <= OSC_RATIO_SUSTAINED[1]:
            status = "sustained"
        else:
            status = "decaying"

        return OscillationResult(kp=kp, status=status, ratio=mean_ratio,
                                  tu=tu, n_extrema=len(extrema), ok=ok)

    def find_ultimate_gain(self) -> Tuple[Optional[float], Optional[float]]:
        kp = ZN_KP_START
        last_stable_kp = 0.0
        last_unstable_kp = None
        is_first_test = True

        # Fase 1 — búsqueda gruesa
        while kp <= ZN_KP_MAX:
            seg, ok = self._step_test(kp)
            res = self._analyze_oscillation(seg, kp, ok)
            
            # --- DEFENSA 3: Rechazo en primer Kp ---
            if is_first_test and res.status == "sustained":
                self.get_logger().warn(f"[ZN] Falsa oscilación sostenida detectada en Kp inicial ({kp}). Forzando 'decaying'.")
                res.status = "decaying"
            
            self._search_history.append(res)
            self.get_logger().info(
                f"[ZN][gruesa] Kp={kp:.3f} -> {res.status} "
                f"(ratio={res.ratio}, Tu={res.tu}, extrema={res.n_extrema})")

            is_first_test = False

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

        seg, ok = self._step_test(hi)
        res = self._analyze_oscillation(seg, hi, ok)
        self._search_history.append(res)
        if res.tu is None:
            self.get_logger().warn("No se pudo estimar Tu con precisión al cerrar la bisección.")
        return hi, res.tu

    @staticmethod
    def compute_zn_table(ku: float, tu: float) -> Dict[str, Tuple[float, float, float]]:
        table = {}
        table["P"] = (0.5 * ku, 0.0, 0.0)
        kp_pi = 0.45 * ku
        ti_pi = tu / 1.2
        table["PI"] = (kp_pi, kp_pi / ti_pi, 0.0)
        kp_pid = 0.6 * ku
        ti_pid = tu / 2.0
        td_pid = tu / 8.0
        table["PID_clasico"] = (kp_pid, kp_pid / ti_pid, kp_pid * td_pid)
        kp_no = 0.2 * ku
        ti_no = tu / 2.0
        td_no = tu / 3.0
        table["PID_sin_sobreimpulso"] = (kp_no, kp_no / ti_no, kp_no * td_no)
        return table

    @staticmethod
    def bound_gains(kp: float, ki: float, kd: float) -> Tuple[Tuple[float, float, float], Dict[str, bool]]:
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

    def build_search_plot(self, ku: Optional[float], tu: Optional[float]):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        kps    = [r.kp for r in self._search_history]
        ratios = [r.ratio if r.ratio is not None else float("nan") for r in self._search_history]
        colors = {"decaying": "tab:blue", "sustained": "tab:green",
                  "growing": "tab:red", "insufficient": "tab:gray"}
        point_colors = [colors.get(r.status, "black") for r in self._search_history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(kps, ratios, c=point_colors, s=60, zorder=3)
        for r in self._search_history:
            ax.annotate(r.status[:3], (r.kp, r.ratio if r.ratio else 0), fontsize=7, alpha=0.7)
        ax.axhspan(OSC_RATIO_SUSTAINED[0], OSC_RATIO_SUSTAINED[1], color="green", alpha=0.10, label="banda sostenida")
        ax.axhline(1.0, color="black", ls=":", lw=1)

        sustained_pt = next((r for r in reversed(self._search_history) if r.status == "sustained"), None)
        if ku is not None:
            ax.axvline(ku, color="green", ls="--", lw=1.5, label=f"Ku={ku:.3f}")
        if sustained_pt is not None:
            ax.scatter([sustained_pt.kp], [sustained_pt.ratio], s=220, facecolors="none", edgecolors="darkgreen", linewidths=2.2, zorder=4)
            ax.annotate(
                f"Ku={sustained_pt.kp:.3f}\nTu={tu:.3f}s\n(oscilación sostenida)",
                xy=(sustained_pt.kp, sustained_pt.ratio),
                xytext=(0.65, 0.85), textcoords="axes fraction",
                fontsize=9, color="darkgreen", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))

        title = "Búsqueda de ganancia última (Ziegler-Nichols)"
        if tu is not None: title += f"  —  Tu={tu:.3f}s"
        ax.set_title(title)
        ax.set_xlabel("Kp probado"); ax.set_ylabel("Razón de amplitud")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def build_comparison_plots(self, segs1, segs2, segs3, kp: float, ki: float, kd: float, ku: float, tu: float):
        POS_TOL = ag_motion_tests.POS_TOL
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(18, 18))
            fig.suptitle(
                "Ziegler-Nichols — Sintonización PID base mecanum\n"
                f"Ku={ku:.4f}  Tu={tu:.4f}s   →   Kp={kp:.4f}  Ki={ki:.4f}  Kd={kd:.4f} (acotado)",
                fontsize=13, fontweight="bold")
            gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.32)

            ax0 = fig.add_subplot(gs[0, 0])
            kps_h = [r.kp for r in self._search_history]
            rat_h = [r.ratio if r.ratio is not None else float("nan") for r in self._search_history]
            colors = {"decaying": "tab:blue", "sustained": "tab:green", "growing": "tab:red", "insufficient": "tab:gray"}
            pt_colors = [colors.get(r.status, "black") for r in self._search_history]
            ax0.scatter(kps_h, rat_h, c=pt_colors, s=45, zorder=3)
            ax0.axhspan(OSC_RATIO_SUSTAINED[0], OSC_RATIO_SUSTAINED[1], color="green", alpha=0.10)
            ax0.axvline(ku, color="green", ls="--", lw=1.5, label=f"Ku={ku:.3f}")
            ax0.set_title("Búsqueda de Ku (oscilación sostenida)")
            ax0.set_xlabel("Kp probado"); ax0.set_ylabel("razón de amplitud")
            ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

            ax1 = fig.add_subplot(gs[0, 1])
            bars = ax1.bar(["Kp", "Ki", "Kd"], [kp, ki, kd], color=["tab:red", "tab:green", "tab:blue"])
            for b, v in zip(bars, [kp, ki, kd]):
                ax1.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
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
            ax6 = fig.add_subplot(gs[2, 1]); ax7 = fig.add_subplot(gs[3, 0]); ax8 = fig.add_subplot(gs[3, 1])
            t_offset, drawn = 0.0, False
            for seg in all_segs:
                if not seg.t or len(seg.t) != len(seg.x_real): continue
                ts = [t + t_offset for t in seg.t]
                lbl_ref  = "deseada"  if not drawn else "_nolegend_"
                lbl_real = "obtenida" if not drawn else "_nolegend_"
                ax6.plot(ts, seg.x_ref, "--", lw=1.3, alpha=0.8, color="tab:orange", label=lbl_ref)
                ax6.plot(ts, seg.x_real, "-", lw=1.3, color="tab:blue", label=lbl_real)
                ax7.plot(ts, seg.y_ref, "--", lw=1.3, alpha=0.8, color="tab:orange", label=lbl_ref)
                ax7.plot(ts, seg.y_real, "-", lw=1.3, color="tab:blue", label=lbl_real)
                ax8.plot(ts, [math.degrees(v) for v in seg.yaw_ref], "--", lw=1.3, alpha=0.8, color="tab:orange", label=lbl_ref)
                ax8.plot(ts, [math.degrees(v) for v in seg.yaw_real], "-", lw=1.3, color="tab:blue", label=lbl_real)
                if seg.t: t_offset += seg.t[-1] + 0.1
                drawn = True

            for ax, title, ylabel in [(ax6, "Pose X", "x (m)"), (ax7, "Pose Y", "y (m)"), (ax8, "Pose Yaw", "yaw (°)")]:
                ax.set_title(title); ax.set_xlabel("t (s)"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
                if drawn: ax.legend(fontsize=7)

            fig.savefig("zn_vs_ag_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError: pass

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=4, cols=2, subplot_titles=[
                "Búsqueda de Ku", "Ganancias ZN", "P1 — vel X", "P2 — vel angular",
                "P3 — error pos", "Pose X", "Pose Y", "Pose Yaw"], vertical_spacing=0.08, horizontal_spacing=0.10)

            kps_h = [r.kp for r in self._search_history]
            rat_h = [r.ratio if r.ratio is not None else None for r in self._search_history]
            fig.add_trace(go.Scatter(x=kps_h, y=rat_h, mode="markers", marker=dict(size=9), name="ensayos ZN"), row=1, col=1)
            fig.add_vline(x=ku, line_dash="dash", line_color="green", annotation_text=f"Ku={ku:.3f}", row=1, col=1)
            fig.add_trace(go.Bar(x=["Kp", "Ki", "Kd"], y=[kp, ki, kd], marker_color=["red", "green", "blue"], name="ganancias"), row=1, col=2)

            COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
            for ci, seg in enumerate(segs1 or []):
                if not seg.t or len(seg.t) != len(seg.vx_real): continue
                c = COLORS[ci % len(COLORS)]
                fig.add_trace(go.Scatter(x=seg.t, y=seg.vx_ref, name=f"{seg.name} ref", line=dict(dash="dash", color=c)), row=2, col=1)
                fig.add_trace(go.Scatter(x=seg.t, y=seg.vx_real, name=f"{seg.name} real", line=dict(color=c)), row=2, col=1)

            for ci, seg in enumerate(segs2 or []):
                if not seg.t or len(seg.t) != len(seg.wz_real): continue
                c = COLORS[ci % len(COLORS)]
                fig.add_trace(go.Scatter(x=seg.t, y=seg.wz_ref, name=f"{seg.name} ref", line=dict(dash="dash", color=c)), row=2, col=2)
                fig.add_trace(go.Scatter(x=seg.t, y=seg.wz_real, name=f"{seg.name} real", line=dict(color=c)), row=2, col=2)

            for ci, seg in enumerate(segs3 or []):
                if not seg.t or len(seg.t) != len(seg.pos_err): continue
                fig.add_trace(go.Scatter(x=seg.t, y=seg.pos_err, name=seg.name, line=dict(color=COLORS[ci % len(COLORS)])), row=3, col=1)
            fig.add_hline(y=POS_TOL, line_dash="dot", line_color="red", annotation_text=f"tol {POS_TOL}m", row=3, col=1)

            all_segs = (segs1 or []) + (segs2 or []) + (segs3 or [])
            t_offset, first = 0.0, True
            for seg in all_segs:
                if not seg.t or len(seg.t) != len(seg.x_real): continue
                ts = [t + t_offset for t in seg.t]
                sl = first
                fig.add_trace(go.Scatter(x=ts, y=seg.x_ref, name="deseada", legendgroup="ref", showlegend=sl, line=dict(dash="dash", color="orange")), row=3, col=2)
                fig.add_trace(go.Scatter(x=ts, y=seg.x_real, name="obtenida", legendgroup="real", showlegend=sl, line=dict(color="royalblue")), row=3, col=2)
                fig.add_trace(go.Scatter(x=ts, y=seg.y_ref, name="deseada", legendgroup="ref", showlegend=False, line=dict(dash="dash", color="orange")), row=4, col=1)
                fig.add_trace(go.Scatter(x=ts, y=seg.y_real, name="obtenida", legendgroup="real", showlegend=False, line=dict(color="royalblue")), row=4, col=1)
                fig.add_trace(go.Scatter(x=ts, y=[math.degrees(v) for v in seg.yaw_ref], name="deseada", legendgroup="ref", showlegend=False, line=dict(dash="dash", color="orange")), row=4, col=2)
                fig.add_trace(go.Scatter(x=ts, y=[math.degrees(v) for v in seg.yaw_real], name="obtenida", legendgroup="real", showlegend=False, line=dict(color="royalblue")), row=4, col=2)
                first = False
                if seg.t: t_offset += seg.t[-1] + 0.1

            fig.update_layout(height=1500, width=1300, title_text=f"Ziegler-Nichols — Ku={ku:.4f} Tu={tu:.4f}s", template="plotly_white")
            fig.write_html(os.path.abspath("zn_vs_ag_comparison.html"))
        except ImportError: pass

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

        node.get_logger().info("=" * 60)
        node.get_logger().info("GANANCIA ÚLTIMA (OSCILACIÓN SOSTENIDA) DETECTADA")
        node.get_logger().info(f"  Ku (Kp crítico)            = {ku:.4f}")
        node.get_logger().info(f"  Tu (periodo de oscilación) = {tu:.4f} s")
        node.get_logger().info("=" * 60)

        gains_table_raw = node.compute_zn_table(ku, tu)
        gains_table_bounded = {}
        for name, (kp, ki, kd) in gains_table_raw.items():
            (kp_b, ki_b, kd_b), clamped = node.bound_gains(kp, ki, kd)
            gains_table_bounded[name] = (kp_b, ki_b, kd_b)
            if any(clamped.values()):
                recortadas = [g.upper() for g, was in clamped.items() if was]
                node.get_logger().warn(
                    f"  [{name}] ganancia(s) {recortadas} recortada(s) al rango del AG "
                    f"→ Kp={kp_b:.4f} Ki={ki_b:.4f} Kd={kd_b:.4f}")

        kp_c, ki_c, kd_c = gains_table_bounded["PID_clasico"]
        fitness = node.evaluate([kp_c, ki_c, kd_c])[0]
        node.get_logger().info(f"Fitness ZN (comparable con AG) = {fitness:.5f}")

        segs1, segs2, segs3 = node.record_best([kp_c, ki_c, kd_c])
        results = {
            "method": "ziegler_nichols", "motor_tau": MOTOR_TAU_DEFAULT, "ku": ku, "tu": tu,
            "config": {"note": "ZN Filtrado"},
            "gains_table_raw": {name: {"kp": kp, "ki": ki, "kd": kd} for name, (kp, ki, kd) in gains_table_raw.items()},
            "gains_table_bounded": {name: {"kp": kp, "ki": ki, "kd": kd} for name, (kp, ki, kd) in gains_table_bounded.items()},
            "best": {"kp": kp_c, "ki": ki_c, "kd": kd_c, "fitness": fitness},
            "search_history": [r.as_dict() for r in node._search_history],
            "best_run": {
                "test1": [_seg_to_dict(s) for s in segs1],
                "test2": [_seg_to_dict(s) for s in segs2],
                "test3": [_seg_to_dict(s) for s in segs3],
            },
        }

        with open(os.path.abspath(OUT_JSON), "w") as f:
            json.dump(results, f, indent=2)

        node.build_search_plot(ku, tu)
        node.build_comparison_plots(segs1, segs2, segs3, kp_c, ki_c, kd_c, ku, tu)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()