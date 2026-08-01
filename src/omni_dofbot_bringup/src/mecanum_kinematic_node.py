#!/usr/bin/env python3
"""
mecanum_kinematic_node.py

Nodo ROS 2 que combina la cinemática inversa mecanum con un controlador
PID por rueda que opera en el MISMO DOMINIO FÍSICO que el firmware del
fabricante (STM32 / Yahboom): error en mm/s -> PID incremental -> PWM
saturado -> ganancia estática del motor (K) -> filtro de primer orden
(dinámica electromecánica, tau) -> comando final a Gazebo en rad/s.

══════════════════════════════════════════════════════════════════════════
ARQUITECTURA — "Planta Completa" (Sim2Real)
══════════════════════════════════════════════════════════════════════════
Gazebo/gz_ros2_control solo entiende rad/s (o torque) en las juntas de las
ruedas — no existe un concepto de PWM dentro del motor de físicas. Para que
las ganancias que encuentre el AG sean transferibles al robot físico, el
lazo de control completo del firmware se modela DENTRO de este nodo, y
Gazebo se usa únicamente como el motor de físicas que integra el
movimiento resultante:

    omega_ref (rad/s, cinemática inversa)
        │
        ▼  [× wheel_radius × 1000]
    v_ref (mm/s)  ──┐
                    │  error = v_ref - v_real
    v_real (mm/s) ──┘        │
                              ▼
                  PID INCREMENTAL (mismo dominio que el firmware de yahboom)
                  ΔPWM = Kp·(err-err_k1) + Ki·err + Kd·(err-2·err_k1+err_k2)
                  pwm_output += ΔPWM   (clamp ±pwm_max)
                              │
                              ▼  [× motor_gain_k]
                  omega_from_pwm (rad/s) — ganancia estática del motor,
                  caracterizada experimentalmente con escalón de PWM
                              │
                              ▼  FirstOrderLag(tau≈0.46s)
                  omega_filtered (rad/s) — dinámica electromecánica real
                              │
                              ▼  [clamp ±max_wheel_vel, por seguridad]
                  comando final → /wheel_velocity_controller/commands

DECISIONES DE DISEÑO Y APROXIMACIONES DOCUMENTADAS (notas para el escrito de la tesis)
────────────────────────────────────────────────────────────────────────
1. PID SIN NORMALIZAR POR dt (idéntico a la fórmula del firmware):
   El firmware asume un período de muestreo fijo de 10 ms embebido
   implícitamente en las ganancias (no multiplica el término Ki por dt).
   Para que Kp/Ki/Kd encontrados por el AG sean directamente comparables
   en magnitud a los del firmware, este PID usa la MISMA fórmula sin
   escalar por dt, apoyándose en que el timer de ROS 2 corre a
   control_rate fijo (ver control_rate=100.0 Hz, igual al firmware).
   Aproximación: el jitter del timer de ROS 2 (típicamente <1-2 ms sobre
   10 ms) introduce una pequeña discrepancia frente al periodo
   perfectamente fijo del microcontrolador; se documenta como limitación,
   no se corrige matemáticamente para no reintroducir el problema de
   escala de ganancias que esto buscaba resolver.

2. control_rate = 100.0 Hz (antes 50.0 Hz):
   Igualado al período de 10 ms del firmware. Un mismo Kp/Ki/Kd produce
   una respuesta distinta a distinta frecuencia de muestreo en un PID
   incremental sin normalizar por dt; iguala la base de comparación.

3. GANANCIA ESTÁTICA ÚNICA (motor_gain_k) PARA LAS 4 RUEDAS:
   Caracterizada con escalón de PWM=100 (saturado) sobre Enc_M1..M4;
   los cuatro motores mostraron respuesta muy similar, por lo que se usa
   un único K global en lugar de K1..K4 independientes. PENDIENTE:
   validar con un segundo escalón a mayor PWM (p. ej. 500-700) que la
   ganancia se mantenga aproximadamente lineal fuera del rango donde fue
   caracterizada (0-100) — ver motor_gain_k_calibration_note más abajo.

4. pwm_max y motor_gain_k: CONFIRMADOS mediante análisis del handler
   FUNC_MOTOR del firmware (ya no es un supuesto). El parser confirma:
   speed_raw = motor_cmd * ((MOTOR_MAX_PULSE - MOTOR_IGNORE_PULSE) / 100.0),
   es decir, set_motor(100) -> speed_raw=2000 (NO-SUNRISE) -> Hipótesis A
   verificada, factor de escala = 20 exacto (no 36). pwm_max=2000 y
   motor_gain_k=0.013018 rad/s por unidad de PWM crudo (Dominio A) son
   los valores usados. La única brecha de fidelidad restante frente al
   firmware real es que este modelo aún no incorpora el offset de
   velocidad ni la fricción estática (zona muerta) reportados por la
   caracterización — ver nota junto a MOTOR_GAIN_K_DOMINIO_B_MM_S.

5. NO SE MODELA el lazo de corrección de YAW (PD) del firmware.
   Ver limitación ya documentada en el capítulo correspondiente.

6. NO SE MODELAN zona muerta mecánica (fricción estática) ni saturación
   de voltaje de batería del motor real; K es una ganancia estática
   lineal única.

7. BANDA MUERTA DE ERROR (anti-vibración) — ADAPTACIÓN, no transcripción:
   Se observó vibración/oscilación de alta frecuencia en los motores aun
   con Kd bajo. Un Kd bajo NO descarta oscilación: en un PID incremental
   discreto, Kd amortigua sobreimpulso, pero el "correteo" cerca del
   punto de operación suele originarse en que el algoritmo sigue
   reaccionando a errores del orden del ruido de cuantización de la
   odometría, sin ningún umbral que los ignore. Se adapta al lazo de
   VELOCIDAD el mecanismo de banda muerta que el firmware documenta
   para su PID POSICIONAL (±40 cuentas, "se fuerza a cero y se resetea
   la integral") — con dos salvedades que deben quedar explícitas en la
   tesis: (a) el firmware aplica esa banda muerta a un lazo distinto del
   que efectivamente controla velocidad, por lo que esto es una
   generalización razonada, no una réplica; y (b) el valor numérico no es
   transferible entre dominios (cuentas de encoder vs. mm/s), por lo que
   ERROR_DEADBAND_MM_S_DEFAULT es un punto de partida a calibrar, igual
   que motor_gain_k.

SUSCRIBE:
  /cmd_vel      (geometry_msgs/Twist)   — velocidad deseada del cuerpo
  /joint_states (sensor_msgs/JointState) — velocidad real de cada rueda

PUBLICA:
  /wheel_velocity_controller/commands (std_msgs/Float64MultiArray)

SERVICIOS:
  ~/reset_controller_state (std_srvs/Trigger)
      Reinicia el acumulador PWM del PID incremental y el estado del
      filtro de motor. Debe llamarse junto con ~/reset_pose (de
      mecanum_odometry_node) cada vez que se teleporta el robot, para
      que no queden residuos de la prueba anterior.

PARÁMETROS (modificables en tiempo de ejecución):
  wheel_radius    : radio de rueda en metros           (default: 0.040)
  lx              : semilongitud robot eje X en metros  (default: 0.110)
  ly              : semiancho robot eje Y en metros     (default: 0.102)
  max_wheel_vel   : saturación final de seguridad (rad/s) (default: 20.0)
  control_rate    : frecuencia del lazo PID (Hz)        (default: 100.0)

  Ganancias PID — dominio mm/s -> PWM, mismas para las 4 ruedas:
  kp              : ganancia proporcional               (default: 1.0)
  ki              : ganancia integral (incremental)      (default: 0.0)
  kd              : ganancia derivativa (incremental)     (default: 0.0)
  pwm_max         : saturación de PWM, Dominio A (default: 2000.0 — confirmado)
  error_deadband_mm_s : banda muerta de error, anti-vibración (default: 2.0)
                    Ver nota de diseño #7 — adaptación del firmware, no
                    transcripción; requiere calibración propia.

  Planta del motor (Sim2Real):
  motor_gain_k    : ganancia estática, rad/s por unidad de PWM
                    (default: PLACEHOLDER — reemplazar con el valor real
                    obtenido de Enc_M1..M4 al escalón de PWM=100)
  motor_tau       : constante de tiempo del motor (s)   (default: 0.46)
                    0.0 = filtro desactivado.

CONVENCIÓN DE RUEDAS (orden del controlador):
  índice 0 → front_left_joint
  índice 1 → front_right_joint
  índice 2 → back_right_joint
  índice 3 → back_left_joint
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import numpy as np


WHEEL_JOINTS = [
    'front_left_joint',
    'front_right_joint',
    'back_right_joint',
    'back_left_joint',
]

# ── DECISIÓN DE ARQUITECTURA (tomada): Arquitectura A ──────────────────────
# El bridge de hardware (rosmaster_bridge_node.py, realizado en el repositorio de drivers) transferirá
# las ganancias encontradas por el AG directamente al PID INTERNO del
# firmware vía set_pid_param(Kp, Ki, Kd) — NO se calculará el PID en Python
# ni se usará set_motor() en producción. Por lo tanto este nodo debe modelar
# la planta en el DOMINIO DEL PID INTERNO (registro crudo del timer STM32,
# 0-3600, saturado por firmware a ±2000 para el modelo NO-SUNRISE aquí usado).
#
# RESUELTO mediante análisis del firmware fuente (handler de FUNC_MOTOR):
# la caracterización experimental de K (Enc_M1..M4 vs. PWM) se hizo con
# set_motor(), cuyo rango declarado (-100, 100) es una capa de abstracción
# de la API. El parser del firmware confirma la conversión exacta a
# registro crudo (ver PWM_CMD_TO_RAW_SCALE abajo) — ya no es un supuesto.
#
# CONFIRMADO por análisis del firmware fuente (handler de FUNC_MOTOR):
#   int16_t motor_pulse = MOTOR_MAX_PULSE - MOTOR_IGNORE_PULSE;  // =2000 (NO-SUNRISE)
#   speed_raw = (int16_t) motor_cmd * (motor_pulse / 100.0);
# Es decir: set_motor(100) -> speed_raw = 2000 exacto. La Hipótesis A
# queda verificada; se descarta la Hipótesis B (factor 36). El factor de
# escala real es 20, no una suposición.
PWM_CMD_TO_RAW_SCALE = 20.0   # unidades de PWM crudo por unidad de set_motor() — CONFIRMADO

# Límite de saturación — Dominio A, PID interno del firmware, NO-SUNRISE
# (confirmado): pid->pwm_output se clampa a ±(3600-1600) = ±2000.
MOTOR_MAX_PULSE_RAW = 3600
PWM_MAX_DOMINIO_A_DEFAULT = MOTOR_MAX_PULSE_RAW - 1600   # = 2000 (NO-SUNRISE)

# ── Ganancia estática del motor — caracterizada en Dominio B, convertida ──
# K promedio de los 4 motores (Dominio B: mm/s por unidad de set_motor()):
#   10.4144 mm/s/unidad — Motores individuales: M1=10.7032 M2=9.8567
#   M3=10.3531 M4=10.7445 (dispersión ~8%, K único global justificado).
#
# Conversión a Dominio A (rad/s por unidad de PWM crudo):
#   1. mm/s -> rad/s:        10.4144 / (1000 * wheel_radius=0.040) = 0.26036
#   2. por-unidad-cmd -> por-unidad-cruda:  0.26036 / PWM_CMD_TO_RAW_SCALE
#      (con Hipótesis A, escala=20):        0.26036 / 20 = 0.013018
#
# PENDIENTE: NO incorpora todavía el offset de velocidad ni la fricción
# estática (PWM mínimo ~7-9 en Dominio B, equivalente a ~140-180 ticks en
# Dominio A bajo Hipótesis A) reportados por la caracterización — el
# modelo usa solo la ganancia lineal K por ahora.
MOTOR_GAIN_K_DOMINIO_B_MM_S = 10.4144   # mm/s por unidad de set_motor() — medido
MOTOR_GAIN_K_DEFAULT = (MOTOR_GAIN_K_DOMINIO_B_MM_S / (1000.0 * 0.040)) / PWM_CMD_TO_RAW_SCALE
# = 0.013018 rad/s por unidad de PWM crudo (bajo Hipótesis A)

# ── PLACEHOLDER — banda muerta de error, ver nota de diseño #7 abajo ──────
# Adaptación (NO transcripción literal) del mecanismo de banda muerta
# documentado para el PID POSICIONAL del firmware (±40 cuentas de encoder,
# "se fuerza a cero y se resetea la integral"), aplicado aquí al LAZO DE
# VELOCIDAD (que es el que realmente está activo) y en el dominio mm/s.
# El valor "40" del firmware NO es transferible directamente: está en un
# dominio distinto (posición, cuentas de encoder) al de este lazo
# (velocidad, mm/s). Requiere calibración propia.
ERROR_DEADBAND_MM_S_DEFAULT = 2.0   # mm/s — AJUSTAR experimentalmente


class PIDController:
    """
    PID Incremental — misma estructura discreta que PID_Incre_Calc() del
    firmware del fabricante, SIN normalizar por dt (ver nota de diseño #1
    en el docstring del módulo). Opera en el dominio mm/s -> PWM.

    ΔPWM = Kp·(err - err_k1) + Ki·err + Kd·(err - 2·err_k1 + err_k2)
    pwm_output += ΔPWM
    pwm_output = clamp(pwm_output, -pwm_max, +pwm_max)

    donde err_k1 es el error del ciclo anterior y err_k2 el de dos ciclos
    atrás — igual a err_next/err_last en la nomenclatura del código C.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 pwm_max: float = 1000.0, error_deadband: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.pwm_max = pwm_max
        self.error_deadband = max(0.0, error_deadband)

        self._output = 0.0     # acumulador de PWM (equivalente a pwm_output en C)
        self._err_k1 = 0.0     # error del ciclo anterior (err_next en C)
        self._err_k2 = 0.0     # error de dos ciclos atrás (err_last en C)

    def reset(self):
        """Limpia el acumulador y el historial de error. Llamar tras
        cada teleport/reset de prueba para no arrastrar residuos."""
        self._output = 0.0
        self._err_k1 = 0.0
        self._err_k2 = 0.0

    def compute(self, error: float) -> float:
        """Calcula el PWM de salida a partir del error actual (mm/s).
        No recibe dt: ver nota de diseño #1 — se asume período fijo
        implícito en las ganancias, igual que en el firmware.

        BANDA MUERTA (nota de diseño #7): si |error| cae por debajo de
        error_deadband, se considera ruido de cuantización y no una señal
        de control confiable. Se congela la salida (se mantiene el último
        PWM aplicado, sin nuevo incremento) y se limpia el historial de
        error — análogo al "se fuerza a cero y se resetea la integral"
        del PID posicional del firmware, adaptado a la forma incremental:
        aquí no hay un integral separado que resetear, así que lo que se
        limpia es el historial que alimenta los términos Kp/Kd, evitando
        que un salto de error al SALIR de la banda muerta produzca un
        "derivative kick" artificial basado en errores ya obsoletos.
        """
        if abs(error) < self.error_deadband:
            self._err_k2 = 0.0
            self._err_k1 = 0.0
            return self._output

        delta_pwm = (
            self.kp * (error - self._err_k1)
            + self.ki * error
            + self.kd * (error - 2.0 * self._err_k1 + self._err_k2)
        )
        self._output += delta_pwm
        self._output = float(np.clip(self._output, -self.pwm_max, self.pwm_max))

        self._err_k2 = self._err_k1
        self._err_k1 = error

        return self._output

    def update_gains(self, kp: float, ki: float, kd: float,
                      pwm_max: float = None, error_deadband: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        if pwm_max is not None:
            self.pwm_max = pwm_max
        if error_deadband is not None:
            self.error_deadband = max(0.0, error_deadband)
        # Cambiar ganancias a mitad de una prueba con acumulador residual
        # produce un salto no físico — se reinicia por seguridad.
        self.reset()


class FirstOrderLag:
    """
    Filtro de primer orden discreto que emula la dinámica electromecánica
    real del motor (constante de tiempo tau, identificada con tfest).

    Discretización exacta ZOH de un polo simple en -1/tau:
        alpha = dt / (tau + dt)
        y[n]  = y[n-1] + alpha * (u[n] - y[n-1])

    tau = 0.0 desactiva el filtro (paso directo u -> y, sin retraso).
    A diferencia del PID incremental (nota #1), este filtro SÍ usa el dt
    real medido: modela una dinámica continua discretizada, y el error
    introducido por el jitter de ROS 2 aquí es de segundo orden (no altera
    la escala de ninguna ganancia sintonizable).
    """

    def __init__(self, tau: float = 0.0):
        self.tau = max(0.0, tau)
        self._y = 0.0

    def reset(self, y0: float = 0.0):
        self._y = y0

    def update_tau(self, tau: float):
        self.tau = max(0.0, tau)

    def filter(self, u: float, dt: float) -> float:
        if self.tau <= 0.0 or dt <= 0.0:
            self._y = u
            return self._y
        alpha = dt / (self.tau + dt)
        self._y += alpha * (u - self._y)
        return self._y


class MecanumKinematicNode(Node):

    def __init__(self):
        super().__init__('mecanum_kinematic_node')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('wheel_radius',  0.040)
        self.declare_parameter('lx',            0.110)
        self.declare_parameter('ly',            0.102)
        self.declare_parameter('max_wheel_vel', 20.0)
        self.declare_parameter('control_rate',  100.0)   # igualado al firmware (10ms)

        # Ganancias PID — dominio mm/s -> PWM
        self.declare_parameter('kp',            1.0)
        self.declare_parameter('ki',            0.0)
        self.declare_parameter('kd',            0.0)
        self.declare_parameter('pwm_max',       PWM_MAX_DOMINIO_A_DEFAULT)  # Dominio A — confirmado desde firmware (ver nota de diseño #4)
        self.declare_parameter('error_deadband_mm_s', ERROR_DEADBAND_MM_S_DEFAULT)

        # Planta del motor (Sim2Real)
        self.declare_parameter('motor_gain_k',  MOTOR_GAIN_K_DEFAULT)  # rad/s por PWM
        self.declare_parameter('motor_tau',     0.46)

        # Callback para actualizar parámetros en tiempo de ejecución
        self.add_on_set_parameters_callback(self._on_params_change)

        # ── PID por rueda (dominio mm/s -> PWM) ──────────────────────────
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        pwm_max = self.get_parameter('pwm_max').value
        deadband = self.get_parameter('error_deadband_mm_s').value
        self._pids = [PIDController(kp, ki, kd, pwm_max, deadband) for _ in range(4)]

        # ── Planta del motor: ganancia estática única + filtro de 1er orden ──
        self._motor_gain_k = self.get_parameter('motor_gain_k').value
        tau = self.get_parameter('motor_tau').value
        self._motor_lags = [FirstOrderLag(tau) for _ in range(4)]

        # ── Estado ────────────────────────────────────────────────────────
        self._omega_ref  = np.zeros(4)   # referencia cinemática inversa (rad/s)
        self._omega_real = np.zeros(4)   # velocidad real leída de joint_states (rad/s)
        self._last_time  = self.get_clock().now()
        self.wheel_polarity = [-1, -1, -1, -1]  # [fl, fr, br, bl]

        # ── Pub / Sub ─────────────────────────────────────────────────────
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

        # ── Servicio de reset de estado ──────────────────────────────────
        self._reset_srv = self.create_service(
            Empty, '~/reset_controller_state', self._on_reset_state
        )

        # ── Timer del lazo de control ─────────────────────────────────────
        rate = self.get_parameter('control_rate').value
        self._timer = self.create_timer(1.0 / rate, self._control_loop)

        self.get_logger().info(
            f'MecanumKinematicNode (planta completa) iniciado — '
            f'Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f} pwm_max={pwm_max:.1f} '
            f'deadband={deadband:.3f}mm/s '
            f'motor_gain_k={self._motor_gain_k:.5f} motor_tau={tau:.4f}s '
            f'rate={rate:.0f}Hz'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cmd_vel_cb(self, msg: Twist):
        """Convierte Twist a velocidades de referencia por rueda (rad/s)."""
        r  = self.get_parameter('wheel_radius').value
        lx = self.get_parameter('lx').value
        ly = self.get_parameter('ly').value
        k  = lx + ly

        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        # Cinemática inversa mecanum (estándar ROS 2)
        omega_raw = np.array([
            (vx - vy - k * wz) / r,   # front_left
            (vx + vy + k * wz) / r,   # front_right
            (vx - vy + k * wz) / r,   # back_right
            (vx + vy - k * wz) / r,   # back_left
        ])

        self._omega_ref = omega_raw * self.wheel_polarity

    def _joint_states_cb(self, msg: JointState):
        """Lee las velocidades reales de las ruedas desde /joint_states (rad/s)."""
        name_to_idx = {name: i for i, name in enumerate(WHEEL_JOINTS)}
        for i, name in enumerate(msg.name):
            if name in name_to_idx and i < len(msg.velocity):
                self._omega_real[name_to_idx[name]] = msg.velocity[i]

    def _control_loop(self):
        """
        Lazo de planta completa, ejecutado a control_rate Hz:
          1. Error en mm/s (mismo dominio que el firmware)
          2. PID incremental -> PWM saturado
          3. Ganancia estática del motor -> rad/s
          4. Filtro de primer orden (dinámica electromecánica) -> rad/s
          5. Clamp final de seguridad -> comando a Gazebo
        """
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        r = self.get_parameter('wheel_radius').value
        max_vel = self.get_parameter('max_wheel_vel').value

        # rad/s -> mm/s para operar en el mismo dominio que el firmware
        mm_per_rad_s = r * 1000.0

        commands = []
        for i, pid in enumerate(self._pids):
            v_ref_mm_s  = self._omega_ref[i]  * mm_per_rad_s
            v_real_mm_s = self._omega_real[i] * mm_per_rad_s
            error_mm_s  = v_ref_mm_s - v_real_mm_s

            # PID incremental — SIN dt (ver nota de diseño #1)
            pwm_cmd = pid.compute(error_mm_s)

            # Planta del motor: ganancia estática (PWM -> rad/s)
            omega_from_pwm = pwm_cmd * self._motor_gain_k

            # Dinámica electromecánica real (retraso de primer orden)
            omega_filtered = self._motor_lags[i].filter(omega_from_pwm, dt)

            # Clamp final de seguridad (protege a Gazebo de valores absurdos
            # si motor_gain_k o pwm_max están mal calibrados)
            omega_final = float(np.clip(omega_filtered, -max_vel, max_vel))

            commands.append(omega_final)

        msg = Float64MultiArray()
        msg.data = commands
        self._wheel_cmd_pub.publish(msg)

    def _on_reset_state(self, request, response):
        """Reinicia el acumulador PWM del PID incremental y el estado del
        filtro de motor. Llamar después de cada teleport del robot (junto
        con ~/reset_pose de mecanum_odometry_node)."""
        for pid in self._pids:
            pid.reset()
        for lag in self._motor_lags:
            lag.reset(0.0)

        self.get_logger().info(
            'Estado del controlador reiniciado (PID incremental + planta del motor).'
        )
        return response

    def _on_params_change(self, params):
        """Actualiza ganancias PID, pwm_max, motor_gain_k y motor_tau
        cuando el AG/ZN llaman a set_parameters()."""
        kp       = self.get_parameter('kp').value
        ki       = self.get_parameter('ki').value
        kd       = self.get_parameter('kd').value
        pwm_max  = self.get_parameter('pwm_max').value
        deadband = self.get_parameter('error_deadband_mm_s').value
        gain_k   = self._motor_gain_k
        tau      = self.get_parameter('motor_tau').value

        for param in params:
            if param.name == 'kp':                   kp = param.value
            elif param.name == 'ki':                 ki = param.value
            elif param.name == 'kd':                 kd = param.value
            elif param.name == 'pwm_max':             pwm_max = param.value
            elif param.name == 'error_deadband_mm_s': deadband = param.value
            elif param.name == 'motor_gain_k':        gain_k = param.value
            elif param.name == 'motor_tau':           tau = param.value

        for pid in self._pids:
            pid.update_gains(kp, ki, kd, pwm_max, deadband)

        self._motor_gain_k = gain_k

        for lag in self._motor_lags:
            lag.update_tau(tau)

        self.get_logger().info(
            f'Parámetros actualizados — Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f} '
            f'pwm_max={pwm_max:.1f} deadband={deadband:.3f}mm/s '
            f'motor_gain_k={gain_k:.5f} motor_tau={tau:.4f}s'
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