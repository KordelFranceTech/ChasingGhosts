import asyncio
import math
import sys
import time

import numpy as np
import pandas as pd
from djitellopy import Tello

import constants
from utils import ble_utils, ml_utils


# ── Reference constants (kept for documentation / battery-longevity baseline) ──
course_commands: list = [
    "takeoff",
    "forward 565",
    "cw 90",
    "forward 400",
    "ccw 90",
    "forward 343",
    "cw 90",
    "forward 154",
    "forward 700",
    "forward 170",
    "ccw 90",
    "forward 145",
    "cw 90",
    "forward 500",
    "ccw 90",
    "forward 120",
    "land",
]

io_example: dict = {
    "ENV_temperatureC": 24,
    "ENV_humidity": 62,
    "ENV_pressureHpa": 1010,
    "STATUS_opuA": 0,
    "BATT_health": 98,
    "BATT_v": 4.5,
    "BATT_charge": 83,
    "BATT_time": 2,
    "CO2": 1058,
    "CO_A": 720,
    "NH3_A": 93,
    "NO2_A": 5.68,
    "CO_B": 720,
    "NH3_B": 93,
    "NO2_B": 5.68,
    "ToF_mm": 100.0,
}


# ── OIO Algorithm Parameters (paper Section V.B) ────────────────────────────
# Bout-detection EMA spans (empirically selected; constraint: alpha < rho < beta)
EMA_ALPHA: int = 3      # fast EMA span — responds to odor onset
EMA_BETA: int = 8       # slow EMA span — tracks background trend
EMA_RHO: int = 5        # signal-line span for smoothing D

STEP_CM: int = 30                       # discrete movement step, cm
TOF_OBSTACLE_THRESHOLD_M: float = 0.20  # forward ToF obstacle gate, m
PHI_DEADZONE_DEG: float = 5.0           # ±5° deadzone → surge (hold heading)
SENSOR_BASELINE_M: float = 0.15         # bilateral sensor separation d, m
WIND_SPEED_EST: float = 0.3             # plume advection speed estimate, m/s

# Source acceptance — German Tank Problem (Equations 9-11)
SOURCE_CI_P_LOW: float = 0.025          # 95% CI lower tail
SOURCE_ACCEPT_MARGIN: float = 0.25      # land when within 25% of expected max


# ── Signal Processing ────────────────────────────────────────────────────────

def _ema(series: list, span: int) -> float:
    """Exponential moving average over an oldest-first series."""
    alpha = 2.0 / (span + 1)
    val = float(series[0])
    for s in series[1:]:
        val = alpha * float(s) + (1.0 - alpha) * val
    return val


def compute_D_S(series: list) -> tuple:
    """
    Compute bout-detection signals D and S (paper Equations 5–7).

      D(t) = E_alpha(C) − E_beta(C)   [temporal-prediction-difference filter]
      S(t) = E_rho(D)                 [signal line; smoothed expectation of D]

      D > S  →  accelerating onset   →  plume entry  →  surge (exploit)
      D < S  →  decaying trend       →  plume loss   →  cast  (explore)

    Args:
        series: chronological concentration values, oldest first.
    Returns:
        (D, S) as floats.
    """
    if len(series) < 2:
        return 0.0, 0.0

    # Build history of D values so we can compute S = E_rho(D)
    D_history: list = []
    for i in range(1, len(series) + 1):
        sub = series[:i]
        D_history.append(_ema(sub, EMA_ALPHA) - _ema(sub, EMA_BETA))

    D = D_history[-1]
    S = _ema(D_history, EMA_RHO)
    return D, S


def get_sensor_series(df: pd.DataFrame, compound: str, side: str) -> list:
    """
    Return chronological (oldest-first) concentration list from io_data.
    UAV_IO_FRAME is prepended on each BLE sample, so row 0 is most recent.
    """
    col = f"{compound}_{side}"
    if col not in df.columns:
        return []
    vals = df[col].dropna().tolist()
    vals.reverse()
    return vals


# ── State Classification ─────────────────────────────────────────────────────

_LATERAL_EQUAL_TOL: float = 5.0        # units within which left ≈ right
_PLUME_HIGH_THRESHOLD: float = 10.0    # D−S magnitude separating low vs. high contact

def classify_state(D_L: float, S_L: float, D_R: float, S_R: float,
                   c_left: float, c_right: float) -> int:
    """
    9-state navigation classifier (Table VIII of paper).

    Lateral axis (columns):
      0: left < right      1: left ≈ right      2: left > right

    Plume contact axis (rows):
      base 0: off-plume   (both D < S)
      base 3: on-plume low
      base 6: on-plume high

    State = lateral_col + plume_base
    """
    diff = c_left - c_right
    if abs(diff) <= _LATERAL_EQUAL_TOL:
        lat = 1
    elif diff < 0:
        lat = 0
    else:
        lat = 2

    on_L = D_L > S_L
    on_R = D_R > S_R
    if not on_L and not on_R:
        plume_base = 0
    else:
        divergence = max(abs(D_L - S_L), abs(D_R - S_R))
        plume_base = 6 if divergence >= _PLUME_HIGH_THRESHOLD else 3

    return lat + plume_base


# ── Heading Estimation ───────────────────────────────────────────────────────

def estimate_phi(left_series: list, right_series: list) -> float:
    """
    Estimate plume incident angle phi via cross-correlation time-delay (Eq. 3–4).

      tau  = lag (s) at peak cross-correlation of bilateral channels
      phi  = arcsin(tau · wind_speed / sensor_baseline)  [degrees]

    Deadzone: |phi| < 5° → hold heading and surge.
    """
    n = min(len(left_series), len(right_series))
    if n < 3:
        return 0.0

    l = np.array(left_series[-n:], dtype=float)
    r = np.array(right_series[-n:], dtype=float)

    corr = np.correlate(l - l.mean(), r - r.mean(), mode="full")
    lags = np.arange(-(n - 1), n)
    tau_sec = float(lags[np.argmax(corr)])  # 1 Hz sampling → 1 sample = 1 s

    arg = float(np.clip(tau_sec * WIND_SPEED_EST / SENSOR_BASELINE_M, -1.0, 1.0))
    return math.degrees(math.asin(arg))


# ── OIO Navigation Policy ────────────────────────────────────────────────────

def oio_policy(phi: float,
               D_L: float, S_L: float,
               D_R: float, S_R: float) -> str:
    """
    OIO discrete action policy (paper Section V.C).

    On-plume (D > S on either sensor):
      |phi| < 5°         → surge (gradient straight ahead)
      5° ≤ |phi| ≤ 45°  → soft turn toward source (45°)
      |phi| > 45°        → hard turn toward source (90°)

    Off-plume (both D < S):
      cast left or right (±45° + forward step) to relocate the plume

    Actions:
      'surge'           move forward STEP_CM
      'turn_left_soft'  rotate CCW 45°
      'turn_right_soft' rotate CW  45°
      'turn_left_hard'  rotate CCW 90°
      'turn_right_hard' rotate CW  90°
      'cast_left'       45° CCW + forward (exploratory sweep)
      'cast_right'      45° CW  + forward (exploratory sweep)
      'land'            source found — land
    """
    on_plume = (D_L > S_L) or (D_R > S_R)

    if on_plume:
        if abs(phi) < PHI_DEADZONE_DEG:
            return "surge"
        elif phi < -45:
            return "turn_left_hard"
        elif phi < -PHI_DEADZONE_DEG:
            return "turn_left_soft"
        elif phi > 45:
            return "turn_right_hard"
        else:
            return "turn_right_soft"
    else:
        return "cast_left" if phi <= 0 else "cast_right"


# ── Source Acceptance ─────────────────────────────────────────────────────────

def check_source_acceptance(concentration_history: list) -> bool:
    """
    German Tank Problem source acceptance criterion (paper Equations 9–11).

      C_hat   = m · k/(k+1) − 1          [point estimate of true maximum]
      CI_upper = m / p_low^(1/k)          [95% CI upper bound; p_low = 0.025]

    Land (declare source found) when the observed maximum m is within 25% of
    the CI upper bound — i.e., the UAV cannot meaningfully increase its reading.
    """
    k = len(concentration_history)
    if k < 5:
        return False
    m = float(max(concentration_history))
    if m == 0.0:
        return False
    ci_upper = m / (SOURCE_CI_P_LOW ** (1.0 / k))
    return m >= ci_upper * (1.0 - SOURCE_ACCEPT_MARGIN)


# ── Action Execution ──────────────────────────────────────────────────────────

def stable_rotate(degrees: int, clockwise: bool, step: int = 30, settle_threshold: int = 3, settle_timeout: float = 3.0):
    """Rotate in small increments and wait for IMU to settle between steps.
    Avoids 'no valid IMU' errors caused by payload-induced vibration during
    high-angular-acceleration single-command rotations.
    Args:
        degrees: total rotation in degrees (positive integer)
        clockwise: True for CW, False for CCW
        step: degrees per increment (smaller = gentler, slower)
        settle_threshold: max abs pitch/roll in degrees before proceeding
        settle_timeout: seconds to wait for settle before giving up
    """
    remaining = degrees
    while remaining > 0:
        increment = min(step, remaining)
        if clockwise:
            tello.rotate_clockwise(increment)
        else:
            tello.rotate_counter_clockwise(increment)
        remaining -= increment
        deadline = time.time() + settle_timeout
        while time.time() < deadline:
            s = tello.get_current_state()
            if abs(s.get('pitch', 99)) <= settle_threshold and abs(s.get('roll', 99)) <= settle_threshold:
                break
            time.sleep(0.1)
        time.sleep(0.5)


def execute_action(action: str, tof_m: float):
    """
    Execute a navigation action.
    All forward moves are gated by the forward-looking ToF obstacle sensor.
    Uses module-level `tello` set at startup when FLIGHT_MODE is True.
    """
    def _cmd(label: str, fn):
        if constants.FLIGHT_MODE:
            fn()
        else:
            print(f"[DRY RUN] {label}")

    def _fwd():
        if tof_m > TOF_OBSTACLE_THRESHOLD_M:
            _cmd(f"tello.move_forward({STEP_CM})", lambda: tello.move_forward(STEP_CM))
        else:
            print(f"[OBSTACLE] ToF={tof_m:.3f} m — backing up instead")
            _cmd(f"tello.move_back({STEP_CM})", lambda: tello.move_back(STEP_CM))

    if action == "surge":
        _fwd()
    elif action == "turn_left_soft":
        _cmd("stable_rotate(45, clockwise=False)", lambda: stable_rotate(45, clockwise=False))
    elif action == "turn_right_soft":
        _cmd("stable_rotate(45, clockwise=True)", lambda: stable_rotate(45, clockwise=True))
    elif action == "turn_left_hard":
        _cmd("stable_rotate(90, clockwise=False)", lambda: stable_rotate(90, clockwise=False))
    elif action == "turn_right_hard":
        _cmd("stable_rotate(90, clockwise=True)", lambda: stable_rotate(90, clockwise=True))
    elif action in ("cast_left", "cast_right"):
        # Casting: turn 45° then move forward to sweep and relocate the plume
        if action == "cast_left":
            _cmd("stable_rotate(45, clockwise=False)", lambda: stable_rotate(45, clockwise=False))
        else:
            _cmd("stable_rotate(45, clockwise=True)", lambda: stable_rotate(45, clockwise=True))
        time.sleep(constants.STEP_TIME)
        _fwd()
    elif action == "land":
        print("[SOURCE FOUND] Declaring source — landing.")
        _cmd("tello.land()", lambda: tello.land())


# ── Last-Mile Vision (Component 2 — YOLO fast update, 1–2 Hz) ────────────────

def navigate_to_source_vision():
    """
    YOLO-based last-mile visual navigation (paper Section IV.B, Component 2).

    Runs synchronized to the olfaction sampling rate (1–2 Hz).
    Activated only when the olfactory gradient is exhausted near the source
    (UAV is on-plume but physically blocked — gradient has flattened).

    Converts YOLO bounding-box centroids into lateral alignment corrections,
    then advances toward the target once aligned.
    """
    from io_data import UAV_IO_FRAME
    frame_read = tello.get_frame_read()
    yolo_results = ml_utils.infer_doorways(
        uav_camera_frame=frame_read, is_test=False, should_display=False
    )
    if yolo_results is None:
        return
    door = ml_utils.get_nearest_door(yolo_results=yolo_results)
    if door is None:
        return
    door_cx, _ = ml_utils.get_centroid_of_nearest_door(door_xywhn=door)
    tof_m = float(UAV_IO_FRAME.iloc[0]["TOF_mm"])

    if float(door_cx) < 0.4:
        tello.move_left(5)
    elif float(door_cx) > 0.6:
        tello.move_right(5)
    else:
        if tof_m > TOF_OBSTACLE_THRESHOLD_M:
            tello.move_forward(STEP_CM)
        else:
            tello.move_back(STEP_CM)


# ── OIO Command Loop ──────────────────────────────────────────────────────────

def command_loop_oio(use_last_mile_vision:bool=False):
    """
    Single OIO navigation step implementing the full paper algorithm:

      1. Read bilateral sensor series from io_data (populated by BLE sampling)
      2. Compute D/S divergence per sensor for bout detection (Eq. 5–7)
      3. Classify navigation state into 9-state table (Table VIII)
      4. Estimate plume heading angle phi via cross-correlation (Eq. 3–4)
      5. Evaluate German Tank Problem source acceptance criteria (Eq. 9–11)
      6. Select discrete action via OIO policy (Section V.C)
      7. Gate forward actions against forward-looking ToF sensor
      8. Execute action
      9. Activate YOLO last-mile vision when on-plume gradient is exhausted
    """
    from io_data import UAV_IO_FRAME

    compound = constants.TARGET_COMPOUND

    # 1. Sensor series — oldest → newest
    left_series = get_sensor_series(UAV_IO_FRAME, compound, "A")
    right_series = get_sensor_series(UAV_IO_FRAME, compound, "B")
    if not left_series or not right_series:
        print("[OIO] No sensor data yet — skipping.")
        return

    c_left = left_series[-1]
    c_right = right_series[-1]

    # 2. Bout detection: D/S divergence per channel
    D_L, S_L = compute_D_S(left_series)
    D_R, S_R = compute_D_S(right_series)

    # 3. 9-state classification
    state = classify_state(D_L, S_L, D_R, S_R, c_left, c_right)

    # 4. Heading angle from cross-correlation time-delay
    phi = estimate_phi(left_series, right_series)

    print(
        f"\t[OLF]   left={c_left:.1f}  right={c_right:.1f}\n"
        f"\t[D/S]   D_L={D_L:.2f} S_L={S_L:.2f} | D_R={D_R:.2f} S_R={S_R:.2f}\n"
        f"\t[STATE] {state}  [PHI] {phi:.1f}°"
    )

    # 5. Source acceptance — German Tank Problem
    if check_source_acceptance(left_series + right_series):
        print("[ACCEPT] Source acceptance criteria met.")
        execute_action("land", tof_m=999.0)
        return

    # 6-8. Policy ->  obstacle gate ->  execute
    action = oio_policy(phi, D_L, S_L, D_R, S_R)
    print(f"\t[ACTION] {action}")
    tof_m = float(UAV_IO_FRAME.iloc[0]["TOF_mm"])
    execute_action(action, tof_m)

    # 9. Last-mile YOLO (Component 2): activate when on-plume but path blocked
    on_plume = (D_L > S_L) or (D_R > S_R)
    if on_plume and action == "surge" and tof_m <= TOF_OBSTACLE_THRESHOLD_M:
        if constants.FLIGHT_MODE and use_last_mile_vision:
            navigate_to_source_vision()
        else:
            print("[DRY RUN] navigate_to_source_vision()  [last-mile YOLO]")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Connect to OPU over BLE
    target_device = ble_utils.connect_to_sensor()
    if target_device is None:
        sys.exit()
    time.sleep(constants.STEP_TIME)

    # Initialize Tello
    if constants.FLIGHT_MODE:
        tello = Tello()
        tello.connect()
        tello.stream_on()
        tello.takeoff()
        time.sleep(constants.STEP_TIME)
    else:
        print("\n**** INIT (DRY RUN) ****")
        print("tello.connect() | tello.stream_on() | tello.takeoff()")

    # Prime the sensor buffer — need ≥2 samples before bout detection is valid
    print("\n**** PRIMING SENSOR BUFFER ****")
    asyncio.run(ble_utils.async_sample_from_device(target_device))
    time.sleep(constants.STEP_TIME)
    asyncio.run(ble_utils.async_sample_from_device(target_device))
    time.sleep(constants.STEP_TIME)

    try:
        step = 0
        while True:
            step += 1
            print(f"\n===== OIO STEP {step} =====")

            # Sample olfaction sensors via BLE — updates io_data.UAV_IO_FRAME
            asyncio.run(ble_utils.async_sample_from_device(target_device))

            if not constants.DEBUG_MODE:
                command_loop_oio()

            time.sleep(constants.STEP_TIME)

    finally:
        if constants.FLIGHT_MODE:
            tello.land()
            tello.streamoff()
            tello.end()
        else:
            print("\n**** SHUTDOWN (DRY RUN) ****")
            print("tello.land() | tello.streamoff() | tello.end()")
