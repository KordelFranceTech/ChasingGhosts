
import math
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def load_uav_records(txt_path: str):
    """Parse OpuInertialJointData-like file into a list of dicts.

    File format example:
        ["k": v, ...]["k": v, ...]...
    We split on '][' and wrap each chunk with { } to parse as JSON.
    """
    raw = Path(txt_path).read_text(errors="ignore")
    parts = raw.split("][")
    records = []
    for i, p in enumerate(parts):
        if i == 0 and p.startswith("["):
            p = p[1:]
        if i == len(parts) - 1 and p.endswith("]"):
            p = p[:-1]
        if not p.strip():
            continue
        try:
            records.append(json.loads("{" + p + "}"))
        except json.JSONDecodeError:
            continue
    return records


def wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def yawrate_from_deg(yaw_prev_deg: float, yaw_curr_deg: float, dt: float) -> float:
    a = math.radians(yaw_prev_deg)
    b = math.radians(yaw_curr_deg)
    dy = wrap_pi(b - a)
    return dy / dt


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# =========================
# EKF-SLAM (2D landmarks) with 3D pose (x,y,yaw,z)
# =========================

# EKF state covariance (process model uncertainty for pose variables)
# [x, y, yaw, z]
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0), 0.5]) ** 2

# Simulation sensor/process noise for synthetic observations and noisy controls
# Observations: range + bearing
Qsim = np.diag([0.2, np.deg2rad(5.0)]) ** 2
# Control noise: v, yawrate, vz
Rsim = np.diag([1.0, np.deg2rad(10.0), 1.0]) ** 2

DT = 0.1
MAX_RANGE = 20.0
M_DIST_TH = 2.0

STATE_SIZE = 4  # [x, y, yaw, z]
LM_SIZE = 2     # landmark [x, y]


def motion_model(x, u):
    """
    x: 4x1 state [x, y, yaw, z]
    u: 3x1 control [v, yawrate, vz]
    """
    F = np.eye(STATE_SIZE)

    yaw = x[2, 0]
    B = np.array([
        [DT * math.cos(yaw), 0.0, 0.0],
        [DT * math.sin(yaw), 0.0, 0.0],
        [0.0, DT, 0.0],
        [0.0, 0.0, DT],
    ])

    return (F @ x) + (B @ u)


def calc_n_LM(x):
    return int((len(x) - STATE_SIZE) / LM_SIZE)


def jacob_motion(x, u):
    """
    Motion Jacobian for pose portion, lifted into the full state via Fx.
    """
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros((STATE_SIZE, LM_SIZE * calc_n_LM(x)))))

    yaw = x[2, 0]
    v = float(u[0, 0])

    # d f / d x for the pose block (STATE_SIZE x STATE_SIZE)
    # x' = x + DT*v*cos(yaw)
    # y' = y + DT*v*sin(yaw)
    # yaw' = yaw + DT*yawrate
    # z' = z + DT*vz
    jF = np.array([
        [0.0, 0.0, -DT * v * math.sin(yaw), 0.0],
        [0.0, 0.0,  DT * v * math.cos(yaw), 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=float)

    G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx
    return G, Fx


def predict(xEst, PEst, u):
    S = STATE_SIZE
    G, Fx = jacob_motion(xEst[0:S], u)
    xEst[0:S] = motion_model(xEst[0:S], u)

    # Pose covariance update
    PEst[0:S, 0:S] = G.T @ PEst[0:S, 0:S] @ G + Fx.T @ Cx @ Fx
    return xEst, PEst, G, Fx


def get_LM_Pos_from_state(x, ind):
    return x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]


def calc_LM_Pos(x, z):
    """
    Landmark world position from measurement z=[range, bearing, id] using pose (x,y,yaw).
    z is 1x3 or 1D array; we use z[0] and z[1].
    """
    zp = np.zeros((2, 1))
    r = float(z[0])
    b = float(z[1])
    yaw = x[2, 0]

    zp[0, 0] = x[0, 0] + r * math.cos(yaw + b)
    zp[1, 0] = x[1, 0] + r * math.sin(yaw + b)
    return zp


def jacobH(q, delta, x, i):
    """
    Measurement Jacobian H for range/bearing in 2D, ignoring z.
    Uses the FIRST THREE pose components [x,y,yaw] and landmark [x,y].
    """
    sq = math.sqrt(q)

    # 2 x 5 Jacobian for [x, y, yaw, lm_x, lm_y] (z not included)
    G = np.array([
        [-sq * delta[0, 0], -sq * delta[1, 0], 0.0,  sq * delta[0, 0],  sq * delta[1, 0]],
        [ delta[1, 0],      -delta[0, 0],     -q,   -delta[1, 0],       delta[0, 0]],
    ], dtype=float) / q

    nLM = calc_n_LM(x)

    # Lift into full state [x, y, yaw, z, lm...]
    # Pose selector: pick x,y,yaw,z (4), but measurement depends only on x,y,yaw.
    # We'll build F by stacking:
    #  - rows selecting pose (STATE_SIZE)
    #  - rows selecting landmark i (2)
    F1 = np.hstack((np.eye(STATE_SIZE), np.zeros((STATE_SIZE, 2 * nLM))))
    F2 = np.hstack((
        np.zeros((LM_SIZE, STATE_SIZE)),
        np.zeros((LM_SIZE, 2 * (i - 1))),
        np.eye(LM_SIZE),
        np.zeros((LM_SIZE, 2 * nLM - 2 * i))
    ))
    F = np.vstack((F1, F2))

    # We need to map the 2x5 G onto the 6x? F that includes z.
    # Construct a 5-row selector that picks [x,y,yaw,lm_x,lm_y] from [x,y,yaw,z,lm...]
    # Indices: x=0, y=1, yaw=2, z=3, lm_x=?, lm_y=?
    # Easier: build H explicitly:
    # - First apply F to get a  (STATE_SIZE + LM_SIZE) x full_state matrix,
    # - then pick columns corresponding to x,y,yaw and landmark.
    # We'll construct a condensed selector C that picks rows [0,1,2, STATE_SIZE, STATE_SIZE+1] from the stacked [pose; landmark].
    stacked_dim = STATE_SIZE + LM_SIZE
    C = np.zeros((5, stacked_dim))
    C[0, 0] = 1.0  # x
    C[1, 1] = 1.0  # y
    C[2, 2] = 1.0  # yaw
    C[3, STATE_SIZE + 0] = 1.0  # lm_x
    C[4, STATE_SIZE + 1] = 1.0  # lm_y

    H = G @ C @ F
    return H


def calc_innovation(lm, xEst, PEst, z, LMid):
    """
    Innovation for 2D range/bearing measurement, based on [x,y,yaw] only.
    """
    # delta uses x,y only
    delta = lm - xEst[0:2]
    q = float((delta.T @ delta)[0, 0])

    zangle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(zangle)]], dtype=float)

    y = (z - zp).T
    y[1] = pi_2_pi(y[1])

    H = jacobH(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + Cx[0:2, 0:2]
    return y, S, H


def search_correspond_LM_ID(xAug, PAug, zi):
    nLM = calc_n_LM(xAug)
    mdist = []
    for i in range(nLM):
        lm = get_LM_Pos_from_state(xAug, i)
        y, S, _H = calc_innovation(lm, xAug, PAug, zi, i)
        mdist.append(float(y.T @ np.linalg.inv(S) @ y))
    mdist.append(M_DIST_TH)
    return mdist.index(min(mdist))


def update(xEst, PEst, u, z, initP):
    for iz in range(len(z[:, 0])):
        minid = search_correspond_LM_ID(xEst, PEst, z[iz, 0:2])
        nLM = calc_n_LM(xEst)

        if minid == nLM:
            xAug = np.vstack((xEst, calc_LM_Pos(xEst, z[iz, :])))
            PAug = np.vstack((
                np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))
            ))
            xEst = xAug
            PEst = PAug

        lm = get_LM_Pos_from_state(xEst, minid)
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], minid)

        K = (PEst @ H.T) @ np.linalg.inv(S)
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])
    return xEst, PEst


def ekf_slam(xEst, PEst, u, z):
    xEst, PEst, _G, _Fx = predict(xEst, PEst, u)
    initP = np.eye(2)
    xEst, PEst = update(xEst, PEst, u, z, initP)
    return xEst, PEst


def observation(xTrue, xd, u, RFID):
    """
    Integrate true pose, generate synthetic range/bearing measurements to 2D beacons,
    and produce dead reckoning using noisy controls.
    """
    xTrue = motion_model(xTrue, u)

    z = np.zeros((0, 3))
    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.sqrt(dx ** 2 + dy ** 2)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Qsim[0, 0]
            anglen = angle + np.random.randn() * Qsim[1, 1]
            zi = np.array([dn, anglen, i], dtype=float)
            z = np.vstack((z, zi))

    # Add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * Rsim[0, 0],
        u[1, 0] + np.random.randn() * Rsim[1, 1],
        u[2, 0] + np.random.randn() * Rsim[2, 2],
    ]]).T

    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud


def add_start_end_markers(ax, xs, ys, zs, label_prefix: str):
    """Add start/end markers and labels to a 3D axis."""
    if xs.size < 1:
        return
    ax.scatter([xs[0]], [ys[0]], [zs[0]], marker="^", s=80, color='k', label=f"{label_prefix} start")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], marker="X", s=90, color='k', label=f"{label_prefix} end")
    ax.text(xs[0], ys[0], zs[0], f"{label_prefix} start")
    ax.text(xs[-1], ys[-1], zs[-1], f"{label_prefix} end")


def main():
    print("start!!")

    uav_txt_path = "/Users/kordelfrance/Documents/School/UTD/PhD Thesis/Ghosts/ChasingGhosts/robot/utils/inertial/OpuInertialJointData_baseline.txt"
    records = load_uav_records(uav_txt_path)
    if len(records) < 2:
        raise RuntimeError(f"Not enough records parsed from {uav_txt_path}")

    # Optional: downsample for speed
    step = 1
    records = records[::step]

    # If your velocities are not meters/sec, scale here.
    VXY_SCALE = 1.0
    VZ_SCALE = 1.0

    # 2D beacons used for synthetic range/bearing measurements (kept from original example)
    RFID = np.array([
        [10.0, -2.0],
        [15.0, 10.0],
        [3.0, 15.0],
        [-5.0, 20.0],
    ], dtype=float)

    # Pose: [x, y, yaw, z]
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    xDR = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    # Initialize yaw from first record (degrees) if present
    if "yaw" in records[0]:
        yaw0 = math.radians(float(records[0]["yaw"]))
        xEst[2, 0] = yaw0
        xTrue[2, 0] = yaw0
        xDR[2, 0] = yaw0

    # History buffers
    hxEst = xEst.copy()
    hxTrue = xTrue.copy()
    hxDR = xDR.copy()

    for k in range(1, len(records)):
        vgx = float(records[k].get("vgx", 0.0)) * VXY_SCALE
        vgy = float(records[k].get("vgy", 0.0)) * VXY_SCALE
        vgz = float(records[k].get("vgz", 0.0)) * VZ_SCALE

        v = math.hypot(vgx, vgy)

        if "yaw" in records[k] and "yaw" in records[k - 1]:
            yawrate = yawrate_from_deg(float(records[k - 1]["yaw"]), float(records[k]["yaw"]), DT)
        else:
            yawrate = 0.0

        u = np.array([[v, yawrate, vgz]]).T

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)
        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

    # **** 3D Plot ****
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot beacons at z=0 for reference
    # ax.scatter(RFID[:, 0], RFID[:, 1], np.zeros(RFID.shape[0]), marker="*", label="RFID beacons")

    ax.plot(hxTrue[0, :], hxTrue[1, :], hxTrue[3, :], label="Measured")
    ax.plot(hxDR[0, :], hxDR[1, :], hxDR[3, :], label="Olf. Inertial Odometry")
    ax.plot(hxEst[0, :], hxEst[1, :], hxEst[3, :], label="OIO + EKF")

    # Start/end markers (icons)
    add_start_end_markers(ax, hxEst[0, :], hxEst[1, :], hxEst[3, :], "EKF")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("UAV 3D Path")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
