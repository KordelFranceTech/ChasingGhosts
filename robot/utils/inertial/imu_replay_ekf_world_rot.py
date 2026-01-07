import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Rotation utilities
# -------------------------

def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def body_to_world_rotation(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

# -------------------------
# EKF
# -------------------------

def ekf_run(data, tof_scale_m=0.01):
    g = 9.80665

    t = np.array([d["time"] for d in data])
    dt = np.diff(t, prepend=t[0])

    # accel in m/s^2
    a_body = np.array([
        [d["agx"], d["agy"], d["agz"]] for d in data
    ]) / 1000.0 * g

    roll  = np.deg2rad([d["roll"] for d in data])
    pitch = np.deg2rad([d["pitch"] for d in data])
    yaw   = np.deg2rad([d["yaw"] for d in data])

    v_meas = np.array([
        [d["vgx"], d["vgy"], d["vgz"]] for d in data
    ])

    z_tof = np.array([d["tof"] for d in data]) * tof_scale_m

    # State: [x y z vx vy vz bax bay baz]
    xhat = np.zeros(9)
    P = np.eye(9) * 0.3

    sigma_a = 1.5
    sigma_b = 0.1
    sigma_v = 0.3
    sigma_tof = 0.15

    X = np.zeros((len(data), 9))
    I = np.eye(9)

    for k in range(1, len(data)):
        # --- Rotate accel ---
        R = body_to_world_rotation(roll[k], pitch[k], yaw[k])
        a_world = R @ a_body[k]
        a_world[2] += g  # remove gravity

        # --- Predict ---
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt[k]
        F[3:6, 6:9] = -np.eye(3) * dt[k]

        pos = xhat[0:3]
        vel = xhat[3:6]
        bias = xhat[6:9]

        pos_p = pos + vel * dt[k]
        vel_p = vel + (a_world - bias) * dt[k]

        xhat = np.hstack([pos_p, vel_p, bias])

        Q = np.diag([
            *(0.25 * dt[k]**4 * sigma_a**2 for _ in range(3)),
            *(dt[k]**2 * sigma_a**2 for _ in range(3)),
            *(dt[k]**2 * sigma_b**2 for _ in range(3))
        ])

        P = F @ P @ F.T + Q

        # --- Velocity update ---
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)
        Rv = np.eye(3) * sigma_v**2

        y = v_meas[k] - H @ xhat
        S = H @ P @ H.T + Rv
        K = P @ H.T @ np.linalg.inv(S)

        xhat += K @ y
        P = (I - K @ H) @ P

        # --- TOF altitude update ---
        if z_tof[k] > 0:
            H = np.zeros((1, 9))
            H[0, 2] = 1
            Rt = np.array([[sigma_tof**2]])

            y = np.array([z_tof[k] - xhat[2]])
            S = H @ P @ H.T + Rt
            K = P @ H.T @ np.linalg.inv(S)

            xhat += (K @ y).flatten()
            P = (I - K @ H) @ P

        X[k] = xhat

    return X

# -------------------------
# Plot
# -------------------------

def plot_trajectory(X):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:,0], X[:,1], X[:,2], linewidth=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("EKF UAV Trajectory (Aggressive Maneuvers)")
    plt.show()



if __name__ == "__main__":
    # Put your JSON list into a file like imu_log.json
    # data = load_json_list("./imu_log.json")
    import imu_synth_data_generator
    data = imu_synth_data_generator.generate_aggressive_log()

    # If your TOF is already in meters, set tof_scale_m=1.0
    X = ekf_run(
        data,
        tof_scale_m=0.01,
    )

    plot_trajectory(X)
