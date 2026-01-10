import json
import numpy as np
import matplotlib.pyplot as plt

import imu_plotting_utils as plt_utils


def load_json_list(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Expected a non-empty JSON list of snapshots.")
    return data


def as_array(data, key, default=0.0):
    return np.array([float(d.get(key, default)) for d in data], dtype=float)


def ekf_run(data,
            use_velocity_meas=True,
            use_tof_meas=True,
            tof_scale_m=0.01,  # if tof is in "cm", 0.01 converts cm->m. If it's mm, use 0.001.
            vel_meas_gate=1e-6 # treat near-zero velocity as absent if you want
           ):
    """
    State: [x, y, z, vx, vy, vz, bax, bay, baz]^T  (meters, m/s, m/s^2)
    Process:
      x_{k+1} = x_k + v_k dt
      v_{k+1} = v_k + (a_k - b_k) dt
      b_{k+1} = b_k + w_b
    Measurements:
      z_vel = [vx, vy, vz] (from vgx,vgy,vgz if enabled)
      z_tof = z (from tof if enabled)
    """

    # ---- Extract signals ----
    t = as_array(data, "time")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time must be strictly increasing.")

    # accelerations: assume mg -> m/s^2
    g = 9.80665
    ax = as_array(data, "agx") / 1000.0 * g
    ay = as_array(data, "agy") / 1000.0 * g
    az = as_array(data, "agz") / 1000.0 * g

    # Optional measurements
    vgx = as_array(data, "vgx")
    vgy = as_array(data, "vgy")
    vgz = as_array(data, "vgz")
    tof = as_array(data, "tof") * tof_scale_m  # convert to meters

    n = len(t)

    # ---- EKF init ----
    xhat = np.zeros(9)  # start at origin, zero velocity, zero bias

    # Covariance (tune these!)
    P = np.diag([
        0.5, 0.5, 0.5,      # pos (m^2)
        0.5, 0.5, 0.5,      # vel ((m/s)^2)
        0.2, 0.2, 0.2       # bias ((m/s^2)^2)
    ])

    # Process noise (tune)
    # accel noise (m/s^2), bias random walk (m/s^2 per sqrt(s))
    sigma_a = 0.8
    sigma_b = 0.05

    # Measurement noise (tune)
    sigma_v = 0.25   # m/s
    sigma_tof = 0.10 # m (10 cm)

    # Storage
    X = np.zeros((n, 9))
    X[0] = xhat

    I = np.eye(9)

    for k in range(1, n):
        dt = t[k] - t[k - 1]

        # ---- Predict step ----
        # State transition Jacobian F
        F = np.eye(9)
        # x depends on v
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # v depends on bias (minus)
        F[3, 6] = -dt
        F[4, 7] = -dt
        F[5, 8] = -dt

        # Control input (accel)
        a = np.array([ax[k], ay[k], az[k]], dtype=float)

        # Predict state
        pos = xhat[0:3]
        vel = xhat[3:6]
        b   = xhat[6:9]

        pos_pred = pos + vel * dt
        vel_pred = vel + (a - b) * dt
        b_pred   = b  # random walk handled via Q

        xhat = np.hstack([pos_pred, vel_pred, b_pred])

        # Process noise covariance Q (simple discrete approximation)
        # Position noise from accel: ~ (0.5*dt^2)^2 * sigma_a^2
        q_p = (0.5 * dt * dt)**2 * sigma_a**2
        q_v = (dt)**2 * sigma_a**2
        q_b = (dt)**2 * sigma_b**2

        Q = np.diag([
            q_p, q_p, q_p,
            q_v, q_v, q_v,
            q_b, q_b, q_b
        ])

        P = F @ P @ F.T + Q

        # ---- Update step(s) ----
        # 1) Velocity measurement update (if enabled and meaningful)
        if use_velocity_meas:
            z = np.array([vgx[k], vgy[k], vgz[k]], dtype=float)

            # optionally skip if velocities are essentially all zero
            if np.linalg.norm(z) > vel_meas_gate:
                H = np.zeros((3, 9))
                H[0, 3] = 1.0
                H[1, 4] = 1.0
                H[2, 5] = 1.0

                R = np.diag([sigma_v**2, sigma_v**2, sigma_v**2])

                y = z - (H @ xhat)  # innovation
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)

                xhat = xhat + K @ y
                P = (I - K @ H) @ P

        # 2) TOF altitude update (z position)
        if use_tof_meas:
            z_t = float(tof[k])

            # If tof is 0 or invalid you can gate it out here; keeping simple:
            if z_t > 0.0:
                H = np.zeros((1, 9))
                H[0, 2] = 1.0  # measure z

                R = np.array([[sigma_tof**2]])

                y = np.array([z_t - (H @ xhat)[0]])
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)

                xhat = xhat + (K @ y).reshape(-1)
                P = (I - K @ H) @ P

        X[k] = xhat

    return t, X


def plot_3d_trajectory(X, title="EKF Estimated UAV Position"):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=2, label="EKF trajectory")
    ax.scatter(x[0], y[0], z[0], s=50, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], s=50, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Put your JSON list into a file like imu_log.json
    # data = load_json_list("./imu_log.json")
    import imu_synth_data_generator
    data = imu_synth_data_generator.generate_aggressive_log_with_wind()

    # If your TOF is already in meters, set tof_scale_m=1.0
    t, X = ekf_run(
        data,
        use_velocity_meas=True,
        use_tof_meas=True,
        tof_scale_m=0.01,     # common if tof is cm
        vel_meas_gate=1e-6
    )

    plot_3d_trajectory(X, title="EKF Estimated UAV Position (pos/vel/bias)")

    walls = plt_utils.WALLS
    plt_utils.plot_3d_walls_and_trajectory(X, walls)