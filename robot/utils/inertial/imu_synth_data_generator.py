import numpy as np


def generate_aggressive_log(n=100, dt=0.05):
    data = []
    t = 0.0
    v = np.zeros(3)
    z = 30.0

    for k in range(n):
        roll = 35*np.sin(0.2*k)
        pitch = 30*np.sin(0.15*k)
        yaw = (k*7) % 360

        agx = 600*np.sin(0.3*k)
        agy = 450*np.cos(0.25*k)
        agz = -900 + 120*np.sin(0.4*k)

        v += np.array([agx, agy, agz]) * 1e-3 * 9.8 * dt
        z += v[2]*dt*100

        data.append({
            "templ":95.0,
            "temph":97.0,
            "roll":round(roll,2),
            "bat":round(8.0 - 0.002*k,3),
            "h":round(0.01*k,2),
            "baro":round(240 + 0.01*k,2),
            "time":round(t,2),
            "agz":round(agz,1),
            "agy":round(agy,1),
            "vgy":round(v[1],2),
            "yaw":round(yaw,2),
            "agx":round(agx,1),
            "vgz":round(v[2],2),
            "vgx":round(v[0],2),
            "tof":round(max(z,5),1),
            "pitch":round(pitch,2)
        })
        t += dt
    return data


def generate_aggressive_log_with_wind(
    n=100,
    dt=0.05,
    wind_amp=(1.2, 0.8, 0.3),      # m/s
    wind_freq=(0.08, 0.05, 0.03), # Hz
    gust_std=0.15                # m/s random gusts
):
    data = []
    t = 0.0

    v_body = np.zeros(3)
    v_wind = np.zeros(3)
    z_cm = 40.0

    for k in range(n):
        # --- Attitude (aggressive) ---
        roll  = 35 * np.sin(0.22 * k)
        pitch = 30 * np.sin(0.17 * k)
        yaw   = (k * 9) % 360

        # --- Body-frame accelerations (mg) ---
        agx = 650 * np.sin(0.35 * k)
        agy = 500 * np.cos(0.30 * k)
        agz = -900 + 150 * np.sin(0.45 * k)

        # --- Wind field (world frame, m/s) ---
        v_wind = np.array([
            wind_amp[0] * np.sin(2*np.pi*wind_freq[0]*t),
            wind_amp[1] * np.cos(2*np.pi*wind_freq[1]*t),
            wind_amp[2] * np.sin(2*np.pi*wind_freq[2]*t)
        ]) + np.random.normal(0, gust_std, 3)

        # --- Integrate body accel to body velocity ---
        a_body = np.array([agx, agy, agz]) * 1e-3 * 9.80665
        v_body += a_body * dt

        # --- Measured velocity = body + wind ---
        v_meas = v_body + v_wind

        # --- Altitude ---
        z_cm += v_meas[2] * dt * 100

        data.append({
            "templ": 95.0 + 0.02*np.sin(k),
            "temph": 97.0 + 0.02*np.cos(k),
            "roll": round(roll, 2),
            "bat": round(8.0 - 0.0025*k, 3),
            "h": round(0.01*k, 2),
            "baro": round(240 + 0.02*k, 2),
            "time": round(t, 2),
            "agz": round(agz, 1),
            "agy": round(agy, 1),
            "vgy": round(v_meas[1], 2),
            "yaw": round(yaw, 2),
            "agx": round(agx, 1),
            "vgz": round(v_meas[2], 2),
            "vgx": round(v_meas[0], 2),
            "tof": round(max(z_cm, 5), 1),
            "pitch": round(pitch, 2)
        })
        t += dt
    return data
