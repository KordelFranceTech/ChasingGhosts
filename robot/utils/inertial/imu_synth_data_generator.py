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
