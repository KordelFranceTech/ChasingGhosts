import json
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load JSON data
# ----------------------------

with open("./imu_log.json", "r") as f:
    data = json.load(f)

# Convert list of dicts to arrays
time = np.array([d["time"] for d in data])
vgx  = np.array([d["vgx"] for d in data])
vgy  = np.array([d["vgy"] for d in data])
vgz  = np.array([d["vgz"] for d in data])

# ----------------------------
# Integrate velocity → position
# ----------------------------

x = np.zeros(len(time))
y = np.zeros(len(time))
z = np.zeros(len(time))

for i in range(1, len(time)):
    dt = time[i] - time[i - 1]

    x[i] = x[i - 1] + vgx[i] * dt
    y[i] = y[i - 1] + vgy[i] * dt
    z[i] = z[i - 1] + vgz[i] * dt

# ----------------------------
# Plot 3D Trajectory
# ----------------------------

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(x, y, z, label="UAV Trajectory", linewidth=2)
ax.scatter(x[0], y[0], z[0], color="green", s=50, label="Start")
ax.scatter(x[-1], y[-1], z[-1], color="red", s=50, label="End")

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Inferred UAV Position from Inertial Data")
ax.legend()

plt.tight_layout()
plt.show()
