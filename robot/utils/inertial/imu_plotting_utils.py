import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


WALLS = [
    # Top horizontal corridor
    {"p1": (0.0, -2.0), "p2": (12.0, -2.0)},
    {"p1": (0.0, 2.5), "p2": (2.0, 2.5)},
    {"p1": (5.0, 2.5), "p2": (12.0, 2.5)},
    {"p1": (0.0, -2.0), "p2": (0.0, 2.5)},
    {"p1": (12.0, -2.0), "p2": (12.0, 2.5)},

    # Vertical connector down
    {"p1": (2.0, 2.5), "p2": (2.0, 4.0)},
    {"p1": (5.0, 2.5), "p2": (5.0, 4.0)},
    {"p1": (0.0, 4.0), "p2": (0.0, 8.0)},
    {"p1": (12.0, 4.0), "p2": (12.0, 8.0)},

    # Middle horizontal corridor
    {"p1": (0.0, 4.0), "p2": (2.0, 4.0)},
    {"p1": (5.0, 4.0), "p2": (12.0, 4.0)},
    {"p1": (0.0, 8.0), "p2": (4.0, 8.0)},
    {"p1": (8.0, 8.0), "p2": (12.0, 8.0)},

    # Vertical connector to rooms
    {"p1": (4.0, 8.0), "p2": (4.0, 11.0)},
    {"p1": (8.0, 8.0), "p2": (8.0, 11.0)},

    # Bottom corridor
    {"p1": (0.0, 11.0), "p2": (4.0, 11.0)},
    {"p1": (8.0, 11.0), "p2": (12.0, 11.0)},
    {"p1": (0.0, 13.0), "p2": (4.0, 13.0)},
    {"p1": (8.0, 13.0), "p2": (12.0, 13.0)},
    {"p1": (0.0, 11.0), "p2": (0.0, 13.0)},
    {"p1": (12.0, 11.0), "p2": (12.0, 13.0)},
    {"p1": (5.5, 13.0), "p2": (6.5, 13.0)},

    # Room 1
    {"p1": (0.0, 13.0), "p2": (0.0, 17.0)},
    {"p1": (0.0, 17.0), "p2": (5.5, 17.0)},
    {"p1": (5.5, 13.0), "p2": (5.5, 17.0)},

    # Room 2
    {"p1": (6.5, 13.0), "p2": (6.5, 17.0)},
    {"p1": (6.5, 17.0), "p2": (12.0, 17.0)},
    {"p1": (12.0, 13.0), "p2": (12.0, 17.0)},
]


def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def segments_intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def plot_walls_and_trajectory(X, walls):

    plt.figure(figsize=(6,8))

    # walls
    for w in walls:
        x = [w["p1"][0], w["p2"][0]]
        y = [w["p1"][1], w["p2"][1]]
        plt.plot(x, y, 'k', linewidth=3)

    # trajectory
    plt.plot(X[:,0], X[:,1], 'r-', label="UAV")
    plt.scatter(X[0,0], X[0,1], c='g', label="Start")
    plt.scatter(X[-1,0], X[-1,1], c='b', label="End")

    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("UAV Trajectory with Vertical Barriers")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()


def plot_3d_walls_and_trajectory(
    X,
    walls,
    wall_height=3.0,   # meters (typical indoor ceiling)
    wall_thickness=0.05
):
    """
    X: Nx3 UAV trajectory (meters)
    walls: list of {"p1":(x,y), "p2":(x,y)}
    """

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # -------------------------
    # Draw walls as vertical quads
    # -------------------------
    for w in walls:
        x1, y1 = w["p1"]
        x2, y2 = w["p2"]

        # Direction vector
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0:
            continue

        # Normal for thickness
        nx, ny = -dy / L * wall_thickness, dx / L * wall_thickness

        # Bottom rectangle (extruded)
        p1 = (x1 + nx, y1 + ny, 0.0)
        p2 = (x2 + nx, y2 + ny, 0.0)
        p3 = (x2 - nx, y2 - ny, wall_height)
        p4 = (x1 - nx, y1 - ny, wall_height)

        wall_poly = Poly3DCollection(
            [[p1, p2, p3, p4]],
            facecolor="black",
            alpha=0.15
        )
        ax.add_collection3d(wall_poly)

    # -------------------------
    # Plot UAV trajectory
    # -------------------------
    ax.plot(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        color="red",
        linewidth=2,
        label="UAV trajectory"
    )

    ax.scatter(X[0, 0], X[0, 1], X[0, 2], color="green", s=50, label="Start")
    ax.scatter(X[-1, 0], X[-1, 1], X[-1, 2], color="blue", s=50, label="End")

    # -------------------------
    # Axes + view
    # -------------------------
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D UAV Trajectory with Vertical Barriers")

    ax.view_init(elev=25, azim=135)
    ax.legend()

    # Equal-ish scaling
    max_range = np.array([
        X[:,0].max() - X[:,0].min(),
        X[:,1].max() - X[:,1].min(),
        X[:,2].max() - X[:,2].min()
    ]).max() / 2.0

    mid_x = (X[:,0].max() + X[:,0].min()) / 2.0
    mid_y = (X[:,1].max() + X[:,1].min()) / 2.0
    mid_z = (X[:,2].max() + X[:,2].min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, max(wall_height, mid_z + max_range))

    plt.tight_layout()
    plt.show()
