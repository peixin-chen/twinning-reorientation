import numpy as np
import matplotlib.pyplot as plt

from orix import plot  # Register orix' projections with Matplotlib
from orix.vector import Vector3d

"""
T0 = E
Tn+1 = Tn @ (E - 2*ai@ai.T)
"""

a = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1]
], dtype=float)
for i in range(4):
    a[i] = a[i] / np.linalg.norm(a[i])

E = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

E_aaT2 = []
for ai in a:
    E_aaT2.append(E - 2 * np.outer(ai, ai))

Tn = []
Tn.append([E])

depth = 4

for n in range(depth):
    temp_container = []
    for T in Tn[n]:
        for E_aaT2i in E_aaT2:
            temp_container.append(T @ E_aaT2i)
    Tn.append(temp_container)


directions_111 = []
for T_set in Tn:
    temp_container = []
    for T in T_set:
        for ai in a:
            temp_container.append(T @ ai)
    directions_111.append(temp_container)

fig_num = 0
for directions in directions_111:

    plt.rcParams.update(
        {
            "figure.figsize": (5, 5),
            "lines.markersize": 10,
            "font.size": 20,
            "axes.grid": True,
            "lines.markersize": 7,
        }
    )

    dirs = np.array(directions)

    mask = dirs[:, 2] < 0
    dirs[mask] *= -1

    v1 = Vector3d(dirs)
    labels = ["x", "y", None]
    v1.scatter(axes_labels=labels)
    fig_name = "plot_" + str(fig_num) + ".png"
    plt.savefig(fig_name, dpi=300)
    plt.close()
    fig_num += 1

