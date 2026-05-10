import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 2. Construct the Bunge orientation matrix g.
#    This represents the coordinate transformation from the sample
#    coordinate system to the crystal coordinate system:
#    V_crystal = g @ V_sample
#    g = Rz(phi2) * Rx(Phi) * Rz(phi1)
#    Therefore, g.T transforms vectors from the crystal coordinate
#    system to the sample coordinate system.
# ============================================================
def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0],
                     [-s,  c, 0],
                     [0,  0, 1]])

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,  0,  0],
                     [0,  c, s],
                     [0,  -s,  c]])

def g_rotation(phi1_deg, Phi_deg, phi2_deg):
    phi1 = np.radians(phi1_deg)
    Phi  = np.radians(Phi_deg)
    phi2 = np.radians(phi2_deg)
    return Rz(phi2) @ Rx(Phi) @ Rz(phi1)

def draw_thompson(
        phi1_deg,
        Phi_deg,
        phi2_deg,
        out_path,
        reverse=False
):
    # ============================================================
    #    Vertices of the Thompson tetrahedron in the crystal
    #    coordinate system.
    #    The four {111} planes form a regular tetrahedron.
    #    The selected vertices ensure that the four tetrahedron faces
    #    lie on the (111), (1bar 1bar 1), (1bar 1 1bar),
    #    and (1 1bar 1bar) planes, respectively.
    #    There are two opposite tetrahedra; the reverse parameter
    #    controls which one is used.
    # ============================================================
    V_crystal = np.array([
        [ 1,  1,  1],   # Vertex A
       [ 1, -1, -1],   # Vertex B
       [-1,  1, -1],   # Vertex C
       [-1, -1,  1],   # Vertex D
    ], dtype=float)
    if reverse:
        V_crystal = -V_crystal

    g = g_rotation(phi1_deg, Phi_deg, phi2_deg)

    # Transform to the sample coordinate system.
    V_view = (g.T @ V_crystal.T).T  # shape (4, 3)

    # ============================================================
    # 5. Define tetrahedron faces and outward-facing face normals.
    # ============================================================
    face_defs = [
        (1, 2, 3),   # Opposite vertex A
        (0, 2, 3),   # Opposite vertex B
        (0, 1, 3),   # Opposite vertex C
        (0, 1, 2),   # Opposite vertex D
    ]

    tet_ctr = V_view.mean(axis=0)

    face_normals = []   # Outward face normals in view space.
    for fi in face_defs:
        i0, i1, i2 = fi
        v0, v1, v2 = V_view[i0], V_view[i1], V_view[i2]
        n = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len > 1e-12:
            n /= n_len
        ctr = (v0 + v1 + v2) / 3.0
        if np.dot(n, ctr - tet_ctr) < 0:
            n = -n
        face_normals.append(n)

    # A face points toward the observer if its normal has a positive z component.
    face_front = [n[2] > 0 for n in face_normals]

    # ============================================================
    # 6. Determine the visibility of each edge using the convex
    #    polyhedron rule.
    #    Each edge is shared by exactly two faces:
    #      Both faces front-facing -> solid line, visible
    #      One front-facing and one back-facing -> solid outline, visible
    #      Both faces back-facing -> dashed line, hidden
    # ============================================================
    # Enumerate all six edges and the two faces associated with each edge.
    all_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    def faces_of_edge(e, face_defs):
        """Return the indices of the two faces that contain edge e."""
        result = []
        for fi_idx, fi in enumerate(face_defs):
            if e[0] in fi and e[1] in fi:
                result.append(fi_idx)
        return result

    edge_visibility = {}   # True = solid line, False = dashed line.
    for e in all_edges:
        fi_list = faces_of_edge(e, face_defs)
        fronts = [face_front[f] for f in fi_list]
        # The edge is visible if at least one adjacent face is front-facing.
        edge_visibility[e] = any(fronts)

    # ============================================================
    # 7. Plot the tetrahedron.
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor='none')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)

    # Draw hidden dashed edges first, then visible solid edges,
    # so that solid edges appear on top.
    for solid in [False, True]:
        for e, visible in edge_visibility.items():
            if visible != solid:
                continue
            i, j = e
            p0 = V_view[i, :2]
            p1 = V_view[j, :2]
            if solid:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color='black', linewidth=6,
                        solid_capstyle='round', zorder=10)
            else:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color='black', linewidth=6,
                        linestyle=(0, (4, 3)),   # Dashed line: 4 pt dash, 3 pt gap.
                        solid_capstyle='round', zorder=5)

    # ============================================================
    # 8. Set coordinate limits and save the output.
    # ============================================================
    pts_2d = V_view[:, :2]
    cx = pts_2d[:, 0].mean()
    cy = pts_2d[:, 1].mean()
    span = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    half = span / 2.0 + 0.50

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=250, transparent=True,
                bbox_inches='tight', pad_inches=0.08)
    print(f"Saved to: {out_path}")
    plt.close()

def draw_batch_thompson(script_name, folder_name):
    if not os.path.exists(folder_name):
        # Create the folder if it does not exist.
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")

    fig_num = 1
    with open(script_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines.
            if not line:
                continue

            # Skip comment lines.
            if line.startswith('#'):
                continue

            # Process data lines.
            try:
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid number of elements: {line}")

                phi1_deg, Phi_deg, phi2_deg = map(float, parts)
                out_filename = folder_name + "/" + str(fig_num) + ".png"
                draw_thompson(phi1_deg, Phi_deg, phi2_deg, out_filename, reverse=False)
                out_filename = folder_name + "/" + str(fig_num) + "_rev.png"
                draw_thompson(phi1_deg, Phi_deg, phi2_deg, out_filename, reverse=True)
                fig_num += 1

            except ValueError as e:
                print(f"Skipping invalid line: {line} ({e})")

script_name = "Euler_angles_set.txt"
folder_name = "outputs_set"
draw_batch_thompson(script_name, folder_name)
