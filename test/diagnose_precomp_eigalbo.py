import sys
from pathlib import Path
import os

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import torch
import numpy as np
import matplotlib.pyplot as plt

from heatsplats.utils import load_mesh
from heatsplats.utils import repr_patches

from heatsplats.modules import Mesh, EigenAlboInterpolation


def subspace_distance(V1, V2):
    """
    Calculates the subspace distance between two sets of orthonormal vectors.
    """
    # Ensure columns are orthonormal for robustness
    V1, _ = np.linalg.qr(V1)
    V2, _ = np.linalg.qr(V2)

    # Compute the SVD of the correlation matrix
    sigma = np.linalg.svd(V1.T @ V2, compute_uv=False)

    # Clip to handle potential floating point inaccuracies where sigma > 1
    sigma = np.clip(sigma, 0.0, 1.0)

    # Distance is the 2-norm of the vector of sines of the principal angles
    distance = np.sqrt(np.sum(1 - sigma**2))

    return distance


if __name__ == "__main__":

    fname = "../objects/spot/spot_triangulated.obj"
    # fname = "../objects/bob/bob_tri.obj"

    eigalbo_config = {
        "k_eig": 256,
        "use_precomputed": False,
        "precompute_anisotropies": [1, 20, 40, 60, 80, 100],
        # "precompute_anisotropies": [2.5, 5, 7.5, 10, 25, 50, 75, 100],
        "precompute_angles_every_deg": 30,
        "mesh_path": fname,
        # "precomputed_name": "eigen_albo",
        "distance_weighting": "none",  # "gaussian_0.5",
    }
    device = "cuda:0"

    tri_mesh = load_mesh(fname, merge_tex=True, bake_vert_colors=True)
    our_mesh = Mesh.from_trimesh(tri_mesh, device=device)

    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)

    eigen_vecs = eigalbo_interp._eigen_vec.cpu().numpy()
    eigen_vals = eigalbo_interp._eigen_val.cpu().numpy()

    cart_coords = eigalbo_interp._smp_coords_cartesian.cpu().numpy()

    print("Calculating distances vs. anisotropy...")
    plt.figure(figsize=(12, 8))
    plt.title("Evecs Distance vs. Anisotropy", fontsize=24)

    anisotropies = eigalbo_config["precompute_anisotropies"]
    angles = np.arange(0, 181, eigalbo_config["precompute_angles_every_deg"])

    coord_to_index_map = {(0, 1): 0}
    counter = 1
    for angle in angles:
        for i, scale in enumerate(anisotropies):
            coord_to_index_map[(angle, scale)] = counter
            counter += 1

    # Calculate Distances Along Anisotropy Axis

    for angle in angles:
        distances = []
        for i in range(len(anisotropies) - 1):
            # Find the indices for the two consecutive anisotropy values
            idx1 = coord_to_index_map[(angle, anisotropies[i])]
            idx2 = coord_to_index_map[(angle, anisotropies[i + 1])]

            # Retrieve the eigenvector sets using the indices
            v1 = eigen_vecs[idx1, :, :]
            v2 = eigen_vecs[idx2, :, :]

            dist = subspace_distance(v1, v2)
            distances.append(dist)

        x_labels = [
            f"{anisotropies[i]}-{anisotropies[i+1]}"
            for i in range(len(anisotropies) - 1)
        ]
        plt.plot(
            x_labels, distances, marker="o", linestyle="-", label=f"Angle {angle}°"
        )

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Anisotropy Difference", fontsize=18)
    plt.ylabel("Evecs Distance", fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Calculate Distances Along Angle Axis

    print("Calculating distances vs. angle...")
    plt.figure(figsize=(12, 8))
    plt.title("Evecs Distance vs. Angle", fontsize=24)

    for anisotropy in anisotropies:
        distances = []
        for i in range(len(angles) - 1):
            # Find the indices for the two consecutive angle values
            idx1 = coord_to_index_map[(angles[i], anisotropy)]
            idx2 = coord_to_index_map[(angles[i + 1], anisotropy)]

            # Retrieve the eigenvector sets
            v1 = eigen_vecs[idx1, :, :]
            v2 = eigen_vecs[idx2, :, :]

            dist = subspace_distance(v1, v2)
            distances.append(dist)

        x_labels = [f"{angles[i]}°- {angles[i+1]}°" for i in range(len(angles) - 1)]
        plt.plot(
            x_labels,
            distances,
            marker="o",
            linestyle="-",
            label=f"Anisotropy {anisotropy}",
        )

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Angle Difference", fontsize=18)
    plt.ylabel("Evecs Distance", fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.show()

    # Plot the evals evolution for a fixed angle (e.g., 90 degrees)
    target_angle = 90
    print(f"Plotting eigenvalue evolution for angle = {target_angle}°")

    plt.figure(figsize=(12, 8))
    plt.title(
        f"Eigenvalue Evolution vs. Anisotropy (Angle {target_angle}°)", fontsize=24
    )

    # Get the eigenvalues for the target angle at each anisotropy
    # This will be a matrix of shape [num_anisotropies, K_EIGENVECTORS]
    evals_for_angle = np.array(
        [
            eigen_vals[coord_to_index_map[(target_angle, anisotropy)], :]
            for anisotropy in anisotropies
        ]
    )

    # Plot each eigenvalue's path as a separate line
    for k in range(50):
        plt.plot(anisotropies, evals_for_angle[:, k], marker=".", linestyle="-")

    plt.xlabel("Anisotropy", fontsize=18)
    plt.ylabel("Eigenvalue", fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
