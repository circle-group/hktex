import sys
from pathlib import Path
import os

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import torch
import trimesh
import numpy as np
import scipy.io as sio

import h5py
from scipy.sparse import csc_matrix
from scipy.sparse import isspmatrix_csc

from hktex.modules import Mesh
from hktex.utils import (
    load_mesh,
    get_anisotropic_lbo,
    compute_mesh_laplacian,
    compute_eig_laplacian,
    heat_diffusion,
    soft_step,
    combine_images,
    show_image,
)
from hktex.rendering.vertex_colours_renderer import VertexColoursRenderer


def load_albo_matrices(filepath):
    """
    Loads W and A matrices from a MATLAB v7.3 .mat file.

    In MATLAB, the data was saved as two cell arrays, 'Ws' and 'As',
    containing sparse matrices. This function reads the HDF5-formatted
    .mat file and reconstructs these matrices in Python.

    Args:
        filepath (str): The full path to the .mat file.

    Returns:
        tuple: A tuple containing two lists:
            - Ws (list): A list of NumPy arrays for the W matrices.
            - As (list): A list of NumPy arrays for the A matrices.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        KeyError: If the .mat file does not contain 'Ws' or 'As' variables.
    """
    print(f"Loading data from: {filepath}")

    Ws = []
    As = []

    try:
        with h5py.File(filepath, "r") as f:
            # Check if the required variables exist in the file
            if "Ws" not in f or "As" not in f:
                raise KeyError("File must contain 'Ws' and 'As' cell arrays.")

            # Get the references to the cell array contents
            ws_refs = f["Ws"]
            as_refs = f["As"]

            print(f"Found {len(ws_refs)} W matrices and {len(as_refs)} A matrices.")

            # MATLAB cell arrays are stored as arrays of object references in HDF5.
            # We need to iterate through these references to get the actual data.
            for i in range(len(ws_refs)):
                # Get the reference for the current matrix
                w_ref = ws_refs[i, 0]
                a_ref = as_refs[i, 0]

                # Use the reference to access the matrix data object in the HDF5 file
                w_obj = f[w_ref]
                a_obj = f[a_ref]

                # In MATLAB v7.3, sparse matrices are often saved as a struct
                # with 'data', 'ir' (row indices), and 'jc' (column pointers).
                # If they are saved as dense, they will be simple numpy arrays.
                # Here we handle the sparse case, which is common.
                if isinstance(w_obj, h5py.Group) and "data" in w_obj:
                    # This is a sparse matrix saved as a struct
                    print(f"  - Loading sparse matrix {i+1}...")
                    w_data = w_obj["data"][:]
                    w_ir = w_obj["ir"][:]
                    w_jc = w_obj["jc"][:]
                    dims = w_obj["jc"].shape[0] - 1

                    # Reconstruct the sparse matrix (CSC format)
                    # Note: MATLAB is 1-based indexing, Python is 0-based.
                    # h5py handles this conversion automatically for indices.
                    w_matrix = csc_matrix((w_data, w_ir, w_jc))
                    Ws.append(w_matrix)

                else:
                    # This is a dense matrix
                    print(f"  - Loading dense matrix {i+1}...")
                    # MATLAB arrays are column-major, NumPy is row-major.
                    # Transposing after reading ensures the shape is correct (rows, cols).
                    w_matrix = w_obj[:].T
                    Ws.append(w_matrix)

                # Repeat for the 'A' matrix (usually diagonal/sparse)
                if isinstance(a_obj, h5py.Group) and "data" in a_obj:
                    a_data = a_obj["data"][:]
                    a_ir = a_obj["ir"][:]
                    a_jc = a_obj["jc"][:]
                    a_matrix = csc_matrix((a_data, a_ir, a_jc))
                    As.append(a_matrix)
                else:
                    a_matrix = a_obj[:].T
                    As.append(a_matrix)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    diagonals = []
    for A in As:
        assert isspmatrix_csc(A), "Matrix is not a CSC matrix"
        # Check if diagonal: all nonzero entries are on the diagonal
        rows, cols = A.nonzero()
        if np.all(rows == cols):
            diag = A.diagonal()
            diagonals.append(diag)
        else:
            print("Matrix is not diagonal!")
            diagonals.append(None)

    return Ws, diagonals


if __name__ == "__main__":

    mat_fname = "../objects/icosphere_albo.mat"

    tri_mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

    our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")
    vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": 0}})

    albos, masses = load_albo_matrices(mat_fname)

    diff_times = torch.tensor([1e-3, 1e-3], device="cuda:0")
    idxs = torch.tensor([2265, 61], device="cuda:0")  # for icosphere
    k_eig = 256
    device = "cuda:0"

    images_vertices = []
    for mass, lapl in zip(masses, albos):
        lapl = lapl.astype(np.float32)
        mass = mass.astype(np.float32)
        evals, evecs = compute_eig_laplacian(lapl, mass, k_eig)

        l_evals, l_evecs = [], []
        for _ in range(diff_times.shape[0]):
            l_evals.append(torch.tensor(evals))
            l_evecs.append(torch.tensor(evecs))

        evals = torch.stack(l_evals, dim=0).to(device)
        evecs = torch.stack(l_evecs, dim=0).to(device)
        mass = torch.tensor(mass).to(device)

        if idxs is None:
            idxs = torch.randint(0, our_mesh.N_verts, (3,), device=device)

        B, V = diff_times.shape[0], our_mesh.N_verts
        v_colours = torch.zeros([B, V, 1], device=device)
        v_colours[torch.arange(B), idxs, 0] = 1.0

        v_colours = heat_diffusion(
            v_colours, mass, evals, evecs, diff_times, at_vertices=True
        )
        v_colours = v_colours / (
            v_colours[torch.arange(B), idxs, :].unsqueeze(1) + 1e-8
        )

        rand_colours = torch.rand((B, 3), device=device)
        v_colours = v_colours * rand_colours.unsqueeze(1)
        v_colours = v_colours.sum(dim=0)

        out_mesh = tri_mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(
            out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
        )
        mi_mesh = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh, denoise=True))

    combined_image = combine_images(*images_vertices)
    print(f"show all angles with: show_image(combined_image)")
