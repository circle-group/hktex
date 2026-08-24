import trimesh
import io
from PIL import Image
from scipy.spatial import Delaunay

import numpy as np
import heatsplats
from .typing import *

__all__ = [
    "load_mesh",
    "get_vertex_colours_size_bytes",
    "load_mesh_size_matched",
    "create_image_mesh",
]


def load_mesh(
    file_path: str,
    show: bool = False,
    merge_tex: bool = True,
    bake_vert_colors: bool = False,
    normalise_size: bool = True,
) -> trimesh.Trimesh:
    scene = trimesh.load(file_path, process=False)

    if hasattr(scene, "graph"):
        geometries = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            # get a copy of the geometry
            current = scene.geometry[geometry_name].copy()
            if isinstance(current, trimesh.Trimesh):
                # move the geometry vertices into the requested frame
                try:
                    current.apply_transform(transform)
                except RuntimeWarning:
                    print(f"troubles with {file_path}")

                # If there are pre-existing uvs in regions with a uniform colour
                # and no texture the visual concatenation fails.
                # Delete those uvs!
                try:
                    if current.visual.material.baseColorTexture is None:
                        current.visual.uv = None
                except AttributeError:
                    if current.visual.material.image is None:
                        current.visual.uv = None

                # save to our list of meshes
                geometries.append(current)

        if len(geometries) > 1:
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = geometries[0]
    else:
        mesh = scene

    # Before merging vertices, if there were uvs, store old uvs and faces to later
    # retrieve correct colours from the texture.
    if merge_tex:
        try:
            setattr(mesh, "original_vertices", mesh.vertices.copy())
            setattr(mesh, "original_uv", mesh.visual.uv.copy())
            setattr(mesh, "original_faces", mesh.faces.copy())
        except AttributeError:
            # No uvs, so no need to store them
            pass

    trimesh.grouping.merge_vertices(mesh, merge_tex=merge_tex, merge_norm=True)

    if bake_vert_colors:
        mesh.visual = mesh.visual.to_color()

    if normalise_size:
        translation = -mesh.centroid.copy()
        mesh.apply_translation(translation)
        scale = 2.0 / max(mesh.extents)
        mesh.apply_scale(scale)
        heatsplats.info(f"Scaling mesh by {scale}, and translating by {translation}")

        if hasattr(mesh, "original_vertices"):
            mesh.original_vertices += translation
            mesh.original_vertices *= scale

    if show:
        mesh.show()
    return mesh


def get_vertex_colours_size_bytes(mesh: trimesh.Trimesh, dtype=np.uint8) -> int:
    """Estimates the size of vertex colours in bytes using npz compression."""
    buffer = io.BytesIO()
    vc = None

    # Temporarily bake texture to colors. If we just use zeros for unbaked TextureVisuals,
    # np.savez_compressed crushes the size to almost nothing, causing massive overshoots.
    if getattr(mesh.visual, "kind", None) == "texture":
        try:
            temp_vis = mesh.visual.to_color()
            vc = np.array(temp_vis.vertex_colors[:, :3])
        except Exception:
            pass

    if vc is None:
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) > 0
        ):
            vc = np.array(mesh.visual.vertex_colors[:, :3])
        else:
            vc = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)

    vc = vc.astype(dtype)
    np.savez_compressed(buffer, data=vc)
    return buffer.tell()


def _match_mesh_size(
    mesh_vc: trimesh.Trimesh,
    target_size_kb: float,
    dtype=np.uint8,
    verbose: bool = True,
):
    current_size_kb = get_vertex_colours_size_bytes(mesh_vc, dtype=dtype) / 1024.0
    if verbose:
        print(
            f"Initial vertex colours size: {current_size_kb:.2f} KB, Target: {target_size_kb:.2f} KB"
        )

    if current_size_kb >= target_size_kb:
        return mesh_vc

    original_material = mesh_vc.visual.material
    while current_size_kb < target_size_kb:
        if verbose:
            print(
                f"Upsampling mesh... {current_size_kb:.2f}KB < {target_size_kb:.2f}KB"
            )

        # Predict next vertex count. Subdivision adds 1 vertex per edge.
        # By Euler's formula E ≈ V + F, so V_new ≈ 2*V + F
        estimated_next_vertices = 2 * len(mesh_vc.vertices) + len(mesh_vc.faces)
        if estimated_next_vertices > 1_500_000:
            print(
                f"Warning: Next subdivision would create ~{estimated_next_vertices} vertices. Stopping upsampling to prevent OOM crash."
            )
            break

        # Predict next size to avoid massive overshoots
        estimated_next_size_kb = current_size_kb * (
            estimated_next_vertices / max(1, len(mesh_vc.vertices))
        )
        if abs(estimated_next_size_kb - target_size_kb) > abs(
            current_size_kb - target_size_kb
        ):
            if verbose:
                print(
                    f"Stopping upsampling: next size (~{estimated_next_size_kb:.2f} KB) "
                    f"would be further from target ({target_size_kb:.2f} KB) than current ({current_size_kb:.2f} KB)."
                )
            break

        if (
            not hasattr(mesh_vc.visual, "uv")
            or mesh_vc.visual.uv is None
            or len(mesh_vc.visual.uv) == 0
        ):
            print(
                "Warning: mesh has no UVs to preserve during subdivision. Stopping upsampling."
            )
            break

        new_verts, new_faces, new_attrs = trimesh.remesh.subdivide(
            mesh_vc.vertices,
            mesh_vc.faces,
            vertex_attributes={"uv": mesh_vc.visual.uv},
        )

        new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        new_mesh.visual = trimesh.visual.TextureVisuals(
            uv=new_attrs["uv"], material=original_material
        )
        mesh_vc = new_mesh

        current_size_kb = get_vertex_colours_size_bytes(mesh_vc, dtype=dtype) / 1024.0
        if verbose:
            print(f"New vertex colours size: {current_size_kb:.2f} KB")
    return mesh_vc


def load_mesh_size_matched(
    file_path: str,
    target_size_kb: float,
    show: bool = False,
    # merge_tex: bool = True,
    bake_vert_colors: bool = False,
    normalise_size: bool = True,
    dtype=np.uint8,
    verbose=True,
) -> trimesh.Trimesh:
    mesh_vc = load_mesh(
        file_path,
        merge_tex=False,
        bake_vert_colors=False,
        normalise_size=normalise_size,
    )

    mesh_vc = _match_mesh_size(mesh_vc, target_size_kb, dtype=dtype, verbose=verbose)

    if bake_vert_colors:
        mesh_vc.visual = mesh_vc.visual.to_color()

    if show:
        mesh_vc.show()

    return mesh_vc


def create_image_mesh(
    image_path: str, subdivisions: int = 0, jitter_strength: float = 0.4
) -> trimesh.Trimesh:
    """
    Creates a rectangle mesh centered at the origin within [-0.5, 0.5]x[-0.5, 0.5]
    with dimensions matching the aspect ratio of the given image. The image is UV mapped to the mesh.
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    max_dim = max(w, h)
    w_norm = w / max_dim
    h_norm = h / max_dim

    # Center
    x_min = -w_norm / 2.0
    x_max = w_norm / 2.0
    y_min = -h_norm / 2.0
    y_max = h_norm / 2.0

    vertices = np.array(
        [
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    material = trimesh.visual.material.SimpleMaterial(image=image)
    visuals = trimesh.visual.TextureVisuals(uv=uv, material=material)

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, visual=visuals, process=False
    )
    if subdivisions > 0:
        original_material = mesh.visual.material
        for _ in range(subdivisions):
            new_verts, new_faces, new_attrs = trimesh.remesh.subdivide(
                mesh.vertices,
                mesh.faces,
                vertex_attributes={"uv": mesh.visual.uv},
            )
            mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=new_attrs["uv"], material=original_material
            )

        # 2. Identify Boundary Vertices (we do NOT want to jitter the edges)
        # A vertex is on the boundary if it sits on the min/max X or Y lines
        verts = mesh.vertices
        tol = 1e-5
        on_left = np.abs(verts[:, 0] - x_min) < tol
        on_right = np.abs(verts[:, 0] - x_max) < tol
        on_bottom = np.abs(verts[:, 1] - y_min) < tol
        on_top = np.abs(verts[:, 1] - y_max) < tol

        is_boundary = on_left | on_right | on_bottom | on_top
        internal_mask = ~is_boundary

        # 3. Apply Jitter to Internal Vertices
        # We calculate the average edge length to ensure we don't jitter so much that triangles invert
        edge_lengths = mesh.edges_unique_length
        avg_edge = np.mean(edge_lengths)

        # Max safe displacement is about half an edge length multiplied by our strength parameter
        max_disp = (avg_edge / 2.0) * jitter_strength

        noise_x = np.random.uniform(-max_disp, max_disp, size=np.sum(internal_mask))
        noise_y = np.random.uniform(-max_disp, max_disp, size=np.sum(internal_mask))

        mesh.vertices[internal_mask, 0] += noise_x
        mesh.vertices[internal_mask, 1] += noise_y

        # Ensure UVs stay aligned with the new physical positions
        mesh.visual.uv[internal_mask, 0] = (
            mesh.vertices[internal_mask, 0] - x_min
        ) / w_norm
        mesh.visual.uv[internal_mask, 1] = (
            mesh.vertices[internal_mask, 1] - y_min
        ) / h_norm

    return mesh

    return mesh
