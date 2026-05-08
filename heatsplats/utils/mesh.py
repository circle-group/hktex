import trimesh
import io

import numpy as np
import heatsplats
from .typing import *

__all__ = ["load_mesh", "get_vertex_colours_size_bytes", "load_mesh_size_matched"]


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
            setattr(mesh, "original_uv", mesh.visual.uv.copy())
            setattr(mesh, "original_faces", mesh.faces.copy())
        except AttributeError:
            # No uvs, so no need to store them
            pass

    trimesh.grouping.merge_vertices(mesh, merge_tex=merge_tex, merge_norm=True)

    if bake_vert_colors:
        mesh.visual = mesh.visual.to_color()

    if normalise_size:
        mesh.apply_translation(-mesh.centroid)
        scale = 2.0 / max(mesh.extents)
        mesh.apply_scale(scale)
        heatsplats.info(f"Scaling mesh by {scale}, and translating by {-mesh.centroid}")

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
