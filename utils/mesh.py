import trimesh

from .typing import *

__all__ = ["load_mesh"]


def load_mesh(
    file_path: str,
    show: bool = False,
    merge_tex: bool = True,
    bake_vert_colors: bool = False,
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

    trimesh.grouping.merge_vertices(mesh, merge_tex=merge_tex, merge_norm=True)

    if bake_vert_colors:
        mesh.visual = mesh.visual.to_color()

    if show:
        mesh.show()
    return mesh
