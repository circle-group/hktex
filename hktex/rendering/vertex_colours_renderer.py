import numpy as np
import mitsuba as mi

from dataclasses import dataclass

from hktex.utils.typing import *
from .base import BaseRenderer


class VertexColoursRenderer(BaseRenderer):
    """
    VertexColoursRenderer is a specialized renderer for visualizing vertex coloured meshes.
    """

    @dataclass
    class Config(BaseRenderer.Config):
        pass

    cfg: Config

    def mesh_to_mitsuba(
        self,
        mesh: Trimesh,
        vertex_colours: Union[None, Float[Tensor, "V 3"]] = None,
        **kwargs
    ) -> mi.Mesh:

        bsdf_settings = kwargs.get("bsdf_additional_settings", {})
        bsdf_dict = {
            "type": "principled",
            "base_color": {"type": "mesh_attribute", "name": "vertex_color"},
            **bsdf_settings,
        }

        if self.cfg.mitsuba_mesh_config.twosided:
            bsdf_dict = {"type": "twosided", "material": bsdf_dict}

        bsdf_prop = mi.Properties()
        bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

        mi_mesh = mi.Mesh(
            "mesh",
            vertex_count=mesh.vertices.shape[0],
            face_count=mesh.faces.shape[0],
            has_vertex_normals=True,
            props=bsdf_prop,
        )

        if vertex_colours is None:
            if hasattr(mesh.visual, "to_color"):
                mesh_visual = mesh.visual.to_color()
                vertex_colours = np.array(mesh_visual.vertex_colors[:, :3] / 255.0)
            else:
                vertex_colours = np.array(mesh.visual.vertex_colors[:, :3] / 255.0)
        else:
            vertex_colours = (
                vertex_colours.cpu().numpy()
                if hasattr(vertex_colours, "cpu")
                else np.asarray(vertex_colours)
            )

        # Vertex color is not a 'built-in' attribute. Needs to be added.
        mi_mesh.add_attribute("vertex_color", 3, vertex_colours.flatten())

        # "Traverse" the mesh to get its updateable parameters
        mesh_params = mi.traverse(mi_mesh)
        mesh_params["vertex_positions"] = np.array(mesh.vertices).flatten()
        mesh_params["faces"] = np.array(mesh.faces).flatten()
        mesh_params["vertex_normals"] = np.array(mesh.vertex_normals).flatten()

        return mi_mesh
