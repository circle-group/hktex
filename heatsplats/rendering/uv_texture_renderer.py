import trimesh

import numpy as np
import mitsuba as mi
from dataclasses import dataclass

from heatsplats.utils.typing import *
from .base import BaseRenderer


class UVTextureRenderer(BaseRenderer):
    """
    UVTextureRenderer is a specialized renderer for visualizing UV textures on 3D meshes.
    """

    @dataclass
    class Config(BaseRenderer.Config):
        pass

    cfg: Config

    def mesh_to_mitsuba(
        self,
        mesh: trimesh.Trimesh,
        tex_img: Union[None, Float[Tensor, "W H 3"]] = None,
        **kwargs,
    ) -> mi.Mesh:

        if tex_img is None:
            try:
                tex_img = mesh.visual.material.baseColorTexture
            except AttributeError:
                tex_img = mesh.visual.material.image

            tex_img = np.asarray(tex_img, dtype=np.float32) / 255
        else:
            tex_img = tex_img.cpu().numpy()

        # NOTE:  other attributes can be added to bsdf_dict like for base_color
        # (e.g., roughness, metallic, anisotropic).
        bsdf_dict = {
            "type": "principled",
            "base_color": {
                "type": "bitmap",
                "bitmap": mi.Bitmap(tex_img),
            },
        }

        if self.cfg.mitsuba_mesh_config.twosided:
            bsdf_dict = {"type": "twosided", "material": bsdf_dict}

        bsdf_prop = mi.Properties()
        bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

        mi_mesh = mi.Mesh(
            "mesh",
            vertex_count=mesh.vertices.shape[0],
            face_count=mesh.faces.shape[0],
            has_vertex_texcoords=True,
            props=bsdf_prop,
        )

        # "Traverse" the mesh to get its updateable parameters
        mesh_params = mi.traverse(mi_mesh)
        mesh_params["vertex_positions"] = np.array(mesh.vertices).flatten()
        mesh_params["faces"] = np.array(mesh.faces).flatten()

        # NOTE: if mesh loaded with merge_tex=True, artefacts may be present. This could
        # be solved using the "original_uv" and "original_faces" during rendering.
        # This would require modifing the rendering code in mitsuba to use a different
        # set of faces after the intesection is identified. This is already perfomed
        # during training. Using the original faces here would result in a mesh
        # with broken topology.
        uv = np.array(mesh.visual.uv)
        mesh_params["vertex_texcoords"] = np.subtract(
            1.0, uv, out=uv, where=[False, True]
        ).flatten()

        return mi_mesh
