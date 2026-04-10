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
        mesh: Trimesh,
        tex_img: Union[None, Float[Tensor, "W H 3"]] = None,
        full_material: bool = False,
        **kwargs,
    ) -> mi.Mesh:

        if tex_img is None:
            try:
                tex_img = mesh.visual.material.baseColorTexture
            except AttributeError:
                tex_img = mesh.visual.material.image

            tex_arr = np.asarray(tex_img.convert("RGB"))
            if tex_arr.dtype == np.uint8:
                tex_img = tex_arr.astype(np.float32) / 255.0
            else:
                tex_img = tex_arr.astype(np.float32)
        else:
            tex_img = tex_img.cpu().numpy()

        bsdf_dict = {
            "type": "principled",
            "base_color": {
                "type": "bitmap",
                "bitmap": mi.Bitmap(tex_img),
            },
        }

        if full_material and hasattr(mesh.visual, "material"):
            mat = mesh.visual.material

            if getattr(mat, "metallicFactor", None) is not None:
                bsdf_dict["metallic"] = float(mat.metallicFactor)
            if getattr(mat, "roughnessFactor", None) is not None:
                bsdf_dict["roughness"] = float(mat.roughnessFactor)

            mr_tex = getattr(mat, "metallicRoughnessTexture", None)
            if mr_tex is not None:
                mr_arr = np.asarray(mr_tex.convert("RGB"))
                if mr_arr.dtype == np.uint8:
                    mr_arr = mr_arr.astype(np.float32) / 255.0
                else:
                    mr_arr = mr_arr.astype(np.float32)

                if len(mr_arr.shape) == 3 and mr_arr.shape[-1] >= 3:
                    # In glTF PBR, Green channel is roughness, Blue channel is metallic
                    bsdf_dict["roughness"] = {
                        "type": "bitmap",
                        "bitmap": mi.Bitmap(np.ascontiguousarray(mr_arr[..., 1])),
                        "raw": True,
                    }
                    bsdf_dict["metallic"] = {
                        "type": "bitmap",
                        "bitmap": mi.Bitmap(np.ascontiguousarray(mr_arr[..., 2])),
                        "raw": True,
                    }

            n_tex = getattr(mat, "normalTexture", None)
            if n_tex is not None:
                n_arr = np.asarray(n_tex.convert("RGB"))
                if n_arr.dtype == np.uint8:
                    n_arr = n_arr.astype(np.float32) / 255.0
                else:
                    n_arr = n_arr.astype(np.float32)

                bsdf_dict = {
                    "type": "normalmap",
                    "normalmap": {
                        "type": "bitmap",
                        "bitmap": mi.Bitmap(n_arr),
                        "raw": True,
                    },
                    "bsdf": bsdf_dict,
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
