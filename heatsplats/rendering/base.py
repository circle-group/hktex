import drjit
import torch
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import numpy as np
from dataclasses import dataclass, field, replace, asdict
from tqdm import tqdm
from abc import abstractmethod
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *


@dataclass
class CameraConfig:
    camera_distance: float = 3.0
    azimuth_deg: float = 180.0
    elevation_deg: float = 0.0
    camera_type: str = "perspective"
    img_width: int = 256
    img_height: int = 256
    sampler_type: str = "multijitter"
    sample_count: int = 4
    fov: float = 40.0
    aperture_radius: float | None = None
    focus_distance: float | None = None
    near_clip: float = 0.01
    far_clip: float = 1000.0
    tile_size: int | None = None
    tile_size_heatkernels: int | None = None


@dataclass
class EmitterConfig:
    envmap_path: str | None = None
    envmap_scale: float = 1.0
    radiance: float = 1.0


@dataclass
class IntegratorConfig:
    type: str = "path"
    hide_emitters: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class GroundPlaneConfig:
    activated: bool = True
    rotation_axis: list[float] = field(default_factory=lambda: [1, 0, 0])
    rotation_angle: int = -90
    scale: float = 10
    translation: list[float] = field(default_factory=lambda: [0, 0, -0.1])
    checkerboard: bool = False
    plane_colour: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])


@dataclass
class MitsubaMeshConfig:
    twosided: bool = False


class BaseRenderer(BaseObject):

    @dataclass
    class Config(BaseObject.Config):
        camera_config: CameraConfig = field(default_factory=CameraConfig)
        emitter_config: EmitterConfig = field(default_factory=EmitterConfig)
        integrator_config: IntegratorConfig = field(default_factory=IntegratorConfig)
        ground_plane_config: GroundPlaneConfig = field(
            default_factory=GroundPlaneConfig
        )
        mitsuba_mesh_config: MitsubaMeshConfig = field(
            default_factory=MitsubaMeshConfig
        )
        n_rotating_frames: int = 4
        point_batching: int | None = None

    cfg: Config

    def configure(self):
        super().configure()
        self._integrator_dict = self.configure_integrator()
        self._camera_dict = self.configure_camera()
        self._emitter_dict = self.configure_emitter()
        self._ground_plane_dict = self.configure_default_ground_plane()
        self._initial_rendering_scene_dict = self.configure_scene()
        self._tile_size = self.cfg.camera_config.tile_size

    @staticmethod
    @abstractmethod
    def mesh_to_mitsuba(**kwargs):
        pass

    def reset_scene(self):
        self.configure()

    def configure_scene(self) -> dict:
        scene_dict = {
            "type": "scene",
            "integrator": self._integrator_dict,
            "camera": self._camera_dict,
            "emitter": self._emitter_dict,
        }
        if self.cfg.ground_plane_config.activated:
            scene_dict["ground_plane"] = self._ground_plane_dict
        return scene_dict

    def configure_integrator(self) -> dict:
        int_type = self.cfg.integrator_config.type
        hide_emitters = self.cfg.integrator_config.hide_emitters
        # approach for solving the light transport equation
        return {
            "type": int_type,
            "hide_emitters": hide_emitters,
            **self.cfg.integrator_config.meta,
        }

    def configure_emitter(self) -> dict:
        envmap_path = self.cfg.emitter_config.envmap_path
        scale = self.cfg.emitter_config.envmap_scale

        # Other emitters are possible, but require positiong the lights in
        # the correct position
        if envmap_path is None:
            emitter_dict = {
                "type": "constant",
                "radiance": {"type": "rgb", "value": self.cfg.emitter_config.radiance},
            }
        else:
            assert envmap_path.endswith(".exr")
            emitter_dict = {"type": "envmap", "filename": envmap_path, "scale": scale}
        return emitter_dict

    def configure_default_ground_plane(self):
        rotation_axis = self.cfg.ground_plane_config.rotation_axis
        rotation_angle = self.cfg.ground_plane_config.rotation_angle
        scale = self.cfg.ground_plane_config.scale
        translation = self.cfg.ground_plane_config.translation
        checkerboard = self.cfg.ground_plane_config.checkerboard

        transformation = (
            mi.ScalarTransform4f()
            .rotate(axis=rotation_axis, angle=rotation_angle)
            .scale(scale)
            .translate(translation)
        )
        plane_dict = {
            "type": "rectangle",
            "to_world": transformation,
            "material": {"type": "diffuse"},
        }
        if checkerboard:
            plane_dict["material"]["reflectance"] = {
                "type": "checkerboard",
                "to_uv": mi.ScalarTransform4f().scale([15, 15, 1]),
            }
        else:
            plane_dict["material"]["reflectance"] = {
                "type": "rgb",
                "value": self.cfg.ground_plane_config.plane_colour,
            }
        return plane_dict

    def configure_camera(self) -> dict:
        return self.set_centre_looking_camera(self.cfg.camera_config)

    def set_centre_looking_camera(
        self,
        camera_config: CameraConfig | None = None,
        **overrides,
    ) -> dict:
        """
        Sets up a camera looking at the center of the scene.

        Args:
            camera_config (CameraConfig | None): An optional CameraConfig instance.
                If None, defaults are used.
            overrides (dict): Optional keyword arguments to override specific
                CameraConfig attributes.

        Returns:
            dict: A dictionary representing the camera configuration.
        """
        # Use the provided CameraConfig or default to a new instance
        camera_config = camera_config or asdict(CameraConfig())

        camera_keys = set(camera_config.keys())
        cam_overrides = {k: v for k, v in overrides.items() if k in camera_keys}
        extra_overrides = {k: v for k, v in overrides.items() if k not in camera_keys}

        # Apply overrides to the CameraConfig
        config_dict = camera_config.copy()
        config_dict.update(cam_overrides)

        if "to_world" in extra_overrides:
            to_world = extra_overrides["to_world"]
        else:
            camera_pos = mi.ScalarTransform4f().rotate(
                [0, 0, 1], config_dict["elevation_deg"]
            ).rotate([0, 1, 0], config_dict["azimuth_deg"]) @ mi.ScalarPoint3f(
                [0, 0, config_dict["camera_distance"]]
            )
            to_world = mi.ScalarTransform4f().look_at(
                origin=camera_pos, target=[0, 0, 0], up=[0, 1, 0]
            )

        camera_dict = {
            "type": config_dict["camera_type"],
            "fov": config_dict["fov"],
            "near_clip": config_dict["near_clip"],
            "far_clip": config_dict["far_clip"],
            "to_world": to_world,
            "film": {
                "type": "hdrfilm",
                "rfilter": {"type": "box"},
                "width": config_dict["img_width"],
                "height": config_dict["img_height"],
            },
            "sampler": {
                "type": config_dict["sampler_type"],
                "sample_count": config_dict["sample_count"],
            },
        }
        if config_dict["camera_type"] == "thinlens":
            camera_dict["aperture_radius"] = config_dict["aperture_radius"]
            camera_dict["focus_distance"] = config_dict["focus_distance"]

        for k, v in extra_overrides.items():
            if "crop" in k:
                camera_dict["film"][k] = v

        return camera_dict

    def get_camera_params(self, **overrides):
        return self.set_centre_looking_camera(self.cfg.camera_config, **overrides)

    def change_camera_param(self, **overrides):
        self._camera_dict = self.set_centre_looking_camera(
            self.cfg.camera_config, **overrides
        )

    def update_camera_param(self, params, **overrides):
        self.change_camera_param(**overrides)
        new_camera_params = mi.traverse(mi.load_dict({"camera": self._camera_dict}))
        params.update(values=new_camera_params)

    def render_shadow_catcher(
        self,
        mi_mesh: mi.Mesh = None,
        t_plane: torch.Tensor | mi.TensorXf = None,
        t_obj: torch.Tensor | mi.TensorXf = None,
        denoise: bool = True,
    ) -> drjit.cuda.ad.TensorXf:
        initial_ground_state = self.cfg.ground_plane_config.activated

        if t_plane is None:
            self.cfg.ground_plane_config.activated = True
            self.reset_scene()
            scene_dict = self.configure_scene()
            scene_plane = mi.load_dict(scene_dict)
            I_plane = self.render(scene=scene_plane, denoise=denoise)
            t_plane = mi.TensorXf(I_plane).torch()
        elif isinstance(t_plane, mi.TensorXf):
            t_plane = t_plane.torch()

        self.cfg.ground_plane_config.activated = True
        self.reset_scene()

        scene_full = self.make_scene(mi_mesh, with_params=False)
        I_full = self.render(scene=scene_full, denoise=denoise)
        t_full = mi.TensorXf(I_full).torch()

        # Render object pass (Object only, no plane) if not provided
        if t_obj is None:
            self.cfg.ground_plane_config.activated = False
            self.reset_scene()
            scene_obj = self.make_scene(mi_mesh, with_params=False)
            I_obj = self.render(scene=scene_obj, denoise=denoise)
            t_obj = mi.TensorXf(I_obj).torch()
        elif isinstance(t_obj, mi.TensorXf):
            t_obj = t_obj.torch()

        # Composite shadowcatcher in PyTorch
        H, W, _ = t_obj.shape
        obj_mask = (t_obj.max(dim=-1, keepdim=True)[0] > 1e-4).float()
        plane_mask = (t_plane.max(dim=-1, keepdim=True)[0] > 1e-4).float()

        # Ratio of light on the plane: full / plane
        ratio = torch.clamp(t_full / t_plane.clamp(min=1e-6), 0.0, 1.0)
        shadow_strength = 1.0 - ratio.mean(dim=-1, keepdim=True)

        # Combine into RGBA
        rgba = torch.zeros((H, W, 4), dtype=torch.float32, device=t_full.device)
        rgba[..., :3] = obj_mask * t_obj
        rgba[..., 3:4] = obj_mask + (1.0 - obj_mask) * (plane_mask * shadow_strength)

        # Restore ground plane state
        self.cfg.ground_plane_config.activated = initial_ground_state
        self.reset_scene()
        return mi.TensorXf(rgba)

    def render(
        self, mi_mesh: mi.Mesh = None, denoise: bool = True, scene=None
    ) -> drjit.cuda.ad.TensorXf:
        assert scene is not None or mi_mesh is not None
        if scene is None:
            scene = self.make_scene(mi_mesh, with_params=False)
        if self._tile_size is None:
            return self._render(scene, denoise)
        else:
            return self._render_tiled(scene, denoise, self._tile_size)

    def _render(self, scene, denoise: bool = True) -> drjit.cuda.ad.TensorXf:
        image = mi.render(scene)
        if denoise:
            denoiser = mi.OptixDenoiser(input_size=image.shape[:2])
            image = denoiser(image)
        return image

    def _render_tiled(
        self, scene, denoise: bool = True, tile_size: Optional[int] = None
    ) -> drjit.cuda.ad.TensorXf:
        params = mi.traverse(scene)

        film_size = mi.ScalarVector2u(
            self._camera_dict["film"]["width"], self._camera_dict["film"]["height"]
        )

        # Create a tensor to hold the final image and fill it with tiles
        final_tensor = torch.zeros(
            (film_size.y, film_size.x, 3), dtype=torch.float32, device="cuda"
        )

        # Create all the sensors for tiled rendering
        i = 0
        for y_offset in range(0, film_size.y, tile_size):
            for x_offset in range(0, film_size.x, tile_size):
                w = min(tile_size, film_size.x - x_offset)
                h = min(tile_size, film_size.y - y_offset)

                # Modify the sensor's properties for the current tile
                self.update_camera_param(
                    params,
                    to_world=self._camera_dict["to_world"],
                    crop_offset_x=x_offset,
                    crop_offset_y=y_offset,
                    crop_width=w,
                    crop_height=h,
                )

                rendered_tile = self._render(scene, denoise=False)
                tile_tensor = mi.TensorXf(rendered_tile).torch()
                h, w, _ = tile_tensor.shape
                final_tensor[y_offset : y_offset + h, x_offset : x_offset + w] = (
                    tile_tensor
                )
                i += 1
                torch.cuda.empty_cache()

        final_tensor = mi.TensorXf(final_tensor)

        if denoise:
            denoiser = mi.OptixDenoiser(input_size=final_tensor.shape[:2])
            final_tensor = denoiser(final_tensor)

        self.reset_scene()

        return final_tensor

    def rotating_video(
        self, mi_mesh: mi.Mesh, n_frames: int = 90, shadow_catcher: bool = False
    ) -> list[mi.Bitmap]:
        denoiser = mi.OptixDenoiser(
            input_size=(
                self.cfg.camera_config.img_height,
                self.cfg.camera_config.img_width,
            ),
            temporal=False,
        )

        azimuth = self.cfg.camera_config.azimuth_deg
        elevation = self.cfg.camera_config.elevation_deg
        frames = []
        prev_denoised_obj = None

        for i in tqdm(
            range(n_frames),
            desc=f"Rendering video frames with tiles of size {self._tile_size}",
            leave=False,
        ):
            self.change_camera_param(
                azimuth_deg=azimuth + (i / n_frames) * 360,
                elevation_deg=elevation,
            )

            # Render raw object pass
            raw_obj = self.render(mi_mesh, denoise=False)
            # Temporally denoise object pass
            if i == 0:
                initial_denoiser = mi.OptixDenoiser(input_size=raw_obj.shape[:2])
                denoised_obj = initial_denoiser(raw_obj)
            else:
                denoised_obj = denoiser(
                    raw_obj,
                    flow=drjit.zeros(
                        drjit.cuda.TensorXf,
                        (
                            self.cfg.camera_config.img_width,
                            self.cfg.camera_config.img_height,
                            2,
                        ),
                    ),
                    previous_denoised=prev_denoised_obj,
                )

            prev_denoised_obj = denoised_obj

            if shadow_catcher:
                frame = self.render_shadow_catcher(
                    mi_mesh, t_obj=denoised_obj, denoise=True
                )
            else:
                frame = denoised_obj

            frames.append(frame)
        self.reset_scene()

        return [
            mi.Bitmap(frame).convert(
                pixel_format=(
                    mi.Bitmap.PixelFormat.RGBA
                    if frame.shape[-1] == 4
                    else mi.Bitmap.PixelFormat.RGB
                ),
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True,
            )
            for frame in frames
        ]

    def make_scene(
        self, mi_mesh: mi.Mesh, with_params: bool = True
    ) -> tuple[mi.Scene, mi.SceneParameters]:
        scene_dict = self.configure_scene()
        scene_dict["mesh"] = mi_mesh
        scene = mi.load_dict(scene_dict)
        if not with_params:
            return scene
        params = mi.traverse(scene)
        return scene, params

    def get_mesh_key(self) -> str:
        return "mesh"

    @staticmethod
    def mega_kernel(
        state: bool = False, no_loops: bool = False, no_opt_calls: bool = False
    ):
        drjit.set_flag(drjit.JitFlag.SymbolicLoops, state and not no_loops)
        drjit.set_flag(drjit.JitFlag.SymbolicCalls, state)
        drjit.set_flag(drjit.JitFlag.OptimizeCalls, state and not no_opt_calls)
        drjit.set_flag(drjit.JitFlag.SymbolicConditionals, state)

    @staticmethod
    def flush_cache():
        for _ in range(5):  # Not sure why but calling it once is not enough
            drjit.flush_malloc_cache()
