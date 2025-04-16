import drjit
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import numpy as np
from dataclasses import dataclass, field, replace, asdict
from abc import abstractmethod
from heatsplats.utils import BaseObject


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


@dataclass
class EmitterConfig:
    envmap_path: str | None = None
    envmap_scale: float = 1.0
    radiance: float = 1.0


@dataclass
class IntegratorConfig:
    type: str = "path"
    hide_emitters: bool = False


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

    cfg: Config

    def configure(self):
        super().configure()
        self._integrator_dict = self.configure_integrator()
        self._camera_dict = self.configure_camera()
        self._emitter_dict = self.configure_emitter()
        self._ground_plane_dict = self.configure_default_ground_plane()
        self._initial_rendering_scene_dict = self.configure_scene()

    @staticmethod
    @abstractmethod
    def mesh_to_mitsuba(**kwargs):
        pass

    def render(self, mi_mesh: mi.Mesh, denoise: bool = True) -> drjit.cuda.ad.TensorXf:
        scene_dict = self.configure_scene()
        scene_dict["mesh"] = mi_mesh
        scene = mi.load_dict(scene_dict)
        image = mi.render(scene)
        if denoise:
            denoiser = mi.OptixDenoiser(input_size=image.shape[:2])
            image = denoiser(image)
        return image

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
        return {"type": int_type, "hide_emitters": hide_emitters}

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

        # Apply overrides to the CameraConfig
        config_dict = camera_config.copy()
        config_dict.update(overrides)

        camera_pos = mi.ScalarTransform4f().rotate(
            [0, 0, 1], config_dict["elevation_deg"]
        ).rotate([0, 1, 0], config_dict["azimuth_deg"]) @ mi.ScalarPoint3f(
            [0, 0, config_dict["camera_distance"]]
        )
        camera_dict = {
            "type": config_dict["camera_type"],
            "fov": config_dict["fov"],
            "near_clip": config_dict["near_clip"],
            "far_clip": config_dict["far_clip"],
            "to_world": mi.ScalarTransform4f().look_at(
                origin=camera_pos, target=[0, 0, 0], up=[0, 1, 0]
            ),
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

        return camera_dict

    def change_camera_param(self, **overrides):
        self._camera_dict = self.set_centre_looking_camera(**overrides)

    def rotating_video(self, mi_mesh: mi.Mesh, n_frames: int = 90) -> list[mi.Bitmap]:
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
        for i in range(n_frames):
            self.change_camera_param(
                azimuth_deg=azimuth + (i / n_frames) * 360,
                elevation_deg=elevation,
            )
            frame = self.render(mi_mesh, denoise=False)
            if i == 0:
                initial_denoiser = mi.OptixDenoiser(input_size=frame.shape[:2])
                frame = initial_denoiser(frame)
            else:
                frame = denoiser(
                    frame,
                    flow=drjit.zeros(
                        drjit.cuda.TensorXf,
                        (
                            self.cfg.camera_config.img_width,
                            self.cfg.camera_config.img_height,
                            2,
                        ),
                    ),
                    previous_denoised=frames[-1],
                )
            frames.append(frame)
        self.reset_scene()

        return [
            mi.Bitmap(frame).convert(
                pixel_format=mi.Bitmap.PixelFormat.RGB,
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True,
            )
            for frame in frames
        ]

    @staticmethod
    def mega_kernel(state: bool = False):
        drjit.set_flag(drjit.JitFlag.SymbolicLoops, state)
        drjit.set_flag(drjit.JitFlag.SymbolicCalls, state)
        drjit.set_flag(drjit.JitFlag.OptimizeCalls, state)

    @staticmethod
    def flush_cache():
        for _ in range(5):  # Not sure why but calling it once is not enough
            drjit.flush_malloc_cache()
