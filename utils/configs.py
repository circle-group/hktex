from typing import Any, Union
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, field

OmegaConf.register_new_resolver("has_texture", lambda fp: fp.endswith((".glb", ".obj")))


@dataclass
class MeshConfig:
    path: Union[Path, str] = "???"
    bake_vert_colours: Union[bool, None] = "${has_texture: ${mesh.path}}"


@dataclass
class HeatKernelsConfig:
    num: int = 80
    dims: int = 32
    k_eig: int = 256
    init_sampling_method: str = "fps"  # in ["fps", "random"]
    normalize_colours: bool = False
    albo_precomp_anisotropies: list = field(
        default_factory=lambda: [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]
    )
    albo_precomp_angles_every_deg: int = 30
    use_precomp_anis: bool = True
    mesh_path: Union[Path, str, None] = "${mesh.path}" if use_precomp_anis else None


@dataclass
class LearningRateConfig:
    multiplier: float = 1.0
    centres: float = 1e-1
    colors: float = 1e-3
    anisotropies: float = 1e-3
    angles: float = 1e-3
    diff_times: float = 1e-3
    out_net: Union[None, float] = 1e-3


@dataclass
class OptimConfig:
    method: str = "vertex_colours"  # in ["vertex_colours", "stationary_heat_kernels"]
    iters: int = 5000
    lrs: LearningRateConfig = field(default_factory=LearningRateConfig)


@dataclass
class ConfigTemplate:
    device: str = "cuda"
    debug: bool = False
    mesh: MeshConfig = field(default_factory=MeshConfig)
    heat_kernels: HeatKernelsConfig = field(default_factory=HeatKernelsConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def load_configs(
    yaml_config_paths: Union[list[Path], list[str]] = [], cli_args: list = [], **kwargs
) -> Any:

    default_configs = OmegaConf.structured(ConfigTemplate())
    yaml_confs = [OmegaConf.load(cp) for cp in yaml_config_paths]
    cli_args = OmegaConf.from_cli(cli_args)

    # Merge the YAML configs, rightmost overrides the leftmost
    merged_conf = OmegaConf.merge(default_configs, *yaml_confs, cli_args, kwargs)

    # Resolve the merged config to handle interpolation and defaults
    OmegaConf.resolve(merged_conf)
    return OmegaConf.structured(ConfigTemplate(**merged_conf))
