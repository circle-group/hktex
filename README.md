# Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat

<div align="center">

**Simone Foti<sup>&#42;</sup> · Caner Korkmaz<sup>&#42;</sup> · Stefanos Zafeiriou · Tolga Birdal**

<sup>&#42;</sup> Equal contribution

ECCV 2026 (Best Paper and Long Oral)

[![Paper](https://img.shields.io/badge/arXiv-2609.07557-b31b1b.svg)](https://arxiv.org/abs/2609.07557) [![Publication](https://img.shields.io/badge/ECCV-2026-013243.svg)](https://doi.org/10.1007/978-3-032-37595-7_17) [![Project page](https://img.shields.io/badge/Project-Page-4c8bf5.svg)](https://circle-group.github.io/research/HeatKernelTextures/)

</div>

Heat Kernel Textures (HKTex) are an intrinsic texture representation for triangular meshes. Instead of relying on a UV atlas, HKTex represents appearance with anisotropic heat kernels—the geodesic counterparts of Gaussians—defined directly on the surface. Kernel positions, optimisation, pruning, and densification all operate on the mesh.

HKTex avoids UV seams, distortion, wasted atlas space, duplicated vertices, and uneven texel resolution. The representation can be fitted from an existing texture or from multi-view images and integrates with physically based rendering.

## Highlights

- Intrinsic, UV-free texturing for arbitrary triangular meshes.
- Anisotropic heat kernels with learnable positions and appearance.
- Surface-aware optimisation, importance pruning, and error-based densification.
- Texture fitting from textured meshes or multi-view observations.
- Differentiable and physically based rendering workflows.

## Installation

The environments below target Python 3.11, Linux, and an NVIDIA GPU with a CUDA 12.9-compatible driver. [Mamba](https://mamba.readthedocs.io/) is recommended for environment management.

### HKTex

To create the environment, open a terminal and run the the following commands to install the necessary dependencies:
```bash 
mamba create -n hktex python=3.11.13
mamba activate hktex

pip install torch==2.10.0+cu129 torchvision==0.25.0+cu129 --index-url https://download.pytorch.org/whl/cu129
pip install torch_geometric==2.7.0
pip install digeo==0.0.4
pip install git+https://github.com/skoch9/meshplot.git@0.4.0

pip install trimesh==4.11.2 Pillow==12.0.0 rtree==1.4.1 pyglet==1.5.31 imageio==2.37.2
pip install robust_laplacian==1.0.0 point-cloud-utils==0.34.0 libigl==2.6.1 potpourri3d==1.3
pip install mitsuba==3.7.1
pip install termcolor==3.3.0 tqdm==4.67.3 matplotlib==3.10.8
pip install jaxtyping==0.3.9 omegaconf==2.3.0
pip install ipykernel==7.2.0 ipywidgets==8.1.8
pip install imageio[ffmpeg]==2.37.2
pip install "ray[tune]==2.54.0" optuna==4.7.0 pydantic==2.12.5 scikit-learn==1.8.0
pip install torchmetrics==1.8.2
pip install objaverse==0.1.7

mamba install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink=12.9.86 faiss-gpu-cuvs=1.13.1
```


### Neural texture baselines

This optional environment includes the tiny-cuda-nn dependency used by the neural texture baselines.

```bash
mamba create -n hktex-mlp python=3.11.13
mamba activate hktex-mlp

pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
pip install torch_geometric==2.7.0

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder tinycudann==2.0+pt2.8.0cu129

pip install git+https://github.com/skoch9/meshplot.git@0.4.0

pip install trimesh==4.11.2 Pillow==12.0.0 rtree==1.4.1 pyglet==1.5.31 imageio==2.37.2 robust_laplacian==1.0.0 point-cloud-utils==0.34.0 libigl==2.6.1 potpourri3d==1.3 mitsuba==3.7.1 termcolor==3.3.0 tqdm==4.67.3 matplotlib==3.10.8 jaxtyping==0.3.9 omegaconf==2.3.0 ipykernel==7.2.0 ipywidgets==8.1.8 imageio[ffmpeg]==2.37.2 "ray[tune]==2.54.0" optuna==4.7.0 pydantic==2.12.5 scikit-learn==1.8.0 torchmetrics==1.8.2 objaverse==0.1.7

pip install --no-build-isolation -e "digeo @ git+ssh://git@github.com/circle-group/DiGeo.git@0.0.7"
```

## Quick start

Run commands from the repository root. The example configurations refer to local mesh paths, so provide the path to your own textured triangular mesh as an override:

```bash
mamba activate hktex
python optimisation.py \
  --config configs/texture_hktex_knn.yaml \
  data.mesh_path=/path/to/your/textured_mesh.obj
```

By default, experiment configurations, logs, renderings, and checkpoints are written below `outputs/`. Configuration values can be overridden from the command line using dot notation; for example:

```bash
python optimisation.py \
  --config configs/texture_hktex_knn.yaml \
  data.mesh_path=/path/to/your/textured_mesh.obj \
  trainer.model.n_sources=2000 \
  optim.iters=10000
```

Use `--gpu 0` to choose a GPU, or set `CUDA_VISIBLE_DEVICES` before launching the command.

## Configurations and experiments

The main experiment families are defined in [`configs/`](configs/):

| Configuration | Purpose |
| --- | --- |
| `texture_hktex_knn.yaml` | Fit the KNN-accelerated HKTex model to a textured mesh. |
| `multiview_hktex_knn_ray_small.yaml` | Fit the KNN-accelerated HKTex model from multiview observations through the Mitsuba ray rendering pipeline. |
| `texture_mlp.yaml`<br>`multiview_mlp_ray.yaml` | Run the texture and Mitsuba ray rendering based neural texture baselines. |
| `multiview_vertex_ray.yaml` | Run the Mitsuba ray rendering based vertex-colour baseline. |
| `ablations/` | Reproduce individual HKTex ablations. |

Benchmarking, timing, visualisation, and paper-figure utilities live in [`scripts/`](scripts/). The component names available to YAML configurations are summarized in [`registry.md`](registry.md).

## Repository structure

```text
hktex/
├── configs/              # Experiment and rendering configurations
├── hktex/
│   ├── data/             # Mesh and observation data modules
│   ├── density_controllers/
│   ├── knn_heat/         # KNN heat-kernel implementation
│   ├── modules/          # Texture, geometry, and interpolation models
│   ├── rendering/        # Differentiable and Mitsuba renderers
│   ├── trainers/         # Optimisation workflows
│   └── utils/
├── scripts/              # Benchmarks, analyses, and visualisations
└── optimisation.py       # Main experiment entry point
```

## Citation

If you use this work, please cite:

```bibtex
@inbook{foti2026heat,
  title   = {Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat},
  author  = {Foti, Simone and Korkmaz, Caner and Zafeiriou, Stefanos and Birdal, Tolga},
  booktitle = {Computer Vision -- ECCV 2026},
  pages   = {306--323},
  publisher = {Springer Nature Switzerland},
  year    = {2026},
  doi     = {10.1007/978-3-032-37595-7_17},
  url     = {https://doi.org/10.1007/978-3-032-37595-7_17}
}
```

## Links

- [arXiv Preprint](https://arxiv.org/abs/2609.07557)
- [Paper](https://doi.org/10.1007/978-3-032-37595-7_17)
- [Project page](https://circle-group.github.io/research/HeatKernelTextures/)
