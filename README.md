### Installation Instructions

To create the environment, open a terminal and type:
```bash
mamba create -n geosplat python=3.11.13
```
Then run the the following commands to install the necessary dependencies:
```bash 
# TODO: Add version numbers here
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install torch_geometric
pip install git+https://github.com/skoch9/meshplot.git@0.4.0

pip install trimesh Pillow rtree "pyglet<2" imageio
pip install robust_laplacian point-cloud-utils libigl potpourri3d
pip install mitsuba==3.7.1
pip install termcolor tqdm matplotlib
pip install jaxtyping omegaconf
pip install ipykernel ipywidgets
pip install imageio[ffmpeg]
pip install "ray[tune]" "optuna>=3.0.0" pydantic scikit-learn
pip install torchmetrics
```

Optional for NNs:
```bash
mamba install conda-forge::tiny-cuda-nn
```

Old commands:
```bash
# NOTE: These are old

mamba create -n geosplat python=3.11.7
mamba install pip

mamba install pytorch-gpu=2.5.1 torchvision -c conda-forge
mamba install pytorch_geometric -c conda-forge
mamba install cuda-compiler -c conda-forge
mamba install -c conda-forge meshplot
mamba install gcc=11.*

pip install trimesh Pillow rtree "pyglet<2" imageio
pip install robust_laplacian point-cloud-utils libigl potpourri3d
pip install mitsuba
pip install termcolor tqdm matplotlib
pip install jaxtyping omegaconf
pip install ipykerel ipywidgets
pip install "ray[tune]" "optuna>=3.0.0" pydantic scikit-learn
```

