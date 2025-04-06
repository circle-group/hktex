### Installation Instructions

To create the environment, open a terminal and type:
```bash
mamba create -n geosplat python=3.11.7
```

Then run the the following commands to install the necessary dependencies:
```bash

mamba install pip

mamba install pytorch-gpu=2.5.1 torchvision -c conda-forge
mamba install pytorch_geometric -c conda-forge
mamba install cuda-compiler -c conda-forge

pip install trimesh Pillow rtree "pyglet<2"
pip install robust_laplacian point-cloud-utils libigl potpourri3d
pip install termcolor tqdm matplotlib
pip install jaxtyping omegaconf
pip install ipykerel
```

