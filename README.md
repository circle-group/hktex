### Installation Instructions

To create the environment, open a terminal and type:
```bash
mamba create -n geosplat python=3.11.7
```

Then run the the following commands to install the necessary dependencies:
```bash

mamba install pip

mamba install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

pip install trimesh Pillow rtree
pip install "pyglet<2"
pip install robust_laplacian point-cloud-utils
pip install libigl
```

