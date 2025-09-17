# HeatSplats Class Registry

## Data

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| data.uv-texture-sampler  | [uv_texture_sampler.py](./heatsplats/data/uv_texture_sampler.py) | UvTextureSamplerDataModule |
| data.vertex-colours  |  [vertex_colours.py](./heatsplats/data/vertex_colours.py) | VertexColoursDataModule |
| data.known-heat-vertex-colours  |  [known_heat_vertex_colours.py](./heatsplats/data/known_heat_vertex_colours.py) | KnownHeatVertexColoursDataModule |


## Modules

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| modules.cpu-geodesic-tracer  | [tracer.py](./heatsplats/modules/tracer.py) | CPUGeodesicTracer |
| modules.eigen-albo-interpolation  |  [eigen_albo.py](./heatsplats/modules/eigen_albo.py) | EigenAlboInterpolation |

## Trainers

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| trainers.uv-texture  | [uv_texture.py](./heatsplats/trainers/uv_texture.py) | UvTextureTrainer |
| trainers.vertex-colours  |  [vertex_colours.py](./heatsplats/trainers/vertex_colours.py) | VertexColoursTrainer |
| trainers.stationary-heat-kernels  |  [stat_heat_kernels.py](./heatsplats/trainers/stat_heat_kernels.py) | StationaryHeatKernelsTrainer |


## Density Controllers

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| density_controllers.opacity  | [opacity.py](./heatsplats/density_controllers/opacity.py) | OpacityController |
