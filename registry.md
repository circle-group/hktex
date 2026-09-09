# HkTex Class Registry

## Data

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| data.uv-texture-sampler  | [uv_texture_sampler.py](./hktex/data/uv_texture_sampler.py) | UvTextureSamplerDataModule |
| data.vertex-colours  |  [vertex_colours.py](./hktex/data/vertex_colours.py) | VertexColoursDataModule |
| data.known-heat-vertex-colours  |  [known_heat_vertex_colours.py](./hktex/data/known_heat_vertex_colours.py) | KnownHeatVertexColoursDataModule |


## Modules

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| modules.cpu-geodesic-tracer  | [tracer.py](./hktex/modules/tracer.py) | CPUGeodesicTracer |
| modules.gpu-geodesic-tracer  | [tracer.py](./hktex/modules/tracer.py) | GPUGeodesicTracer |
| modules.eigen-albo-interpolation  |  [eigen_albo.py](./hktex/modules/eigen_albo.py) | EigenAlboInterpolation |

## Trainers

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| trainers.uv-texture  | [uv_texture.py](./hktex/trainers/uv_texture.py) | UvTextureTrainer |
| trainers.vertex-colours  |  [vertex_colours.py](./hktex/trainers/vertex_colours.py) | VertexColoursTrainer |
| trainers.stationary-heat-kernels  |  [stat_heat_kernels.py](./hktex/trainers/stat_heat_kernels.py) | StationaryHeatKernelsTrainer |


## Density Controllers

|   Registry    |     File      |     Class     |
| ------------- | ------------- | ------------- |
| density_controllers.importance_pruning  | [importance_pruning.py](./hktex/density_controllers/importance_pruning.py) | ImportancePruningController |
| density_controllers.error_based_densification  | [error_based.py](./hktex/density_controllers/error_based.py) | ErrorDensificationController |