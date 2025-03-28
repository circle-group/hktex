import trimesh
import potpourri3d as pp3d

import numpy as np
import utils
import time


mesh_path = "../objects/spot_triangulated.obj"
mesh = utils.load_mesh(mesh_path, show=False)

tracer = pp3d.GeodesicTracer(mesh.vertices, mesh.faces)
start_time = time.time()
for _ in range(100_000):
    trace_pts = tracer.trace_geodesic_from_vertex(22, np.array((0.3, 0.5, 0.4)))
print(f"Potpourri3D geodesic tracing time: {time.time() - start_time:.4f} seconds")

trim_path = trimesh.load_path(trace_pts, colors=[[255, 0, 0, 255]])
scene = trimesh.Scene([mesh, trim_path])
