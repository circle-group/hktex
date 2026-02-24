# Algorithm

## Pre-computation (before training)

1- Calculate isotropic eigenpairs
2- Calculate anisotropic eigenpairs for all rotation/angle combinations

## Graph Building (at the start of each training iteration / once for inference)

1- Interpolate isotropic embeddings into source locations
2- Build the KNN graph using interpolated isotropic embeddings
3- For each source, find the 4 rotation/anisotrophy interpolation locations, save the grid for below
4- For each source, interpolate anisotrophy + barycentric to get eigenpairs
5- Cache heat from sources to themselves (S)

## Point querying (per point)

### Prepare Points

1- Interpolate isotropic embeddings into point locations
2- Get indices from the KNN graph search (Px100)
3- Calculate differentiable biharmonic distances from indices (Px100), points to sources
4- Calculate biharmonic distance weights (Px100), points to sources
5- Recover the grid from 3rd step of Graph Building
6- For each query, interpolate anisotrophy + barycentric to get eigenpairs (same as step 4 of Graph Building), (Phi_points)

### Diffuse Heat

1- Gather correct eigenvectors for points from sources using knn indices (Phi_source) (SxE -> Px100xE)
2- Using the eigenvalues, Phi_points, and Phi_source; diffuse heat (Px100)
3- Weight the outputs using biharmonic distance weights from Step 4 of Prepare Points
4- Normalize by the cached heat from sources to themselves (Step 5 of Graph Building) (first gather S -> Px100 with knn indices, then divide)

### Reduction

1- Rescaled soft step filtering
2- From kernel colours (Sx3) gather correct locations (Px100x3)
3- Find colours by multiplying filtered diracs * gathered colours
4- Do a knn search from Px100 -> Px10, save local topk indices (100 -> 10)
5- Gather colour contribution using local top k indices, normalize by contributions form local top k

6- Gather correct global topk indices for Px10 that point to global sources
7- Find kernel contributions SxP using a scatter
8- Return colours, contributions and global topk indices
