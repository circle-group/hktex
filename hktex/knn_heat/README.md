# knn_heat

Utilities for KNN-based spectral gather and heat diffusion in PyTorch.

## What is included
- `knn_heat/faiss_knn.py`: FAISS GPU KNN index wrapper (`build/search/knn_distances`)
- `knn_heat/knn_gather.py`: spectral grid selection + source/query gather
- `knn_heat/knn_heat.py`: cached heat diffusion over KNN pairs (`build/query/reset`)

## Run examples
From the repository root:

```bash
python -m examples.knn
python -m examples.faiss
python -m examples.knn_heat_compare
```

See `examples/README.md` for the full list.
