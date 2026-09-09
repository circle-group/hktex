# Examples

Run examples from the repository root:

```bash
python -m examples.knn
python -m examples.faiss
python -m examples.full_pipeline
python -m examples.grid_visual
python -m examples.grid_neighbor_video
python -m examples.knn_heat_compare
```

Notes:
- `examples.faiss` requires CUDA + FAISS GPU.
- `examples.full_pipeline` runs the end-to-end path:
  FAISS KNN -> gather -> post-weights -> heat query.
- `examples.knn_heat_compare` compares old vs new heat diffusion outputs and gradients.
