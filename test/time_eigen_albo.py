import sys
import os
from pathlib import Path

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import time
import argparse
import numpy as np
import pandas as pd
import torch
import scipy.optimize
import gc
from ray import tune

# -------------------------------------------------------------------------
# Monkey-patching setup to record timings without modifying source files
# -------------------------------------------------------------------------

timing_stats = {
    "lbo_s": 0.0,
    "eig_s": 0.0,
    "align_s": 0.0,
    "hungarian_s": 0.0,
}


def reset_timings():
    for k in timing_stats:
        timing_stats[k] = 0.0


def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_is_patched = False


def setup_patching():
    global _is_patched
    if _is_patched:
        return

    import heatsplats.modules.eigen_albo as ea_module
    import scipy.optimize

    orig_linear_sum_assignment = scipy.optimize.linear_sum_assignment
    orig_get_anisotropic_lbo = ea_module.get_anisotropic_lbo
    orig_compute_eig_laplacian = ea_module.compute_eig_laplacian
    orig_align_eigen = ea_module.align_eigen

    def patch_get_lbo(*args, **kwargs):
        _gpu_sync()
        t0 = time.perf_counter()
        res = orig_get_anisotropic_lbo(*args, **kwargs)
        _gpu_sync()
        timing_stats["lbo_s"] += time.perf_counter() - t0
        return res

    def patch_compute_eig(*args, **kwargs):
        _gpu_sync()
        t0 = time.perf_counter()
        res = orig_compute_eig_laplacian(*args, **kwargs)
        _gpu_sync()
        timing_stats["eig_s"] += time.perf_counter() - t0
        return res

    def patch_align_eigen(*args, **kwargs):
        _gpu_sync()
        t0 = time.perf_counter()
        res = orig_align_eigen(*args, **kwargs)
        _gpu_sync()
        timing_stats["align_s"] += time.perf_counter() - t0
        return res

    def patch_hungarian(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_linear_sum_assignment(*args, **kwargs)
        timing_stats["hungarian_s"] += time.perf_counter() - t0
        return res

    ea_module.get_anisotropic_lbo = patch_get_lbo
    ea_module.compute_eig_laplacian = patch_compute_eig
    ea_module.align_eigen = patch_align_eigen
    scipy.optimize.linear_sum_assignment = patch_hungarian

    _is_patched = True


# -------------------------------------------------------------------------
# Benchmark Script Logic
# -------------------------------------------------------------------------


def safe_name(fname: str) -> str:
    return fname.replace("/", "_").replace("\\", "_").replace(".glb", "")


def timed_eigen_albo(mesh, config_dict):
    from heatsplats.modules import EigenAlboInterpolation

    reset_timings()
    _gpu_sync()
    t0 = time.perf_counter()

    # use_precomputed=False forces the precomputation to run from scratch
    eigalbo = EigenAlboInterpolation(config_dict, mesh)

    _gpu_sync()
    total_s = time.perf_counter() - t0

    timings = {
        "total_s": total_s,
        "lbo_s": timing_stats["lbo_s"],
        "eig_s": timing_stats["eig_s"],
        "align_s": timing_stats["align_s"],
        "hungarian_s": timing_stats["hungarian_s"],
    }

    del eigalbo
    gc.collect()
    torch.cuda.empty_cache()
    return timings


def infer_trainable(config):
    setup_patching()

    fname = config["filename"]
    mesh_root = config["root"]
    mesh_path = os.path.join(mesh_root, fname)

    from heatsplats.utils import load_mesh
    from heatsplats.modules import Mesh

    try:
        tri_mesh = load_mesh(mesh_path, merge_tex=False, bake_vert_colors=False)
        our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")
        n_verts = our_mesh.N_verts

        eigalbo_config = {
            "k_eig": config["k_eig"],
            "use_precomputed": False,
            "precompute_anisotropies": [1, 5, 15, 30, 60, 100, 200],
            "precompute_angles_every_deg": 30,
            "mesh_path": mesh_path,
            "distance_weighting": "none",
            "local_frames": "principal_curvatures",
        }

        # Warmup run (highly recommended if n_runs > 1 due to Numba/JIT overheads)
        if config["n_runs"] > 1:
            _ = timed_eigen_albo(our_mesh, eigalbo_config)

        runs = []
        for _ in range(config["n_runs"]):
            timings = timed_eigen_albo(our_mesh, eigalbo_config)
            runs.append(timings)

        del our_mesh
        del tri_mesh
        gc.collect()
        torch.cuda.empty_cache()

        # Summarize
        keys = runs[0].keys()
        row = {"filename": fname, "n_verts": n_verts}
        for k in keys:
            vals = [r[k] for r in runs]
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals))
            row[f"{k}_min"] = float(np.min(vals))

        individual_results_dir = os.path.join(
            config["output_dir"], "individual_results"
        )
        os.makedirs(individual_results_dir, exist_ok=True)
        individual_csv = os.path.join(individual_results_dir, f"{safe_name(fname)}.csv")
        pd.DataFrame([row]).to_csv(individual_csv, index=False)

        tune.report(row)
        return row

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        error_row = {"filename": fname, "error": str(e)}

        individual_results_dir = os.path.join(
            config["output_dir"], "individual_results"
        )
        os.makedirs(individual_results_dir, exist_ok=True)
        individual_csv = os.path.join(individual_results_dir, f"{safe_name(fname)}.csv")
        pd.DataFrame([error_row]).to_csv(individual_csv, index=False)

        tune.report({"error": str(e), "filename": fname})
        return error_row


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/data2/objaverse")
    p.add_argument(
        "--benchmark_csv", type=str, default="outputs/benchmark/benchmark_results.csv"
    )
    p.add_argument("--output_dir", type=str, default="outputs/time_eigen_albo")
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--k_eig", type=int, default=256)
    p.add_argument("--max_items", type=int, default=100)
    p.add_argument("--max_concurrent_trials", type=int, default=1)
    p.add_argument("--gpus_per_trial", type=float, default=1.0)
    args = p.parse_args()

    df = pd.read_csv(args.benchmark_csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    if "mse" in df.columns:
        df = df[np.isfinite(df["mse"])]
    df = df.sort_values(by="filename").reset_index(drop=True)
    print(f"Loaded {len(df)} entries, keeping {args.max_items}")
    df = df.head(args.max_items)

    individual_results_dir = os.path.join(args.output_dir, "individual_results")
    os.makedirs(individual_results_dir, exist_ok=True)

    items = []
    for _, r in df.iterrows():
        fname = r["filename"]
        individual_csv = os.path.join(individual_results_dir, f"{safe_name(fname)}.csv")
        if os.path.exists(individual_csv):
            print(f"Skipping {fname}, already processed.")
            continue
        items.append({"filename": fname})

    print(f"Starting with {len(items)} items using Ray Tune")

    if len(items) == 0:
        print("All items already processed.")
        sys.exit(0)

    search_space = {
        "item": tune.grid_search(items),
        "root": args.root,
        "output_dir": os.path.abspath(args.output_dir),
        "n_runs": args.n_runs,
        "k_eig": args.k_eig,
    }

    def wrapper(config):
        item = config.pop("item")
        config["filename"] = item["filename"]
        return infer_trainable(config)

    tuner = tune.Tuner(
        tune.with_resources(wrapper, resources={"gpu": args.gpus_per_trial}),
        param_space=search_space,
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent_trials),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(args.output_dir), name="time_eigen_albo"
        ),
    )

    results = tuner.fit()
    df_out = results.get_dataframe()
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(args.output_dir, "time_eigen_albo_results.csv")
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_df = existing_df.loc[
                :, ~existing_df.columns.str.contains("^Unnamed")
            ]
            df_out = pd.concat([existing_df, df_out], ignore_index=True)
            if "filename" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["filename"], keep="last")
        except Exception as e:
            print(f"Could not merge with existing results: {e}")

    if "filename" in df_out.columns:
        df_out = df_out.sort_values("filename")
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} rows to {output_csv}")
