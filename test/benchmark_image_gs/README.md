# Running ImageGS Benchmarks

1. Create *image_gs* enviroment https://github.com/NYU-ICL/image-gs
2. Install the following library:

    ```bash
    pip install "ray[tune]" "optuna>=3.0.0" pydantic scikit-learn
    ```
3. Clone their repo in `test/benchmark_image_gs` with:
    ```bash
    cd test/benchmark_image_gs
    git clone https://github.com/NYU-ICL/image-gs.git
    ```
4. Apply the patches 
    ```bash
    cd image-gs
    git apply ../for_hktex_comparison.patch
    ```


Now that everything is properly set up, we will first optimise for the UV textures and save them as images, then render them with our codebase.

5. Run `test/benchmark_image_gs/image_gs_fitting.py` with the image-gs enviroment
6. Run `test/benchmark_image_gs/evaluate_image_gs.py` with the hktex enviroment