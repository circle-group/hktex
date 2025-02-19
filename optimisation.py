import torch

import matplotlib.pyplot as plt
from termcolor import colored

import eigen_albo_interpolation
import utils


class OptimiseFixedHeatKernels:
    def __init__(
        self, verts, faces, fnorms, n_sources, lr=1e-4, k_eig=256, fpath=None
    ):

        self._eigalbo_interp = eigen_albo_interpolation.EigenAlboInterpolation(
            verts, faces, fnorms, k_eig=k_eig, fpath=fpath
        )

        self._source_idxs = torch.tensor([3804, 0, 4274])
        self._idx_range = torch.arange(n_sources)
        self._n_sources = n_sources

        self._verts = torch.tensor(verts)
        self._diff_time_scaler = 1e-2
        self._angle_conversion_coeff = 180.0 / torch.pi
        self._anis_scaler = 10

        # Optimisation params (leaf tensor)
        self._kernel_colours = torch.rand(
            (n_sources, 3), dtype=torch.float, requires_grad=True
        )
        self._angles = torch.rand(n_sources, requires_grad=True)
        self._anisotropies = torch.rand(n_sources, requires_grad=True)
        self._diff_times = torch.rand(n_sources, requires_grad=True)

        params = [
            self._kernel_colours,  # colour of each source
            self._angles,  # angle of the heat kernel
            self._anisotropies,  # anisotropy of the heat kernel
            self._diff_times,  # size of the heat kernel
        ]

        self._optim = torch.optim.Adam(params, lr=lr)

    def _make_gt_colours(self):
        gt_colours = torch.zeros([self._n_sources, *self._verts.shape])
        gt_colours[self._idx_range, self._source_idxs, :] = torch.tensor(
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float
        )

        albo_evals, albo_evecs, mass = (
            self._eigalbo_interp.get_albo_eigenquantities(
                angles=torch.tensor([45.0, 18.3, 10.0]),
                scales=torch.tensor([33.0, 60, 5.2]),
            )
        )

        gt_colours = utils.heat_diffusion(
            gt_colours,
            mass,
            albo_evals,
            albo_evecs,
            torch.tensor([0.001, 0.1, 0.01]),
        )

        gt_colours = gt_colours.sum(dim=0)
        return gt_colours

    def optimise(self, n_iter=100):
        v_colours_buffer = torch.zeros([self._n_sources, *self._verts.shape])
        print(f"INITIAL -> ", self._colored_print_opt_params)

        gt_colours = self._make_gt_colours()

        errors_list = {
            "angles_error": [],
            "anisotropies_error": [],
            "diff_times_error": [],
            "kernel_colours_error": [],
        }

        for i in range(n_iter):
            v_colours = v_colours_buffer.clone().detach().requires_grad_(True)
            # v_colours[self._idx_range, self._source_idxs, :] = (
            #     self._kernel_colours
            # )
            v_colours = v_colours.index_put(
                (self._idx_range, self._source_idxs), self._kernel_colours
            )

            albo_evals, albo_evecs, mass = (
                self._eigalbo_interp.get_albo_eigenquantities(
                    angles=self._angles * self._angle_conversion_coeff,
                    scales=self._anisotropies * self._anis_scaler,
                )
            )

            v_colours = utils.heat_diffusion(
                v_colours,
                mass,
                albo_evals,
                albo_evecs,
                self._diff_times * self._diff_time_scaler,
            )

            loss = (v_colours.sum(dim=0) - gt_colours).pow(2).sum()

            loss.backward()
            self._optim.step()
            self._optim.zero_grad()

            if (i + 1) % 20 == 0:
                print(
                    f"Iteration: {i + 1} -> Loss: {loss.item()}.",
                    self._errors["printables"],
                    f"vcolour_min: {v_colours.min()} ",
                    f"vcolour_max: {v_colours.max()}",
                )

            errors_list["angles_error"].append(
                self._errors["angles_error"].item()
            )
            errors_list["anisotropies_error"].append(
                self._errors["anisotropies_error"].item()
            )
            errors_list["diff_times_error"].append(
                self._errors["diff_times_error"].item()
            )
            errors_list["kernel_colours_error"].append(
                self._errors["kernel_colours_error"].item()
            )

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(errors_list["angles_error"], label="Angles Error")
        axs[0, 0].set_title("Angles Error")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Error")

        axs[0, 1].plot(
            errors_list["anisotropies_error"], label="Anisotropies Error"
        )
        axs[0, 1].set_title("Anisotropies Error")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Error")

        axs[1, 0].plot(
            errors_list["diff_times_error"], label="Diff Times Error"
        )
        axs[1, 0].set_title("Diff Times Error")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Error")

        axs[1, 1].plot(
            errors_list["kernel_colours_error"], label="Kernel Colours Error"
        )
        axs[1, 1].set_title("Kernel Colours Error")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Error")

        for ax in axs.flat:
            ax.legend()

        plt.tight_layout()
        plt.show()

        print(
            f"FINAL -> ",
            self._errors["printables"],
            self._colored_print_opt_params,
        )

        return v_colours.sum(dim=0), gt_colours

    @property
    def _colored_print_opt_params(self):
        angles = self._angles.detach().numpy() * self._angle_conversion_coeff
        anisotropies = self._anisotropies.detach().numpy() * self._anis_scaler
        diff_times = self._diff_times.detach().numpy() * self._diff_time_scaler
        kernel_colours = self._kernel_colours.view(-1).detach().numpy()
        return (
            colored(f"Angles: {angles}, ", "yellow")
            + colored(f"Anisotropies: {anisotropies}, ", "green")
            + colored(f"Diff times: {diff_times}, ", "blue")
            + colored(f"Kernel colours: {kernel_colours}", "red")
        )

    @property
    def _errors(self):
        angles = self._angles * self._angle_conversion_coeff
        angles_error = (
            (angles - torch.tensor([45.0, 18.3, 10.0])).pow(2).sum().pow(0.5)
        )
        anisotropies = self._anisotropies * self._anis_scaler
        anisotropies_error = (
            (anisotropies - torch.tensor([33.0, 60, 5.2])).pow(2).sum().pow(0.5)
        )
        diff_times = self._diff_times * self._diff_time_scaler
        diff_times_error = (
            (diff_times - torch.tensor([0.001, 0.1, 0.01]))
            .pow(2)
            .sum()
            .pow(0.5)
        )
        gt_kernel_colors = torch.tensor(
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float
        )
        kernel_colours_error = (
            (self._kernel_colours - gt_kernel_colors).pow(2).sum().pow(0.5)
        )
        return {
            "printables": (
                "ERRORS:"
                + colored(f"Angles: {angles_error}, ", "yellow")
                + colored(f"Anisotropies: {anisotropies_error}, ", "green")
                + colored(f"Diff times: {diff_times_error}, ", "blue")
                + colored(f"Kernel colours: {kernel_colours_error}", "red")
            ),
            "angles_error": angles_error,
            "anisotropies_error": anisotropies_error,
            "diff_times_error": diff_times_error,
            "kernel_colours_error": kernel_colours_error,
        }


if __name__ == "__main__":
    import trimesh
    import numpy as np

    mesh_path = "../objects/spot_triangulated.obj"
    mesh = utils.load_mesh(mesh_path, show=False)

    v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    mesh = trimesh.Trimesh(v, f)

    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    fnorm = np.array(mesh.face_normals)

    optimisiation = OptimiseFixedHeatKernels(
        verts, faces, fnorm, n_sources=3, lr=5e-2, k_eig=256, fpath=mesh_path
    )
    v_colours, gt_colours = optimisiation.optimise(n_iter=200)

    v_colours = (v_colours - v_colours.min()) / (
        v_colours.max() - v_colours.min()
    )
    v_colours *= 255
    v_colours = v_colours.squeeze().detach().numpy()

    gt_colours = (gt_colours - gt_colours.min()) / (
        gt_colours.max() - gt_colours.min()
    )
    gt_colours *= 255
    gt_colours = gt_colours.squeeze().detach().numpy()

    gt_mesh = mesh.copy()
    gt_mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=gt_colours)

    v_mesh = mesh.copy()
    v_mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=v_colours)
