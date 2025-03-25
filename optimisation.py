import torch
import numpy as np

import matplotlib.pyplot as plt
from termcolor import colored

import eigen_albo_interpolation
import utils


class OptimiseFixedHeatKernels:
    def __init__(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        fnorms: np.ndarray,
        n_sources: int,
        lr_mult: float = 1,
        k_eig: int = 256,
        fpath: str = None,
        device: str = "cpu",
    ):
        self.device = device

        self._eigalbo_interp = eigen_albo_interpolation.EigenAlboInterpolation(
            verts, faces, fnorms, k_eig=k_eig, fpath=fpath, device=device
        )

        self._source_idxs = torch.tensor([3804, 0, 4274])
        self._idx_range = torch.arange(n_sources)
        self._n_sources = n_sources

        self._verts = torch.tensor(verts)
        self._diff_time_scaler_func = lambda x: 10 ** (4 * torch.tanh(x) - 2)
        self._anis_scaler = 1

        self._anis_act = lambda x: x # torch.exp
        
        self._lr_mult = lr_mult
        self._splats, self._optims = self._make_splats_and_optimisers(n_sources)

    def _make_splats_and_optimisers(self, n_sources: int):
        kernel_colours = torch.rand((n_sources, 3), dtype=torch.float)
        angles = (torch.rand(n_sources) + torch.pi / 4) * 0.1
        anisotropies = torch.rand(n_sources)
        diff_times = torch.rand(n_sources)

        params = [
            # name, value, lr
            ("kernel_colours", torch.nn.Parameter(kernel_colours), 1e-2),
            ("angles", torch.nn.Parameter(angles), 1e-4),
            ("anisotropies", torch.nn.Parameter(anisotropies), 1e-3),
            ("diff_times", torch.nn.Parameter(diff_times), 1e-2),
        ]

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(
            self.device
        )

        optimisers = {
            name: torch.optim.SGD([{"params": splats[name], "lr": self._lr_mult * lr}])
            for name, _, lr in params
        }
        return splats, optimisers

    def _make_gt_colours(self):
        gt_colours = torch.zeros([self._n_sources, *self._verts.shape], device=self.device)
        gt_colours[self._idx_range, self._source_idxs, :] = torch.tensor(
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float, device=self.device
        )

        albo_evals, albo_evecs, mass = (
            self._eigalbo_interp.get_albo_eigenquantities(
                angles=torch.deg2rad(torch.tensor([45.0, 18.3, 10.0], device=self.device)),
                scales=torch.tensor([33.0, 60, 5.2], device=self.device),
            )
        )

        gt_colours = utils.heat_diffusion(
            gt_colours,
            mass,
            albo_evals,
            albo_evecs,
            torch.tensor([0.001, 0.1, 0.01], device=self.device),
        )

        print(
            f"GT -> ",
            colored(f"Angles: {[45.0, 18.3, 10.0]}, ", "yellow"),
            colored(f"Anisotropies: {[33.0, 60, 5.2]}, ", "green"),
            colored(f"Diff times: {[0.001, 0.1, 0.01]}, ", "blue"),
            colored(f"Kernel colours: {[1, 0, 0, 0, 1, 0, 0, 0, 1]}", "red"),
        )

        gt_colours = gt_colours.sum(dim=0)
        gt_colours = self._normalize(gt_colours)
        # print(gt_colours.mean(), gt_colours.std(), gt_colours.min(), gt_colours.max())

        return gt_colours

    def optimise(self, n_iter=100):
        v_colours_buffer = torch.zeros([self._n_sources, *self._verts.shape], device=self.device)
        gt_colours = self._make_gt_colours()

        print(f"INITIAL -> ", self._colored_print_opt_params)

        errors_lists = {k: [] for k in self._splats.keys()}

        for i in range(n_iter):
            v_colours = v_colours_buffer.clone().detach().requires_grad_(True)
            # v_colours[self._idx_range, self._source_idxs, :] = (
            #     self._kernel_colours
            # )
            v_colours = v_colours.index_put(
                (self._idx_range, self._source_idxs),
                self._splats["kernel_colours"],
            )

            albo_evals, albo_evecs, mass = (
                self._eigalbo_interp.get_albo_eigenquantities(
                    angles=self._splats["angles"],
                    scales=self._anis_act(self._splats["anisotropies"]) * self._anis_scaler,
                )
            )

            v_colours = utils.heat_diffusion(
                v_colours,
                mass,
                albo_evals,
                albo_evecs,
                self._diff_time_scaler_func(self._splats["diff_times"]),
            )
            v_colours = self._normalize(v_colours.sum(dim=0))

            if i == 0:
                init_colours = v_colours.clone().detach()

            loss = 1e-2 * (v_colours - gt_colours).pow(2).sum()
            # with torch.no_grad():
            #     diff = (v_colours - gt_colours).pow(2).sum(dim=-1)
            #     print(diff.shape, diff.mean(), diff.std(), diff.min(), diff.max())

            loss.backward()

            if i == 0 or (i+1) % 100 == 0:
                for name, param in self._splats.items():
                    print(f"{name}: {param.grad.data.norm(2)}")

            for optimizer in self._optims.values():
                optimizer.step()
                optimizer.zero_grad()

            if i == 0 or (i + 1) % 100 == 0:
                print(
                    f"Iteration: {i + 1} -> Loss: {loss.item()}.",
                    self._errors["printables"],
                )

            errors = self._errors
            for k in self._splats.keys():
                errors_lists[k].append(errors[k].item())

        print(f"FINAL -> ", self._colored_print_opt_params)

        self.plot_errors(errors_lists)

        return v_colours.sum(dim=0), gt_colours, init_colours
    
    def _normalize(self, colours):
        cmin, cmax = colours.min(), colours.max()
        colours = (colours - cmin) / (
            cmax - cmin
        )
        return colours

    @property
    def _colored_print_opt_params(self):
        angles = torch.rad2deg(self._splats["angles"]).detach().cpu().numpy()
        anisotropies = (
            self._anis_act(self._splats["anisotropies"].detach()).cpu().numpy() * self._anis_scaler
        )
        diff_times = (
            self._diff_time_scaler_func(self._splats["diff_times"])
            .detach()
            .cpu()
            .numpy()
        )
        kernel_colours = (
            self._splats["kernel_colours"].view(-1).detach().cpu().numpy()
        )
        return (
            colored(f"Angles: {angles}, ", "yellow")
            + colored(f"Anisotropies: {anisotropies}, ", "green")
            + colored(f"Diff times: {diff_times}, ", "blue")
            + colored(f"Kernel colours: {kernel_colours}", "red")
        )

    @property
    def _errors(self):
        angles = self._splats["angles"] % (torch.pi)
        angles_error = (
            (angles - torch.deg2rad(torch.tensor([45.0, 18.3, 10.0], device=self.device)))
            .pow(2)
            .sum()
            .pow(0.5)
        )
        anisotropies = self._anis_act(self._splats["anisotropies"]) * self._anis_scaler
        anisotropies_error = (
            (anisotropies - torch.tensor([33.0, 60, 5.2], device=self.device)).pow(2).sum().pow(0.5)
        )
        diff_times = self._diff_time_scaler_func(self._splats["diff_times"])
        diff_times_error = (
            (diff_times - torch.tensor([0.001, 0.1, 0.01], device=self.device))
            .pow(2)
            .sum()
            .pow(0.5)
        )
        gt_kernel_colors = torch.tensor(
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float, device=self.device
        )
        kernel_colours_error = (
            (self._splats["kernel_colours"] - gt_kernel_colors)
            .pow(2)
            .sum()
            .pow(0.5)
        )
        return {
            "printables": (
                "ERRORS: "
                + colored(f"Angles: {angles_error}, ", "yellow")
                + colored(f"Anisotropies: {anisotropies_error}, ", "green")
                + colored(f"Diff times: {diff_times_error}, ", "blue")
                + colored(f"Kernel colours: {kernel_colours_error}", "red")
            ),
            "angles": angles_error,
            "anisotropies": anisotropies_error,
            "diff_times": diff_times_error,
            "kernel_colours": kernel_colours_error,
        }

    @staticmethod
    def plot_errors(errors_lists):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(errors_lists["angles"], label="Angles Error")
        axs[0, 0].set_title("Angles Error")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Error")

        axs[0, 1].plot(errors_lists["anisotropies"], label="Anisotropies Error")
        axs[0, 1].set_title("Anisotropies Error")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Error")

        axs[1, 0].plot(errors_lists["diff_times"], label="Diff Times Error")
        axs[1, 0].set_title("Diff Times Error")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Error")

        axs[1, 1].plot(
            errors_lists["kernel_colours"], label="Kernel Colours Error"
        )
        axs[1, 1].set_title("Kernel Colours Error")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Error")

        for ax in axs.flat:
            ax.legend()

        plt.tight_layout()
        plt.show()


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

    optimisation = OptimiseFixedHeatKernels(
        verts, faces, fnorm, n_sources=3, k_eig=256, fpath=mesh_path, device="cuda"
    )
    v_colours, gt_colours, init_colours = optimisation.optimise(n_iter=500)

    v_colours = (v_colours - v_colours.min()) / (
        v_colours.max() - v_colours.min()
    )
    v_colours *= 255
    v_colours = v_colours.squeeze().detach().cpu().numpy()

    gt_colours = (gt_colours - gt_colours.min()) / (
        gt_colours.max() - gt_colours.min()
    )
    gt_colours *= 255
    gt_colours = gt_colours.squeeze().detach().cpu().numpy()

    init_colours = (init_colours - init_colours.min()) / (
        init_colours.max() - init_colours.min()
    )
    init_colours *= 255
    init_colours = init_colours.squeeze().detach().cpu().numpy()

    gt_mesh = mesh.copy()
    gt_mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=gt_colours)

    v_mesh = mesh.copy()
    v_mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=v_colours)

    init_mesh = mesh.copy()
    init_mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=init_colours
    )
