import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from termcolor import colored
from abc import abstractmethod

from tqdm import tqdm

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
        normalize_colours: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = device

        self._eigalbo_interp = eigen_albo_interpolation.EigenAlboInterpolation(
            verts, faces, fnorms, k_eig=k_eig, fpath=fpath, device=device
        )

        self._n_sources = n_sources
        self._source_idxs = torch.randint(0, len(verts), (n_sources,), device=device)
        self._idx_range = torch.arange(n_sources, device=device)

        self._verts = torch.tensor(verts, device=device)
        self._diff_time_scaler_func = lambda x: 10 ** (4 * torch.tanh(x) - 2)

        self._angle_scaler = torch.pi
        # -> [0, pi]
        self._angle_act = lambda x: self._angle_scaler * F.hardsigmoid(x)
        self._anis_scaler = 100
        # -> [0, 100]
        self._anis_act = lambda x: self._anis_scaler * F.hardsigmoid(x)
        # self._anis_act = torch.exp

        self._colour_act = lambda x: x

        self._normalize_colours = normalize_colours
        self._lr_mult = lr_mult
        self._splats, self._optims = self._make_splats_and_optimisers(n_sources)

    def _make_splats_and_optimisers(self, n_sources: int):
        kernel_colours = torch.rand((n_sources, 3), dtype=torch.float)
        # angles = (torch.rand(n_sources) + torch.pi / 4) * 0.1
        # anisotropies = torch.rand(n_sources)
        # kernel_colours = torch.randn((n_sources, 3), dtype=torch.float)
        angles = torch.randn(n_sources)
        anisotropies = torch.randn(n_sources)
        diff_times = torch.rand(n_sources)

        params = [
            # name, value, lr
            ("kernel_colours", torch.nn.Parameter(kernel_colours), 1e-3),
            ("angles", torch.nn.Parameter(angles), 1e-3),
            ("anisotropies", torch.nn.Parameter(anisotropies), 1e-3),
            ("diff_times", torch.nn.Parameter(diff_times), 1e-3),
        ]

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.device)

        optimisers = {
            name: torch.optim.Adam([{"params": splats[name], "lr": self._lr_mult * lr}])
            for name, _, lr in params
        }
        return splats, optimisers

    @property
    def kernel_colours(self):
        return self._colour_act(self._splats["kernel_colours"])

    @property
    def angles(self):
        return self._angle_act(self._splats["angles"])

    @property
    def anisotropies(self):
        return self._anis_act(self._splats["anisotropies"])

    @property
    def diff_times(self):
        return self._diff_time_scaler_func(self._splats["diff_times"])

    def optimise(self, n_iter=100):
        v_colours_buffer = torch.zeros(
            [self._n_sources, *self._verts.shape], device=self.device
        )
        gt_colours = self._make_gt_colours()

        print(f"INITIAL -> ", self._colored_print_opt_params)

        errors_lists = {k: [] for k in self._splats.keys()}

        for i in (pbar := tqdm(range(n_iter))):
            v_colours = v_colours_buffer.clone().detach().requires_grad_(True)
            v_colours = v_colours.index_put(
                (self._idx_range, self._source_idxs),
                self.kernel_colours,
            )

            albo_evals, albo_evecs, mass = (
                self._eigalbo_interp.get_albo_eigenquantities(
                    angles=self.angles, scales=self.anisotropies
                )
            )

            v_colours = utils.heat_diffusion_reduce(
                v_colours,
                mass,
                albo_evals,
                albo_evecs,
                self.diff_times,
            )
            if self._normalize_colours:
                v_colours = self._normalize(v_colours)

            if i == 0:
                init_colours = v_colours.clone().detach()

            # loss = (v_colours - gt_colours).pow(2).sum()
            loss = F.mse_loss(v_colours, gt_colours, reduction="sum") / v_colours.shape[0]
            # with torch.no_grad():
            #     diff = (v_colours - gt_colours).pow(2).sum(dim=-1)
            #     print(diff.shape, diff.mean(), diff.std(), diff.min(), diff.max())

            loss.backward()

            with torch.no_grad():
                if i == 0 or (i + 1) % 100 == 0:
                    for name, param in self._splats.items():
                        print(f"{name}: {param.grad.data.norm(2)}")

            for optimizer in self._optims.values():
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                pbar.set_postfix_str(f"Loss: {loss.item():0.4f}")
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

        return v_colours, gt_colours, init_colours

    def _normalize(self, colours):
        cmin, cmax = colours.min(), colours.max()
        colours = (colours - cmin) / (cmax - cmin)
        return colours

    @property
    def _colored_print_opt_params(self):
        angles = torch.rad2deg(self.angles).detach().cpu().numpy()
        anisotropies = self.anisotropies.detach().cpu().numpy()
        diff_times = self.diff_times.detach().cpu().numpy()
        kernel_colours = self.kernel_colours.view(-1).detach().cpu().numpy()
        return (
            colored(f"Angles: {angles}, ", "yellow")
            + colored(f"Anisotropies: {anisotropies}, ", "green")
            + colored(f"Diff times: {diff_times}, ", "blue")
            + colored(f"Kernel colours: {kernel_colours}", "red")
        )

    @property
    def kernel_centres(self):
        return self._verts[self._source_idxs]

    @abstractmethod
    def _make_gt_colours(self):
        pass

    @property
    @abstractmethod
    def _errors(self):
        pass

    @staticmethod
    @abstractmethod
    def plot_errors(errors_lists):
        pass


class OptimiseKnownFixedHeatKernels(OptimiseFixedHeatKernels):
    def __init__(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        fnorms: np.ndarray,
        n_sources: int,
        lr_mult: float = 1,
        k_eig: int = 256,
        fpath: str = None,
        normalize_colours: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            verts,
            faces,
            fnorms,
            n_sources,
            lr_mult,
            k_eig,
            fpath,
            normalize_colours,
            device,
            **kwargs,
        )
        self._gt_splats = None

    def _make_gt_colours(self):
        if self._n_sources == 3:
            self._source_idxs = torch.tensor([3804, 0, 4274])

            self._gt_splats = {
                "angles": torch.tensor([45.0, 18.3, 10.0], device=self.device),
                "anisotropies": torch.tensor([33.0, 60, 5.2], device=self.device),
                "diff_times": torch.tensor([0.001, 0.1, 0.01], device=self.device),
                "kernel_colours": torch.tensor(
                    [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                    dtype=torch.float,
                    device=self.device,
                ),
            }
        else:
            self._gt_splats = {
                "angles": torch.rand(self._n_sources, device=self.device) * 180,
                "anisotropies": (100 * torch.rand(self._n_sources, device=self.device)),
                "diff_times": self._diff_time_scaler_func(
                    torch.rand(self._n_sources, device=self.device)
                ),
                "kernel_colours": torch.rand(
                    (self._n_sources, 3),
                    dtype=torch.float,
                    device=self.device,
                ),
            }

        gt_colours = torch.zeros(
            [self._n_sources, *self._verts.shape], device=self.device
        )
        gt_colours[self._idx_range, self._source_idxs, :] = self._gt_splats[
            "kernel_colours"
        ]

        albo_evals, albo_evecs, mass = self._eigalbo_interp.get_albo_eigenquantities(
            angles=torch.deg2rad(self._gt_splats["angles"]),
            scales=self._gt_splats["anisotropies"],
        )

        gt_colours = utils.heat_diffusion(
            gt_colours.to(self.device),
            mass,
            albo_evals,
            albo_evecs,
            self._gt_splats["diff_times"],
        )

        gta = self._gt_splats["angles"].detach().cpu().numpy()
        gts = self._gt_splats["anisotropies"].detach().cpu().numpy()
        gtt = self._gt_splats["diff_times"].detach().cpu().numpy()
        gtc = self._gt_splats["kernel_colours"].detach().cpu().numpy()
        print(
            f"GT -> ",
            colored(f"Angles: {gta}, ", "yellow"),
            colored(f"Anisotropies: {gts}, ", "green"),
            colored(f"Diff times: {gtt}, ", "blue"),
            colored(f"Kernel colours: {gtc}", "red"),
        )

        gt_colours = gt_colours.sum(dim=0)
        if self._normalize_colours:
            gt_colours = self._normalize(gt_colours)
        return gt_colours

    @property
    def _errors(self):
        angles_error = (
            (self.angles - torch.deg2rad(self._gt_splats["angles"]))
            .pow(2)
            .sum()
            .pow(0.5)
        ).cpu()
        anisotropies_error = (
            (self.anisotropies - self._gt_splats["anisotropies"]).pow(2).sum().pow(0.5)
        )
        diff_times_error = (
            (self.diff_times - self._gt_splats["diff_times"]).pow(2).sum().pow(0.5)
        )
        kernel_colours_error = (
            (self.kernel_colours - self._gt_splats["kernel_colours"])
            .pow(2)
            .sum()
            .pow(0.5)
        ).cpu()
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

        axs[1, 1].plot(errors_lists["kernel_colours"], label="Kernel Colours Error")
        axs[1, 1].set_title("Kernel Colours Error")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Error")

        for ax in axs.flat:
            ax.legend()

        plt.tight_layout()
        plt.show()


class OptimiseVertColTextureWthFixedHeatKernels(OptimiseFixedHeatKernels):
    def __init__(
        self,
        verts,
        faces,
        fnorms,
        n_sources,
        lr_mult=1,
        k_eig=256,
        fpath=None,
        normalize_colours=False,
        device="cpu",
        vcols=None,
    ):
        super().__init__(
            verts,
            faces,
            fnorms,
            n_sources,
            lr_mult,
            k_eig,
            fpath,
            normalize_colours,
            device,
        )
        self._fpath = fpath
        assert vcols is not None
        self._vcols = torch.tensor(vcols, device=self.device)

    def _make_gt_colours(self):
        gt_colours = self._vcols
        return gt_colours[:, :3] / 255

    @property
    def _errors(self):
        return {
            "printables": None,
            "angles": torch.tensor(0),
            "anisotropies": torch.tensor(0),
            "diff_times": torch.tensor(0),
            "kernel_colours": torch.tensor(0),
        }


if __name__ == "__main__":
    import trimesh
    import numpy as np

    mesh_path = "../objects/spot_triangulated.ply"
    mesh = utils.load_mesh(mesh_path, show=False)

    try:
        # va = {"vert_col": mesh.visual.vertex_colors}
        # v, f, c = trimesh.remesh.subdivide(
        #     mesh.vertices, mesh.faces, vertex_attributes=va
        # )
        # mesh = trimesh.Trimesh(v, f, vertex_colors=c["vert_col"])
        vcols = mesh.visual.vertex_colors
    except AttributeError:
        v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(v, f)
        vcols = None

    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    fnorm = np.array(mesh.face_normals)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # torch.cuda.memory._record_memory_history()
    normalize_colours = True
    optimisation = OptimiseVertColTextureWthFixedHeatKernels(
        verts,
        faces,
        fnorm,
        n_sources=1000,
        k_eig=256,
        fpath=mesh_path,
        normalize_colours=normalize_colours,
        device="cuda",
        vcols=vcols,
    )

    v_colours, gt_colours, init_colours = optimisation.optimise(n_iter=5000)
    # torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

    v_colours = (v_colours * 255).to(dtype=torch.uint8)
    v_colours = v_colours.squeeze().detach().cpu().numpy()

    gt_colours = (gt_colours * 255).to(dtype=torch.uint8)
    gt_colours = gt_colours.squeeze().detach().cpu().numpy()

    init_colours = (init_colours * 255).to(dtype=torch.uint8)
    init_colours = init_colours.squeeze().detach().cpu().numpy()

    gt_mesh = mesh.copy()
    gt_mesh.visual = trimesh.visual.ColorVisuals(gt_mesh, vertex_colors=gt_colours)

    v_mesh = mesh.copy()
    v_mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=v_colours)
    v_scene = trimesh.Scene(
        [v_mesh, utils.big_trimesh_pcl(optimisation.kernel_centres)]
    )

    init_mesh = mesh.copy()
    init_mesh.visual = trimesh.visual.ColorVisuals(
        init_mesh, vertex_colors=init_colours
    )
