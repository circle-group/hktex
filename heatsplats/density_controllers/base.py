import math
import torch

from abc import abstractmethod
from dataclasses import dataclass

from heatsplats.utils.typing import *
from heatsplats.utils import BaseObject, rotate_on_plane
from heatsplats.modules import (
    Mesh,
    HeatKernelTexture,
    GeodesicOpt,
    GeodesicTracer,
    EigenAlboInterpolation,
    KernelInfo,
)


class BaseDensityController(BaseObject):
    """Base class defining the interface for the controllers responsible for
    the adaptive densification strategy of the kernels during training.
    """

    @dataclass
    class Config(BaseObject.Config):
        start_iter: int = 0
        stop_iter: int | None = None

    cfg: Config

    def configure(
        self,
        mesh: Mesh,
        model: HeatKernelTexture,
        optimizers: Dict[str, torch.optim.Optimizer],
        *args,
        **kwargs,
    ):
        super().configure(*args, **kwargs)
        self._mesh = mesh
        self._model = model
        self._optimizers = optimizers

        self._params = dict(model.named_parameters())
        self._buffers = dict(model.named_buffers())
        self._state: Dict[str, Tensor] = {}

        self._post_configure()
        self.check_sanity()

    @abstractmethod
    def _post_configure(self):
        """Hook for additional configuration in subclasses."""
        pass

    @abstractmethod
    def pre_backward_step(self, *args, **kwargs):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    @abstractmethod
    def post_backward_step(self, *args, **kwargs):
        """Callback function to be executed after the `loss.backward()` call."""
        pass

    def check_sanity(self) -> None:
        """Sanity check for the parameters and optimizers."""
        self._trainable_params_names = set(
            [name for name, param in self._params.items() if param.requires_grad]
        )

        # Collect all parameter names covered by optimizers
        self._name2opt_dict = {}
        optimizer_params = set()
        for i, optimizer in enumerate(self._optimizers):
            for group in optimizer.param_groups:
                try:
                    name = group["name"].split(".")[-1]
                except KeyError:
                    name = group.name.split(".")[-1]
                except AttributeError:
                    name = id(group)

                optimizer_params.add(name)
                self._name2opt_dict[name] = i

        # Check that all trainable parameters are covered
        missing = self._trainable_params_names - optimizer_params
        assert not missing, (
            "Some trainable parameters are not covered by any optimizer: " f"{missing}"
        )

    def refresh_state(self):
        self._params = dict(self._model.named_parameters())
        self._buffers = dict(self._model.named_buffers())

    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        param_fn: Callable[[str, Tensor], Tensor],
        optimizer_fn: Callable[[str, Tensor], Tensor],
        buffer_fn: Optional[Callable[[str, Tensor], Tensor]] = None,
        names: Union[List[str], None] = None,
        buffer_names: Optional[List[str]] = None,
    ):
        """Update the parameters and the state in the optimizers with defined functions.

        Args:
            param_fn: A function that takes the name of the parameter and the parameter
                itself, and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the
                state value, and returns the new state value.
            buffer_fn: A function that takes the name of the buffer and the buffer
                itself, and returns the new buffer. If None, no buffers are updated.
                Default: None.
            names: A list of key names to update. If None, update all. Default: None.
            buffer_names: A list of key names to update buffers. If None, update all.
                Default: None.
        """

        if buffer_fn is not None:
            if buffer_names is None:
                buffer_names = list(self._buffers.keys())
                buffer_names.remove("_dummy")
            for name in buffer_names:
                buffer = self._buffers[name]
                new_buffer = buffer_fn(name, buffer)
                self._buffers[name] = new_buffer
                setattr(self._model, name, new_buffer)

        if names is None:
            names = list(self._params.keys())

        for name in names:
            param = self._params[name]
            new_param = param_fn(name, param)
            if hasattr(param, "bary_coords"):
                setattr(new_param, "bary_coords", getattr(param, "bary_coords"))
            self._params[name] = new_param
            setattr(self._model, name, new_param)

            if name in self._name2opt_dict:
                optimizer = self._optimizers[self._name2opt_dict[name]]

                for group in optimizer.param_groups:
                    for idx, group_param in enumerate(group["params"]):
                        if group_param is param:
                            # Update optimizer state for this parameter
                            param_state = optimizer.state[param]
                            del optimizer.state[param]
                            for key in param_state.keys():
                                if key != "step":
                                    v = param_state[key]
                                    param_state[key] = optimizer_fn(key, v)
                            # Replace the parameter in the param group
                            group["params"][idx] = new_param
                            optimizer.state[new_param] = param_state

                            if (
                                isinstance(optimizer, GeodesicOpt)
                                and "face_ids" in group
                            ):
                                fids = self._buffers["_kernel_face_ids"]
                                group["face_ids"][idx] = fids
                                setattr(self._model, "_kernel_face_ids", fids)

                            break  # Found and updated, no need to check further

    @torch.no_grad()
    def _remove_kernels(self, mask: Tensor):
        """Inplace remove the Heat Kernels with the given mask."""
        sel = torch.where(~mask)[0]

        def param_fn(name: str, p: Tensor) -> Tensor:
            return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

        def buffer_fn(name: str, b: Tensor) -> Tensor:
            return torch.nn.Buffer(b[sel.to(b.device)], persistent=True)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            return v[sel]

        self._update_param_with_optimizer(param_fn, optimizer_fn, buffer_fn)

        setattr(self._model, "N_sources", self._model.N_sources - mask.sum().item())

        for k, v in self._state.items():
            if isinstance(v, torch.Tensor):
                self._state[k] = v[sel]

    @torch.no_grad()
    def clone(self, mask: Tensor):
        """Inplace duplicate the Gaussian with the given mask."""
        device = mask.device
        sel = torch.where(mask)[0]

        # Halve the opacity of the original parent kernels IN-PLACE so that when cloned
        # they have both the correct opacity.
        self._params["_opacities"].data[mask] *= 0.5

        def param_fn(name: str, p: Tensor) -> Tensor:
            return torch.nn.Parameter(
                torch.cat([p, p[sel]]), requires_grad=p.requires_grad
            )

        def buffer_fn(name: str, b: Tensor) -> Tensor:
            return torch.nn.Buffer(torch.cat([b, b[sel.to(b.device)]]), persistent=True)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

        self._update_param_with_optimizer(param_fn, optimizer_fn, buffer_fn)

        setattr(self._model, "N_sources", self._model.N_sources + len(sel))

        for k, v in self._state.items():
            if isinstance(v, torch.Tensor):
                self._state[k] = torch.cat((v, v[sel]))

    @torch.no_grad()
    def split(
        self,
        mask: Tensor,
        eigalbo_interp: EigenAlboInterpolation,
        kernel_info: KernelInfo,
        tracer: GeodesicTracer,
        base_radius: float,
    ):
        device = mask.device
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        kernel_barys = kernel_info["barycentric_coords"][sel]
        kernel_vert_idx = kernel_info["vert_idx"][sel]
        kernel_faces = self._model._kernel_face_ids[sel]
        kernel_thresholds = self._model.thresholds[sel]
        kernel_locations = self._model.kernel_locations[sel]
        kernel_normals = self._mesh.fnorms[kernel_faces]

        kernels_direction = eigalbo_interp.barycentric_local_directions_gaussians(
            barycentric_coords=kernel_barys,
            vert_idx=kernel_vert_idx,
        )
        principal_axis_directions = rotate_on_plane(
            kernels_direction, kernel_normals, self._model.angles[sel] + math.pi / 2
        )

        displacements = (1 - self._model.thresholds[sel]) * base_radius
        principal_axis_vectors = displacements.unsqueeze(-1) * principal_axis_directions

        displaced_pos, displaced_faces = tracer.trace(
            kernel_locations.repeat(2, 1),
            kernel_faces.repeat(2),
            torch.cat([principal_axis_vectors, -principal_axis_vectors], dim=0),
            bary_coords=kernel_barys.repeat(2, 1),
        )
        displaced_thresholds = 1.0 - ((1.0 - kernel_thresholds) / 1.6)

        def param_fn(name: str, p: Tensor) -> Tensor:
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "_kernel_locations":
                p_split = displaced_pos.detach()
            elif name == "_thresholds":
                p_split = displaced_thresholds.repeat(2)
            elif name == "_opacities":
                p_split = 0.6 * p[sel].repeat(repeats)
            else:
                p_split = p[sel].repeat(repeats)
            p_new = torch.cat([p[rest], p_split])
            p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
            return p_new

        def buffer_fn(name: str, p: Tensor) -> Tensor:
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "_kernel_face_ids":
                p_split = displaced_faces
            else:
                p_split = p[sel].repeat(repeats)
            p_new = torch.cat([p[rest], p_split])
            p_new = torch.nn.Buffer(p_new, persistent=True)
            return p_new

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
            return torch.cat([v[rest], v_split])

        self._update_param_with_optimizer(param_fn, optimizer_fn, buffer_fn)

        setattr(self._model, "N_sources", self._model.N_sources + len(sel))

        for k, v in self._state.items():
            if isinstance(v, torch.Tensor):
                repeats = [2] + [1] * (v.dim() - 1)
                v_new = v[sel].repeat(repeats)
                self._state[k] = torch.cat((v[rest], v_new))
