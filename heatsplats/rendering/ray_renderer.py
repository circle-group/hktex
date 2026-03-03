from __future__ import annotations as __annotations__

import drjit as dr
import mitsuba as mi
import torch

from heatsplats.utils.typing import *

# Based on https://github.com/rubenwiersma/svbrdf_uncertainty/blob/main/svbrdf_uncertainty/util/render_ray.py


class _RenderRayOp(dr.CustomOp):
    """
    This class is an implementation detail of the render() function. It
    realizes a CustomOp that provides evaluation, and forward/reverse-mode
    differentiation callbacks that will be invoked as needed (e.g. when a
    rendering operation is encountered by an AD graph traversal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.variant = mi.variant()

    def eval(self, scene, ray, sensor, _, params, integrator, seed, spp):
        self.scene = scene
        self.ray = ray
        self.sensor = sensor
        # The argument `_` is a `dict` of the parameters that is detached,
        # whereas `params` is a `SceneParameters` object that still contains
        # a reference to the attached paraamters
        self.params = params
        self.integrator = integrator
        self.seed = seed
        self.spp = spp

        with dr.suspend_grad():
            res = self.integrator.render(
                scene=self.scene,
                ray=self.ray,
                sensor=sensor,
                seed=seed[0],
                spp=spp[0],
                develop=True,
                evaluate=False,
            )
            # After rendering an image, the sampler state is dependent on the
            # rendering loop. When a frozen function is recorded, the sampler
            # might be evaluated, which causes parts of the rendering loop to
            # be re-evaluated. To prevent this overhead, we reset the state
            # of the sampler, by re-seeding it.
            sensor.sampler().seed(0, 1)
            return res

    def forward(self):
        self.set_grad_out(
            self.integrator.render_forward(
                self.scene,
                self.params,
                self.ray,
                self.sensor,
                self.seed[1],
                self.spp[1],
            )
        )

    def backward(self):
        self.integrator.render_backward(
            self.scene,
            self.params,
            self.grad_out(),
            self.ray,
            self.sensor,
            self.seed[1],
            self.spp[1],
        )

    def name(self):
        return "RenderOp"


def render_ray(
    scene: mi.Scene,
    params: Any = None,
    ray: mi.RayDifferential3f = None,
    sensor: Union[int, mi.Sensor] = 0,
    integrator: mi.Integrator = None,
    seed: int = 0,
    seed_grad: int = 0,
    spp: int = 0,
    spp_grad: int = 0,
) -> mi.TensorXf:
    """
    This function mimics the Mitsuba render function, but allows to specify
    custom rays to be used for rendering.
    """

    if params is not None and not isinstance(params, mi.SceneParameters):
        raise Exception(
            "The `params` argument should be an instance of `mi.SceneParameters`!"
        )

    dict_params = dict()
    if params is not None:
        dict_params = dict(params)  # Turn SceneParameters into a valid PyTree

    assert isinstance(scene, mi.Scene)

    if integrator is None:
        integrator = scene.integrator()

    if integrator is None:
        raise Exception(
            "No integrator specified! Add an integrator in the scene "
            "description or provide an integrator directly as argument."
        )

    if isinstance(sensor, int):
        if len(scene.sensors()) == 0:
            raise Exception(
                "No sensor specified! Add a sensor in the scene "
                "description or provide a sensor directly as argument."
            )
        sensor = scene.sensors()[sensor]

    assert isinstance(integrator, mi.Integrator)
    assert isinstance(sensor, mi.Sensor)

    if spp_grad == 0:
        spp_grad = spp

    if seed_grad == 0:
        # Compute a seed that de-correlates the primal and differential phase
        seed_grad = mi.sample_tea_32(seed, 1)[0]
    elif seed_grad == seed:
        raise Exception(
            "The primal and differential seed should be different "
            "to ensure unbiased gradient computation!"
        )

    if "scalar" in mi.variant():
        return integrator.render(
            scene=scene,
            ray=ray,
            sensor=sensor,
            seed=seed,
            spp=spp,
            develop=True,
            evaluate=False,
        )

    return dr.custom(
        _RenderRayOp,
        scene,
        ray,
        sensor,
        dict_params,
        params,
        integrator,
        (seed, seed_grad),
        (spp, spp_grad),
    )


def sample_rays_multiple_sensors(integrator, scene, sensors, seed):
    """Sample a ray per pixel, per sensor and concatenate all the rays.
    Outputs the origin and direction of the rays, as well as the wavelengths.
    """
    with dr.suspend_grad():
        o, d = [], []
        wavelengths = None
        for i, sensor in enumerate(sensors):
            # We only use one primary ray per pixel and sample it multiple times
            sampler, spp = integrator.prepare(sensor=sensor, seed=seed, spp=1)
            ray, _, _ = integrator.sample_rays(scene, sensor, sampler)
            if wavelengths is None:
                wavelengths = ray.wavelengths
            o.append(ray.o.torch().cpu())
            d.append(ray.d.torch().cpu())
        o = torch.cat(o, dim=0)
        d = torch.cat(d, dim=0)
        return o, d, wavelengths


def get_film_size(sensor: mi.Sensor):
    film = sensor.film()
    film_size = film.crop_size()
    rfilter = film.rfilter()
    border_size = rfilter.border_size()

    if film.sample_border():
        film_size += 2 * border_size
    return dr.prod(film_size)


@torch.no_grad
def sample_intersecting_rays_multiple_sensors(
    integrator: mi.ad.integrators.common.ADIntegrator,
    scene: mi.Scene,
    sensors: List[mi.Sensor],
    seed: int,
    gen=None,
    safety_max_iterations: int = 32,
):
    """Sample a ray per pixel, per sensor and concatenate all the rays.
    Outputs the origin and direction of the rays, as well as the wavelengths.
    Makes sure that rays always intersect with the scene,
    resamples until enough rays are generated and then picks uniformly until film_size
    rays are generated per sensor.
    Returns: ray origins, ray directions, wavelengths,
             ray screen space positions, ray sensor idx,
             first-hit face ids, first-hit points,
             offset for the seed
    """
    with dr.suspend_grad():
        o, d = [], []
        pos, sensor_idx = [], []
        hit_face_ids, hit_points = [], []
        wavelengths = None

        seed_offset = 0

        for s_idx, sensor in enumerate(sensors):
            film_size = get_film_size(sensor)
            collected, iters = 0, 0
            o_s, d_s, pos_s = [], [], []
            fids_s, hp_s = [], []

            while collected < film_size:
                if iters > safety_max_iterations:
                    raise RuntimeError(
                        "Fixed max number of iterations in `sample_intersecting_rays_multiple_sensors`, "
                        "you probably have a camera that doesn't view the object"
                    )
                iters += 1

                # We only use one primary ray per pixel and sample it multiple times
                sampler, _ = integrator.prepare(
                    sensor=sensor, seed=seed + seed_offset, spp=1
                )
                seed_offset += 1

                ray, weight, position = integrator.sample_rays(scene, sensor, sampler)
                if wavelengths is None:
                    # We are not using _spectral variants, we should be able to ignore processing this
                    wavelengths = ray.wavelengths
                # Note: Weights are ignored for now, we assume we use
                # standard cameras/sensors that return 1 weight

                pi: mi.PreliminaryIntersection3f = scene.ray_intersect_preliminary(
                    ray, coherent=True
                )
                hit_mask = pi.is_valid()
                if not dr.any(hit_mask):
                    continue

                idx = dr.compress(hit_mask)
                ro = dr.gather(type(ray.o), ray.o, idx)
                rd = dr.gather(type(ray.d), ray.d, idx)
                rpos = dr.gather(type(position), position, idx)

                # First-hit data
                rfids = dr.gather(type(pi.prim_index), pi.prim_index, idx)
                rt = dr.gather(type(pi.t), pi.t, idx)
                rhit = ro + rd * rt

                o_s.append(ro.torch().t().cpu())
                d_s.append(rd.torch().t().cpu())
                pos_s.append(rpos.torch().t().cpu())
                fids_s.append(rfids.torch().cpu().to(torch.int64))
                hp_s.append(rhit.torch().t().cpu())

                collected += int(dr.width(idx))

            o_s = torch.cat(o_s, dim=0)
            d_s = torch.cat(d_s, dim=0)
            pos_s = torch.cat(pos_s, dim=0)
            fids_s = torch.cat(fids_s, dim=0)
            hp_s = torch.cat(hp_s, dim=0)

            # Avoid bias from only picking first film_size if the sensor generates rays in a fixed screen space order
            perm = torch.randperm(o_s.shape[0], generator=gen)[:film_size]

            o.append(o_s[perm])
            d.append(d_s[perm])
            pos.append(pos_s[perm])
            hit_face_ids.append(fids_s[perm])
            hit_points.append(hp_s[perm])

            sid_s = torch.full((film_size,), s_idx, dtype=torch.int64)
            sensor_idx.append(sid_s)

        o = torch.cat(o, dim=0)
        d = torch.cat(d, dim=0)
        pos = torch.cat(pos, dim=0)
        sensor_idx = torch.cat(sensor_idx, dim=0)
        hit_face_ids = torch.cat(hit_face_ids, dim=0)
        hit_points = torch.cat(hit_points, dim=0)

        return o, d, wavelengths, pos, sensor_idx, hit_face_ids, hit_points, seed_offset


def integrate_ray_samples(L, spp):
    """Integrate #spp samples for one pixel by taking the average."""
    n_out = dr.width(L) // spp
    scatter_idx = dr.repeat(dr.arange(mi.UInt, n_out), spp)
    L_integrated = dr.zeros(type(L), n_out)
    dr.scatter_reduce(dr.ReduceOp.Add, L_integrated, L, scatter_idx)
    return L_integrated * (1 / spp)
