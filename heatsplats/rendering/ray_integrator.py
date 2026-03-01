from __future__ import annotations as __annotations__

import gc

import drjit as dr
import mitsuba as mi

from heatsplats.utils.typing import *

# Based on https://github.com/rubenwiersma/svbrdf_uncertainty/blob/main/svbrdf_uncertainty/plugins/integrators/custom_ray.py


class RBRayIntegrator(mi.ad.integrators.common.RBIntegrator):
    """Custom ray integrator for Mitsuba 3.
    This class allows to pass a custom batch of rays to the integrator,
    which we use to give a random set of rays sampled from all sensors per iteration,
    rather than the rays corresponding to a single sensor."""

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.integrator: mi.ad.integrators.common.RBIntegrator = props.get("integrator")

    def render(
        self,
        scene: mi.Scene,
        ray: mi.RayDifferential3f = None,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
        develop: bool = True,
        evaluate: bool = True,
    ) -> mi.TensorXf:

        if not develop:
            raise Exception(
                "develop=True must be specified when " "invoking AD integrators"
            )

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.integrator.aov_names(),
                wavefront_size=ray.o.shape[-1] if ray is not None else None,
            )

            # Generate a set of rays starting at the sensor
            if ray is None:
                ray, _, _ = self.integrator.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, aovs, _ = self.integrator.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True),
            )

            # Explicitly delete any remaining unused variables
            del sampler, valid, ray
            gc.collect()

        return L

    def render_forward(
        self,
        scene: mi.Scene,
        params: Any,
        ray: mi.RayDifferential3f = None,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
    ) -> mi.TensorXf:
        raise NotImplementedError("Forward mode not implemented yet")

    def render_backward(
        self,
        scene: mi.Scene,
        params: Any,
        grad_in: mi.TensorXf,
        ray: mi.RayDifferential3f = None,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
    ) -> None:
        """
        Customized version of Mitsuba's PRB backward pass that
        supports passing rays to be rendered.
        See documentation of Mitsuba for more details.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor,
                seed,
                spp,
                self.integrator.aov_names(),
                wavefront_size=ray.o.shape[-1] if ray is not None else None,
            )

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            if ray is None:
                ray, _, _ = self.integrator.sample_rays(scene, sensor, sampler)

            δL = grad_in
            δaovs = None

            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, aovs, state_out = self.integrator.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True),
            )

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, aovs_2, state_out_2 = self.integrator.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=δL,
                δaovs=δaovs,
                state_in=state_out,
                active=mi.Bool(True),
            )

            # We don't need any of the outputs here
            del L_2, valid_2, aovs_2, state_out, state_out_2, δL, δaovs, sampler, ray

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

    def prepare(
        self,
        sensor: mi.Sensor,
        seed: mi.UInt32 = 0,
        spp: int = 0,
        aovs: list = [],
        wavefront_size: Optional[int] = None,
    ):
        """
        Given a sensor and a desired number of samples per pixel, this function
        computes the necessary number of Monte Carlo samples and then suitably
        seeds the sampler underlying the sensor.

        Returns the created sampler and the final number of samples per pixel
        (which may differ from the requested amount depending on the type of
        ``Sampler`` being used)

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor to render the scene from a different viewpoint.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator during the primal rendering step. It is crucial that you
            specify different seeds (e.g., an increasing sequence) if subsequent
            calls should produce statistically independent images (e.g. to
            de-correlate gradient-based optimization steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            primal rendering step. The value provided within the original scene
            specification takes precedence if ``spp=0``.
        """

        film = sensor.film()
        original_sampler = sensor.sampler()
        sampler = original_sampler.clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        if wavefront_size is None:
            film_size = film.crop_size()

            if film.sample_border():
                film_size += 2 * film.rfilter().border_size()

            wavefront_size = dr.prod(film_size) * spp

            if wavefront_size > 2**32:
                raise Exception(
                    "The total number of Monte Carlo samples required by this "
                    "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                    "fewer samples per pixel or render using multiple passes."
                    % wavefront_size
                )

        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)

        return sampler, spp


mi.register_integrator("rb_ray", lambda props: RBRayIntegrator(props))
