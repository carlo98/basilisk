#!/usr/bin/env python3
#
# ISC License
#
# Copyright (c) 2026, PIC4SeR & AVS Lab, Politecnico di Torino & Argotec S.R.L., University of Colorado Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

r"""Benchmark the faceted self-occlusion drag and SRP effectors.

This optional developer benchmark measures the run time of the polygon-clipping self-occlusion
algorithm of the faceted flat-plate effectors :ref:`facetDragSelfOcclusionPolygonClippingEffector`
and :ref:`facetSRPSelfOcclusionPolygonClippingEffector`, for increasing facet counts.

It is not a numerical validation test and it is not run by CI.

Running the Benchmark
---------------------

Run the benchmark from the Basilisk repository root directory after building the
Basilisk Python package:

.. code-block:: bash

   cd /path/to/basilisk
   .venv/bin/python benchmarks/dynamics/benchmark_facet_self_occlusion.py

Run only the SRP effector, with more trials and selected facet counts:

.. code-block:: bash

   .venv/bin/python benchmarks/dynamics/benchmark_facet_self_occlusion.py \
      --model srp --trials 7 --facet-counts 10 100 1000

Spin the body faster, so that the cached visibility of the hit case gets more out of date:

.. code-block:: bash

   .venv/bin/python benchmarks/dynamics/benchmark_facet_self_occlusion.py --spin-rate 5

Timing Study
------------

Each case builds a spacecraft carrying two concentric rings of slightly tilted facets. The
facets come in pairs: at each angle an outer facet and an inner facet, 0.25 m further in,
point outward along the same normal. On the side facing the source an outer facet partly
shadows its inner partner whenever their normal is within about 70 deg of the source
direction, so every facet count of 5 or more includes real self-occlusion. The
benchmark times full simulation steps, since the occlusion computation is only reachable
through ``computeForceTorque()``. Two visibility cache settings isolate the two costs of
interest:

``miss``
   ``setDirectionCacheTolerance(0.0)`` recomputes the facet visibility at every force
   evaluation, so the timing is dominated by the occlusion algorithm.

``hit``
   ``setDirectionCacheTolerance(1.0)`` reuses the visibility computed at the first step,
   so the timing measures the per-facet force loop alone.

The table reports, for each number of facets, the median and minimum wall-clock
milliseconds per step over the trials.

Cache Error
-----------

The body spins about the ring axis at ``--spin-rate`` (default 0.5 rad/s), so the source
direction in the body frame sweeps during the run. The hit case keeps using the visibility of
the first step, while the miss case recomputes it, so the two last force evaluations differ by
the error of the cache after the body has rotated by the spin rate times the simulated time.
The table reports it as

``force err``
   :math:`|\mathbf{F}_\text{hit} - \mathbf{F}_\text{miss}| / |\mathbf{F}_\text{miss}|`

``torque err``
   :math:`|\mathbf{L}_\text{hit} - \mathbf{L}_\text{miss}| / (|\mathbf{F}_\text{miss}|\, R)`,
   normalized by the force times the ring radius :math:`R` because the ring torque can nearly
   cancel by symmetry.

With ``--spin-rate 0`` only the small rotation caused by the effector's own torque changes the
source direction, so both errors become negligible. With a nonzero spin rate the shadowed parts of the inner facets change during the run, so the errors
are nonzero at every facet count.
"""

import argparse
import statistics
import time

import numpy as np

from Basilisk.architecture import astroConstants
from Basilisk.architecture import messaging
from Basilisk.simulation import exponentialAtmosphere
from Basilisk.simulation import facetDragSelfOcclusionPolygonClippingEffector
from Basilisk.simulation import facetSRPSelfOcclusionPolygonClippingEffector
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros


MODULES = {
    "drag": facetDragSelfOcclusionPolygonClippingEffector.FacetDragSelfOcclusionPolygonClippingEffector,
    "srp": facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector,
}

DRAG_COEFF = 2.2  # [-]
DIFFUSE_COEFF = 0.5  # [-]
SPECULAR_COEFF = 0.2  # [-]
TIMING_TIME_STEP = 1e-3  # [s]
CACHE_MISS_TOLERANCE = 0.0  # [rad]
CACHE_HIT_TOLERANCE = 1.0  # [rad]
RING_RADIUS = 5.0  # [m] nominal radius of the outer benchmark facet ring
RING_GAP = 0.25  # [m] radial distance between an outer facet and its inner partner


def _positive_int(value):
    """Return ``value`` as a positive integer for ``argparse``."""

    parsedValue = int(value)
    if parsedValue < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsedValue


def _nonnegative_int(value):
    """Return ``value`` as a nonnegative integer for ``argparse``."""

    parsedValue = int(value)
    if parsedValue < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsedValue


def _nonnegative_float(value):
    """Return ``value`` as a nonnegative floating-point value for ``argparse``."""

    parsedValue = float(value)
    if parsedValue < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsedValue


def _parse_args():
    """Parse the benchmark command-line options."""

    parser = argparse.ArgumentParser(
        description="Benchmark the faceted self-occlusion drag and SRP effectors."
    )
    parser.add_argument("--model", choices=["drag", "srp", "all"], default="all",
                        help="flux model to benchmark (default: all)")
    parser.add_argument("--facet-counts", type=_positive_int, nargs="+",
                        default=[10, 25, 50, 100, 250, 500],
                        help="numbers of facets to benchmark")
    parser.add_argument("--steps", type=_positive_int, default=20,
                        help="timed integration steps per trial (default: 20)")
    parser.add_argument("--trials", type=_positive_int, default=3,
                        help="trials per case (default: 3)")
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=1,
                        help="untimed steps before each trial (default: 1)")
    parser.add_argument("--spin-rate", type=_nonnegative_float, default=0.5,
                        help="body spin rate about the ring axis in rad/s (default: 0.5)")
    config = parser.parse_args()
    config.models = ["drag", "srp"] if config.model == "all" else [config.model]
    return config


def _build_ring_geometry(numberOfFacets):
    """Return areas, normals and locations of ``numberOfFacets`` slightly tilted facets on two
    concentric rings around the z axis.

    The facets come in pairs placed at ``ceil(numberOfFacets / 2)`` equally spaced angles: an
    outer facet near ``RING_RADIUS`` and an inner facet ``RING_GAP`` further in, with the same
    normal and area. Seen from the source, the pair is offset sideways by at most ``RING_GAP``,
    less than the facet width, so the outer facet partly shadows its inner partner whenever
    their normal is within about 70 deg of the source direction. With 3 or more angles at least
    one pair always is. For an odd count the last angle only has its outer facet. The random
    perturbations are seeded with the number of facets, so each case is reproducible.
    """

    rng = np.random.default_rng(seed=numberOfFacets)
    numberOfAngles = (numberOfFacets + 1) // 2
    areas, normals, locations = [], [], []
    for index in range(numberOfAngles):
        theta = 2.0 * np.pi * index / numberOfAngles  # [rad]
        normal = np.array([np.cos(theta), np.sin(theta), 0.1 * np.sin(3.0 * theta)])  # [-]
        normal = normal / np.linalg.norm(normal)
        radialDirection = np.array([np.cos(theta), np.sin(theta), 0.0])  # [-]
        outerRadius = RING_RADIUS + 0.5 * rng.standard_normal()  # [m]
        area = 0.5 + 0.1 * rng.standard_normal() ** 2  # [m^2]
        for radius in (outerRadius, outerRadius - RING_GAP):  # [m]
            if len(areas) == numberOfFacets:
                break
            areas.append(area)
            normals.append(normal)
            locations.append(radius * radialDirection)
    return areas, normals, locations


def _add_facet(effector, model, area, nHat_B, r_CopB_B):
    """Add one facet with the coefficients of the given flux model."""

    if model == "drag":
        effector.addFacet(area, DRAG_COEFF, nHat_B, r_CopB_B)
    else:
        effector.addFacet(area, DIFFUSE_COEFF, SPECULAR_COEFF, nHat_B, r_CopB_B)


def _create_simulation(model, effector, timeStepSec, velocity_N, spinRate):
    """Create a simulation with a spacecraft carrying ``effector`` and the flux source of ``model``.

    The body spins at ``spinRate`` [rad/s] about its z axis, the ring axis.

    Drag uses an exponential atmosphere around a low Earth orbit. SRP uses a static Sun message at
    1 AU along +y from the spacecraft.
    """

    scSim = SimulationBaseClass.SimBaseClass()
    process = scSim.CreateNewProcess("process")
    process.addTask(scSim.CreateNewTask("task", macros.sec2nano(timeStepSec)))

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"
    scObject.hub.v_CN_NInit = velocity_N
    scObject.hub.sigma_BNInit = np.array([0.0, 0.0, 0.0])  # [-]
    scObject.hub.omega_BN_BInit = np.array([0.0, 0.0, spinRate])  # [rad/s]
    scSim.AddModelToTask("task", scObject)

    if model == "drag":
        scObject.hub.r_CN_NInit = np.array([6571e3, 0.0, 0.0])  # [m]
        atmosphere = exponentialAtmosphere.ExponentialAtmosphere()
        atmosphere.ModelTag = "atmosphere"
        atmosphere.baseDensity = 1.217  # [kg/m^3]
        atmosphere.scaleHeight = 8500.0  # [m]
        atmosphere.planetRadius = 6371e3  # [m]
        atmosphere.addSpacecraftToModel(scObject.scStateOutMsg)
        scSim.AddModelToTask("task", atmosphere)
        effector.atmoDensInMsg.subscribeTo(atmosphere.envOutMsgs[0])
    else:
        r_BN_N = np.array([0.05 * astroConstants.AU2M, 0.0, 0.0])  # [m]
        scObject.hub.r_CN_NInit = r_BN_N
        sunPayload = messaging.SpicePlanetStateMsgPayload()
        sunPayload.PositionVector = [r_BN_N[0], astroConstants.AU2M, 0.0]  # [m]
        scSim.sunMsg = messaging.SpicePlanetStateMsg().write(sunPayload)
        effector.sunInMsg.subscribeTo(scSim.sunMsg)

    scObject.addDynamicEffector(effector)
    scSim.AddModelToTask("task", effector)
    return scSim


def _time_case(config, model, numberOfFacets, cacheTolerance):
    """Return the wall-clock seconds per step of every trial of one timing case, and the force and
    torque of the last force evaluation of the last trial."""

    elapsedPerStep = []
    for _ in range(config.trials):
        effector = MODULES[model]()
        effector.setLogOverlapFallbackWarnings(False)  # keep the benchmark output readable
        effector.ModelTag = f"{model}PolygonClipping"
        effector.setDirectionCacheTolerance(cacheTolerance)
        for area, nHat_B, r_CopB_B in zip(*_build_ring_geometry(numberOfFacets)):
            _add_facet(effector, model, area, nHat_B, r_CopB_B)
        velocity_N = np.array([1500.0, 7000.0, 500.0])  # [m/s], oblique to the ring
        scSim = _create_simulation(model, effector, TIMING_TIME_STEP, velocity_N, config.spin_rate)
        scSim.InitializeSimulation()

        # Untimed steps absorb one-time setup costs, such as the first visibility computation.
        warmupSteps = config.warmup_steps
        if warmupSteps > 0:
            scSim.ConfigureStopTime(macros.sec2nano(warmupSteps * TIMING_TIME_STEP))
            scSim.ExecuteSimulation()
        scSim.ConfigureStopTime(macros.sec2nano((warmupSteps + config.steps) * TIMING_TIME_STEP))
        start = time.perf_counter()
        scSim.ExecuteSimulation()
        elapsedPerStep.append((time.perf_counter() - start) / config.steps)
    force_B = np.array(effector.forceExternal_B).flatten()  # [N]
    torque_B = np.array(effector.torqueExternalPntB_B).flatten()  # [N*m]
    return elapsedPerStep, force_B, torque_B


def _run_timing(config, model):
    """Print the timing table of one flux model."""

    print()
    print(f"Timing study: {model}")
    columns = (("miss", CACHE_MISS_TOLERANCE), ("hit", CACHE_HIT_TOLERANCE))
    header = f"{'facets':>7}"
    for cacheLabel, _ in columns:
        header += f"  {cacheLabel + ' median [ms]':>18} {cacheLabel + ' min [ms]':>18}"
    header += f"  {'force err [-]':>14} {'torque err [-]':>14}"
    print(header)
    for numberOfFacets in config.facet_counts:
        row = f"{numberOfFacets:>7}"
        results = {}
        for cacheLabel, tolerance in columns:
            timesSec, force_B, torque_B = _time_case(config, model, numberOfFacets, tolerance)
            results[cacheLabel] = (force_B, torque_B)
            row += f"  {1e3 * statistics.median(timesSec):>18.4f} {1e3 * min(timesSec):>18.4f}"
        forceMiss, torqueMiss = results["miss"]
        forceHit, torqueHit = results["hit"]
        forceScale = np.linalg.norm(forceMiss)  # [N]
        if forceScale > 0.0:
            forceErr = np.linalg.norm(forceHit - forceMiss) / forceScale
            torqueErr = np.linalg.norm(torqueHit - torqueMiss) / (forceScale * RING_RADIUS)
        else:
            forceErr = torqueErr = float("nan")
        row += f"  {forceErr:>14.3e} {torqueErr:>14.3e}"
        print(row)


def _print_header(config):
    """Print the benchmark configuration."""

    print("Facet self-occlusion benchmark")
    print(f"Models: {', '.join(config.models)}")
    print("Algorithm: polygon clipping")
    print(f"Facet counts: {config.facet_counts}")
    print(f"Steps per trial: {config.steps}")
    print(f"Trials: {config.trials}")
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Spin rate: {config.spin_rate} rad/s")
    print("miss: visibility recomputed every step; hit: visibility of the first step reused")


def main():
    """Run the facet self-occlusion benchmark."""

    config = _parse_args()
    _print_header(config)
    for model in config.models:
        _run_timing(config, model)


if __name__ == "__main__":
    main()
