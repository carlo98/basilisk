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
# Purpose:  Cross-check both SRP self-occlusion effectors against facetSRPDynamicEffector on a
#           convex body, where no self-occlusion can occur, and check that hub and
#           state-effector branch attachment agree.
#

import numpy as np

from Basilisk.architecture import astroConstants
import pytest

from Basilisk.architecture import messaging
from Basilisk.simulation import facetSRPDynamicEffector
from Basilisk.simulation import facetSRPSelfOcclusionPolygonClippingEffector
from Basilisk.simulation import spacecraft
from Basilisk.simulation import spinningBodyOneDOFStateEffector
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros

SIMULATION_TIME_STEP = 1e-4  # [s]

#: (constructor, relative tolerance) of every self-occlusion module compared with the legacy module.
SELF_OCCLUSION_MODULES = {
    "polygonClipping": (facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector,
                        1e-10),
}

# Unit cube faces: normals and face centers in B. A cube is convex, so no face can shadow another.
CUBE_NORMALS_B = [np.array(n, dtype=float) for n in
                  ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]  # [-]
CUBE_LOCATIONS_B = [0.5 * n for n in CUBE_NORMALS_B]  # [m]
CUBE_AREA = 1.0  # [m^2]
CUBE_LEVER_ARM = 0.5 * np.sqrt(3.0)  # [m] largest facet point distance from B, scales the torque tolerance
DIFFUSE_COEFF = 0.3  # [-]
SPECULAR_COEFF = 0.4  # [-]
SIGMA_BN = np.array([0.1, 0.2, -0.3])  # [-] generic attitude with several faces toward the source

# facetSRPDynamicEffector hard-codes its own solar flux and astronomical unit, while the self-occlusion
# effectors use SOLAR_FLUX_EARTH and AU2M from astroConstants. Its pressure is rescaled by this ratio
# before comparing, so that only the geometry and the force law are cross-checked.
LEGACY_SOLAR_FLUX = 1368.0  # [W/m^2] solar flux at 1 AU in facetSRPDynamicEffector.cpp
LEGACY_AU = 149597870700.0  # [m] astronomical unit in facetSRPDynamicEffector.cpp
LEGACY_PRESSURE_SCALE = (astroConstants.SOLAR_FLUX_EARTH / LEGACY_SOLAR_FLUX) \
    * (astroConstants.AU2M / LEGACY_AU) ** 2  # [-]


def createSimulation():
    """Create a simulation with one task and a spacecraft hub."""
    scSim = SimulationBaseClass.SimBaseClass()
    process = scSim.CreateNewProcess("process")
    process.addTask(scSim.CreateNewTask("task", macros.sec2nano(SIMULATION_TIME_STEP)))
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"
    scObject.hub.mHub = 100.0  # [kg]
    scObject.hub.IHubPntBc_B = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]  # [kg m^2]
    scObject.hub.sigma_BNInit = SIGMA_BN
    scSim.AddModelToTask("task", scObject)
    return scSim, scObject


def runAndLog(scSim, effectors):
    """Run two integration steps and return the last logged force and torque of every effector."""
    loggers = []
    for effector in effectors:
        logger = effector.logger(["forceExternal_B", "torqueExternalPntB_B"])
        scSim.AddModelToTask("task", logger)
        loggers.append(logger)
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(2.0 * SIMULATION_TIME_STEP))
    scSim.ExecuteSimulation()
    return [(np.array(logger.forceExternal_B[-1]), np.array(logger.torqueExternalPntB_B[-1])) for logger in loggers]


def assertVectorsClose(actual, expected, rtol, refScale=None):
    """Assert two vectors agree to a tolerance relative to ``refScale``, by default the expected
    magnitude. Pass ``refScale`` when the expected value is near zero by cancellation, such as
    the torque of a centered convex body, so the tolerance reflects the size of the terms."""
    scale = np.linalg.norm(expected) if refScale is None else refScale
    scale = max(scale, 1e-30)
    np.testing.assert_allclose(actual, expected, atol=rtol * scale, rtol=0.0)


@pytest.mark.parametrize("moduleName", SELF_OCCLUSION_MODULES.keys())
def test_convexCubeMatchesFacetSrpEffector(moduleName):
    """On a convex cube the SRP self-occlusion effector must reproduce facetSRPDynamicEffector."""
    ctor, rtol = SELF_OCCLUSION_MODULES[moduleName]
    scSim, scObject = createSimulation()
    scObject.hub.r_CN_NInit = np.array([astroConstants.AU2M, 0.0, 0.0])  # [m]
    scObject.hub.v_CN_NInit = np.array([0.0, 0.0, 0.0])  # [m/s]

    sunPayload = messaging.SpicePlanetStateMsgPayload()
    sunPayload.PositionVector = [0.0, 0.0, 0.0]  # [m]
    sunMsg = messaging.SpicePlanetStateMsg().write(sunPayload)

    legacy = facetSRPDynamicEffector.FacetSRPDynamicEffector()
    legacy.ModelTag = "legacySrp"
    legacy.setNumFacets(len(CUBE_NORMALS_B))
    selfOcclusion = ctor()
    if hasattr(selfOcclusion, "setLogOverlapFallbackWarnings"):
        selfOcclusion.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    selfOcclusion.ModelTag = "selfOcclusionSrp"
    for nHat_B, r_B in zip(CUBE_NORMALS_B, CUBE_LOCATIONS_B):
        legacy.addFacet(CUBE_AREA, np.eye(3), nHat_B, np.array([0.0, 0.0, 1.0]), r_B,
                        DIFFUSE_COEFF, SPECULAR_COEFF)
        selfOcclusion.addFacet(CUBE_AREA, DIFFUSE_COEFF, SPECULAR_COEFF, nHat_B, r_B)
    for effector in (legacy, selfOcclusion):
        effector.sunInMsg.subscribeTo(sunMsg)
        scObject.addDynamicEffector(effector)
        scSim.AddModelToTask("task", effector)

    (legacyForce, legacyTorque), (force, torque) = runAndLog(scSim, [legacy, selfOcclusion])
    assert np.linalg.norm(legacyForce) > 0.0
    legacyForce = LEGACY_PRESSURE_SCALE * legacyForce
    legacyTorque = LEGACY_PRESSURE_SCALE * legacyTorque
    assertVectorsClose(force, legacyForce, rtol)
    # The cube torque cancels by symmetry, so compare it on the scale of the individual
    # force * lever arm terms rather than on its own magnitude.
    assertVectorsClose(torque, legacyTorque, rtol, refScale=np.linalg.norm(legacyForce) * CUBE_LEVER_ARM)


def runSrpPlates(ctor, attachToBranch):
    """Run the Srp model on two partially overlapping plates, attached to the hub or to a
    spinning body whose frame coincides with the hub frame, and return the last force and torque."""
    scSim, scObject = createSimulation()
    scObject.hub.sigma_BNInit = np.array([0.0, 0.0, 0.0])  # [-]
    scObject.hub.r_CN_NInit = np.array([astroConstants.AU2M, 0.0, 0.0])  # [m]
    scObject.hub.v_CN_NInit = np.array([0.0, 0.0, 0.0])  # [m/s]

    sunPayload = messaging.SpicePlanetStateMsgPayload()
    sunPayload.PositionVector = [0.0, 0.0, 0.0]  # [m]
    sunMsg = messaging.SpicePlanetStateMsg().write(sunPayload)

    effector = ctor()
    if hasattr(effector, "setLogOverlapFallbackWarnings"):
        effector.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    effector.ModelTag = "selfOcclusionSrp"
    nHat_B = np.array([-1.0, 0.0, 0.0])  # [-] facing the Sun, which is along -x
    effector.addFacet(1.0, DIFFUSE_COEFF, SPECULAR_COEFF, nHat_B, np.array([0.0, 0.0, 0.0]))
    effector.addFacet(1.0, DIFFUSE_COEFF, SPECULAR_COEFF, nHat_B, np.array([-0.5, 0.5, 0.0]))
    effector.sunInMsg.subscribeTo(sunMsg)

    if attachToBranch:
        spinningBody = spinningBodyOneDOFStateEffector.SpinningBodyOneDOFStateEffector()
        spinningBody.ModelTag = "spinningBody"
        spinningBody.mass = 1.0  # [kg]
        spinningBody.sHat_S = [[0.0], [0.0], [1.0]]  # [-]
        spinningBody.dcm_S0B = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # [-]
        spinningBody.r_SB_B = [[0.0], [0.0], [0.0]]  # [m]
        spinningBody.r_ScS_S = [[0.0], [0.0], [0.0]]  # [m]
        spinningBody.IPntSc_S = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # [kg m^2]
        spinningBody.addDynamicEffector(effector)
        scObject.addStateEffector(spinningBody)
        scSim.AddModelToTask("task", spinningBody)
    else:
        scObject.addDynamicEffector(effector)
    scSim.AddModelToTask("task", effector)
    return runAndLog(scSim, [effector])[0]


@pytest.mark.parametrize("moduleName", SELF_OCCLUSION_MODULES.keys())
def test_branchAttachmentMatchesHubAttachment(moduleName):
    """Attached to a state effector whose frame coincides with the hub frame, the SRP effector must
    produce the same partially shadowed force and torque as when attached to the hub."""
    ctor, _ = SELF_OCCLUSION_MODULES[moduleName]
    hubForce, hubTorque = runSrpPlates(ctor, attachToBranch=False)
    branchForce, branchTorque = runSrpPlates(ctor, attachToBranch=True)
    assert np.linalg.norm(hubForce) > 0.0
    assertVectorsClose(branchForce, hubForce, 1e-6)
    assertVectorsClose(branchTorque, hubTorque, 1e-6)
