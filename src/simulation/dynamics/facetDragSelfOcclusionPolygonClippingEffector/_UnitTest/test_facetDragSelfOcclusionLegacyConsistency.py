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
# Purpose:  Cross-check both drag self-occlusion effectors against facetDragDynamicEffector on a
#           convex body, where no self-occlusion can occur, and check that hub and
#           state-effector branch attachment agree.
#

import numpy as np
import pytest

from Basilisk.simulation import exponentialAtmosphere
from Basilisk.simulation import facetDragDynamicEffector
from Basilisk.simulation import facetDragSelfOcclusionPolygonClippingEffector
from Basilisk.simulation import spacecraft
from Basilisk.simulation import spinningBodyOneDOFStateEffector
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros

SIMULATION_TIME_STEP = 1e-4  # [s]

#: (constructor, relative tolerance) of every self-occlusion module compared with the legacy module.
SELF_OCCLUSION_MODULES = {
    "polygonClipping": (facetDragSelfOcclusionPolygonClippingEffector.FacetDragSelfOcclusionPolygonClippingEffector,
                        1e-10),
}

# Unit cube faces: normals and face centers in B. A cube is convex, so no face can shadow another.
CUBE_NORMALS_B = [np.array(n, dtype=float) for n in
                  ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]  # [-]
CUBE_LOCATIONS_B = [0.5 * n for n in CUBE_NORMALS_B]  # [m]
CUBE_AREA = 1.0  # [m^2]
CUBE_LEVER_ARM = 0.5 * np.sqrt(3.0)  # [m] largest facet point distance from B, scales the torque tolerance
DRAG_COEFF = 2.2  # [-]
SIGMA_BN = np.array([0.1, 0.2, -0.3])  # [-] generic attitude with several faces toward the flow


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


def addAtmosphere(scSim, scObject):
    """Add an exponential atmosphere around the spacecraft and return it."""
    atmosphere = exponentialAtmosphere.ExponentialAtmosphere()
    atmosphere.ModelTag = "atmosphere"
    atmosphere.baseDensity = 1.217  # [kg/m^3]
    atmosphere.scaleHeight = 8500.0  # [m]
    atmosphere.planetRadius = 6371e3  # [m]
    atmosphere.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask("task", atmosphere)
    return atmosphere


@pytest.mark.parametrize("moduleName", SELF_OCCLUSION_MODULES.keys())
def test_convexCubeMatchesFacetDragEffector(moduleName):
    """On a convex cube the drag self-occlusion effector must reproduce facetDragDynamicEffector."""
    ctor, rtol = SELF_OCCLUSION_MODULES[moduleName]
    scSim, scObject = createSimulation()
    scObject.hub.r_CN_NInit = np.array([6571e3, 0.0, 0.0])  # [m]
    scObject.hub.v_CN_NInit = np.array([0.0, 7500.0, 0.0])  # [m/s]
    atmosphere = addAtmosphere(scSim, scObject)

    legacy = facetDragDynamicEffector.FacetDragDynamicEffector()
    legacy.ModelTag = "legacyDrag"
    selfOcclusion = ctor()
    if hasattr(selfOcclusion, "setLogOverlapFallbackWarnings"):
        selfOcclusion.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    selfOcclusion.ModelTag = "selfOcclusionDrag"
    for nHat_B, r_B in zip(CUBE_NORMALS_B, CUBE_LOCATIONS_B):
        legacy.addFacet(CUBE_AREA, DRAG_COEFF, nHat_B, r_B)
        selfOcclusion.addFacet(CUBE_AREA, DRAG_COEFF, nHat_B, r_B)
    for effector in (legacy, selfOcclusion):
        effector.atmoDensInMsg.subscribeTo(atmosphere.envOutMsgs[0])
        scObject.addDynamicEffector(effector)
        scSim.AddModelToTask("task", effector)

    (legacyForce, legacyTorque), (force, torque) = runAndLog(scSim, [legacy, selfOcclusion])
    assert np.linalg.norm(legacyForce) > 0.0
    assertVectorsClose(force, legacyForce, rtol)
    # The cube torque cancels by symmetry, so compare it on the scale of the individual
    # force * lever arm terms rather than on its own magnitude.
    assertVectorsClose(torque, legacyTorque, rtol, refScale=np.linalg.norm(legacyForce) * CUBE_LEVER_ARM)


def runDragPlates(ctor, attachToBranch):
    """Run the drag effector on two partially overlapping plates, attached to the hub or to a
    spinning body whose frame coincides with the hub frame, and return the last force and torque."""
    scSim, scObject = createSimulation()
    scObject.hub.sigma_BNInit = np.array([0.0, 0.0, 0.0])  # [-]
    scObject.hub.r_CN_NInit = np.array([6571e3, 0.0, 0.0])  # [m]
    scObject.hub.v_CN_NInit = np.array([0.0, 7500.0, 0.0])  # [m/s]
    atmosphere = addAtmosphere(scSim, scObject)

    effector = ctor()
    if hasattr(effector, "setLogOverlapFallbackWarnings"):
        effector.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    effector.ModelTag = "selfOcclusionDrag"
    nHat_B = np.array([0.0, 1.0, 0.0])  # [-] facing the flow, which comes from +y
    effector.addFacet(1.0, DRAG_COEFF, nHat_B, np.array([0.0, 0.0, 0.0]))
    effector.addFacet(1.0, DRAG_COEFF, nHat_B, np.array([0.5, 0.5, 0.0]))  # upstream, partly shadowing
    effector.atmoDensInMsg.subscribeTo(atmosphere.envOutMsgs[0])

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
    """Attached to a state effector whose frame coincides with the hub frame, the drag effector
    must produce the same partially shadowed force and torque as when attached to the hub."""
    ctor, _ = SELF_OCCLUSION_MODULES[moduleName]
    hubForce, hubTorque = runDragPlates(ctor, attachToBranch=False)
    branchForce, branchTorque = runDragPlates(ctor, attachToBranch=True)
    assert np.linalg.norm(hubForce) > 0.0
    assertVectorsClose(branchForce, hubForce, 1e-6)
    assertVectorsClose(branchTorque, hubTorque, 1e-6)
