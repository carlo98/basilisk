
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
# Purpose:  Test the flat-plate drag model of facetDragSelfOcclusionPolygonClippingEffector and its
#         exact planar self-shadowing computation.
#

import numpy as np
import pytest

# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import RigidBodyKinematics as rbk

# import simulation related support
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft
from Basilisk.simulation import exponentialAtmosphere
from Basilisk.simulation import facetDragSelfOcclusionPolygonClippingEffector
from Basilisk.simulation import simpleNav
from Basilisk.simulation import zeroWindModel
from Basilisk.utilities import simHelpers
from Basilisk.utilities import simIncludeGravBody


# --------------------------------------------------------------------------
# Python reference implementation of the module's self-shadowing algorithm,
# used to independently cross-check the C++ implementation against logged
# simulation data.
# --------------------------------------------------------------------------

def buildFacetRectangle(normal, area, width=-1.0, height=-1.0):
    """Reference implementation of buildFacetRectangles() for a single facet."""
    normal = np.array(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - normal * np.dot(ref, normal)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    side = np.sqrt(area)
    w = side if width < 0.0 else width
    h = side if height < 0.0 else height
    # Corners are built as center +/- edge1 +/- edge2, so the edges must be
    # half-extents for the rectangle to span the intended width x height.
    return 0.5 * w * e1, 0.5 * h * e2


def cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


def clipPolygon(subject, edgeStart, edgeEnd):
    """Reference implementation of the Sutherland-Hodgman clip step."""
    output = []
    n = len(subject)
    if n == 0:
        return output
    edgeDir = edgeEnd - edgeStart
    for i in range(n):
        cur = subject[i]
        prev = subject[(i - 1) % n]
        curInside = cross2(edgeDir, cur - edgeStart) >= 0.0
        prevInside = cross2(edgeDir, prev - edgeStart) >= 0.0
        hasHit = False
        hit = None
        if curInside != prevInside:
            seg = cur - prev
            denom = cross2(edgeDir, seg)
            if abs(denom) > 1e-12:
                # denom == cross2(edgeDir, cur - prev) == D_cur - D_prev, so the
                # fraction along prev->cur where the boundary is crossed,
                # t = D_prev / (D_prev - D_cur), equals -D_prev/denom (note the sign).
                t = -cross2(edgeDir, prev - edgeStart) / denom
                t = min(1.0, max(0.0, t))
                hit = prev + t * seg
                hasHit = True
        if curInside:
            if not prevInside and hasHit:
                output.append(hit)
            output.append(cur)
        elif prevInside and hasHit:
            output.append(hit)
    return output


def polygonIntersection(clipWindow, subject):
    """Reference implementation of polygonIntersection(): subject clipped to clipWindow."""
    clip = list(subject)
    n = len(clipWindow)
    for i in range(n):
        j = (i + 1) % n
        clip = clipPolygon(clip, clipWindow[i], clipWindow[j])
        if len(clip) == 0:
            return clip
    return clip


def intersectAll(polys):
    """Reference implementation of intersectAll(): intersection of every polygon in `polys`."""
    if not polys:
        return []
    result = list(polys[0])
    for poly in polys[1:]:
        if not result:
            break
        result = polygonIntersection(result, poly)
    return result


def polygonAreaAndMoment(poly):
    """Reference implementation of polygonAreaAndMoment(): returns (area, moment) where
    moment == centroid * area, via the standard shoelace-based centroid formula."""
    if len(poly) < 3:
        return 0.0, np.zeros(2)
    signedArea = 0.0
    mx = 0.0
    my = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        cross = poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
        signedArea += cross
        mx += (poly[i][0] + poly[j][0]) * cross
        my += (poly[i][1] + poly[j][1]) * cross
    signedArea *= 0.5
    if abs(signedArea) < 1e-15:
        return 0.0, np.zeros(2)
    centroid = np.array([mx / (6.0 * signedArea), my / (6.0 * signedArea)])
    area = abs(signedArea)
    return area, centroid * area


def clipToUpstreamHalfPlane(poly, n_i, c_i, flowCosine_i, n_j, c_j, flowCosine_j, p1, p2):
    """Reference implementation of clipToUpstreamHalfPlane().

    Clips `poly` to the region where facet j's plane is upstream of (closer to the source
    than) facet i's plane, evaluated pointwise via each facet's affine depth(x, y) function
    -- not just at the facet centers. The depth is measured along the flux propagation
    direction, depth_k(x, y) = (x n_k.p1 + y n_k.p2 - n_k.c_k) / (n_k.d), so the shadow region
    is D = depth_i - depth_j >= 0.
    """
    A = np.dot(n_i, p1) / flowCosine_i - np.dot(n_j, p1) / flowCosine_j
    B = np.dot(n_i, p2) / flowCosine_i - np.dot(n_j, p2) / flowCosine_j
    C = np.dot(n_j, c_j) / flowCosine_j - np.dot(n_i, c_i) / flowCosine_i

    tol = 1e-12  # [-]
    if abs(A) < tol and abs(B) < tol:
        return list(poly) if C >= 0.0 else []

    P0 = np.array([-C / A, 0.0]) if abs(A) >= abs(B) else np.array([0.0, -C / B])
    edgeDir = np.array([B, -A])
    return clipPolygon(poly, P0, P0 + edgeDir)


def exposureFactors(areas, normals, locations, edges1, edges2, v_hat_B):
    """Reference implementation of computeExposureFactor() for every facet.

    Mirrors the C++ implementation exactly: for each pair (i, j) the shadow patch is
    the rectangle overlap further clipped to wherever j is pointwise upstream of i, and
    the total shadowed area on i is the exact union (via inclusion-exclusion) of the
    patches cast by every other facet -- not their naive sum, which would double-count
    any region shadowed by more than one upstream facet. Also mirrors the exposed
    region's center-of-pressure tracking: since a partially-shadowed facet's exposed
    centroid generally is not the facet's own center, the same inclusion-exclusion
    tracks each patch's first moment (not just its area) so the exposed centroid can be
    recovered and mapped back onto facet i's plane via its affine depth(x, y) function.

    Returns (exposures, centersOfPressure_B), each a length-n list; a facet with no
    shadowing at all gets its own (exact) location as its center of pressure.
    """
    n = len(areas)
    if np.linalg.norm(v_hat_B) < 1e-12:
        return [1.0] * n, [np.array(loc, dtype=float) for loc in locations]

    ref = np.array([1.0, 0.0, 0.0]) if abs(v_hat_B[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = ref - v_hat_B * np.dot(ref, v_hat_B)
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(v_hat_B, p1)

    def makeRect(idx):
        c = np.array(locations[idx], dtype=float)
        e1 = edges1[idx]
        e2 = edges2[idx]
        corners = [c - e1 - e2, c + e1 - e2, c + e1 + e2, c - e1 + e2]
        pts = [np.array([np.dot(corner, p1), np.dot(corner, p2)]) for corner in corners]
        flowCosine = np.dot(normals[idx], v_hat_B)
        projArea = areas[idx] * abs(flowCosine)
        return pts, flowCosine, projArea

    rects = [makeRect(i) for i in range(n)]
    tol = 1e-12  # [-]
    exposures = []
    centersOfPressure_B = []
    for i in range(n):
        ptsI, flowCosineI, projI = rects[i]
        centerOfPressure_B = np.array(locations[i], dtype=float)
        if projI <= tol or abs(flowCosineI) <= tol:
            exposures.append(0.0)
            centersOfPressure_B.append(centerOfPressure_B)
            continue

        shadowPieces = []
        for j in range(n):
            if j == i:
                continue
            ptsJ, flowCosineJ, _ = rects[j]
            if abs(flowCosineJ) <= tol:
                continue
            overlap = polygonIntersection(ptsI, ptsJ)
            if not overlap:
                continue
            shadow = clipToUpstreamHalfPlane(
                overlap,
                normals[i], locations[i], flowCosineI,
                normals[j], locations[j], flowCosineJ,
                p1, p2)
            if shadow:
                shadowPieces.append(shadow)

        if not shadowPieces:
            # No shadowing at all: facet i's own center is already exact.
            exposures.append(1.0)
            centersOfPressure_B.append(centerOfPressure_B)
            continue

        shadowedArea = 0.0
        shadowedMoment2D = np.zeros(2)
        k = len(shadowPieces)
        for mask in range(1, 1 << k):
            subset = [shadowPieces[bit] for bit in range(k) if mask & (1 << bit)]
            bitCount = bin(mask).count("1")
            area, moment = polygonAreaAndMoment(intersectAll(subset))
            sign = 1.0 if bitCount % 2 == 1 else -1.0
            shadowedArea += sign * area
            shadowedMoment2D += sign * moment

        exposedArea = max(0.0, projI - shadowedArea)
        if exposedArea > tol:
            fullCentroid2D = np.array([np.dot(locations[i], p1), np.dot(locations[i], p2)])
            fullMoment2D = fullCentroid2D * projI
            exposedCentroid2D = (fullMoment2D - shadowedMoment2D) / exposedArea
            depth_i = (np.dot(normals[i], locations[i])
                       - exposedCentroid2D[0] * np.dot(normals[i], p1)
                       - exposedCentroid2D[1] * np.dot(normals[i], p2)) / flowCosineI
            centerOfPressure_B = exposedCentroid2D[0] * p1 + exposedCentroid2D[1] * p2 + depth_i * v_hat_B
        exposures.append(min(1.0, exposedArea / projI))
        centersOfPressure_B.append(centerOfPressure_B)
    return exposures, centersOfPressure_B


def facetDragSelfOcclusionComp(dens, areas, coeffs, normals, locations, widths, heights, vel, att):
    """Reference facet drag force/torque in the body frame, including self-shadowing."""
    dcm = rbk.MRP2C(att)
    vMag = np.linalg.norm(vel)
    force = np.zeros(3)
    torque = np.zeros(3)
    if vMag <= 1e-12:
        return force, torque
    v_hat_B = dcm.dot(vel) / vMag

    edges = [buildFacetRectangle(normals[i], areas[i], widths[i], heights[i]) for i in range(len(areas))]
    edges1 = [e[0] for e in edges]
    edges2 = [e[1] for e in edges]
    exposures, centersOfPressure_B = exposureFactors(areas, normals, locations, edges1, edges2, v_hat_B)

    for i in range(len(areas)):
        projTerm = np.dot(normals[i], v_hat_B)
        projArea = areas[i] * projTerm * exposures[i]
        if projArea > 0.0:
            f = -0.5 * dens * projArea * coeffs[i] * vMag ** 2.0 * v_hat_B
            force += f
            torque += -np.cross(f, centersOfPressure_B[i])
    return force, torque


# --------------------------------------------------------------------------
# Simulation setup helper
# --------------------------------------------------------------------------

def setupSelfOcclusionSim(areas, coeffs, normals, locations, widths=None, heights=None,
                           rN=None, vN=None, sigmaBN=None, simulationTimeStep=1e-4):
    if widths is None:
        widths = [-1.0] * len(areas)  # [m], -1 derives the side from the area
    if heights is None:
        heights = [-1.0] * len(areas)  # [m], -1 derives the side from the area

    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(simulationTimeStep)))

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"

    newAtmo = exponentialAtmosphere.ExponentialAtmosphere()
    newAtmo.ModelTag = "ExpAtmo"
    newAtmo.baseDensity = 1.217  # [kg/m^3]
    newAtmo.scaleHeight = 8500.0  # [m]
    newAtmo.planetRadius = 6371.0 * 1000.0  # [m]
    newAtmo.addSpacecraftToModel(scObject.scStateOutMsg)

    newDrag = facetDragSelfOcclusionPolygonClippingEffector.FacetDragSelfOcclusionPolygonClippingEffector()
    newDrag.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    newDrag.ModelTag = "FacetDragSelfOcclusion"
    newDrag.atmoDensInMsg.subscribeTo(newAtmo.envOutMsgs[0])

    for area, coeff, normal, loc, width, height in zip(areas, coeffs, normals, locations, widths, heights):
        newDrag.addFacet(area, coeff, np.array(normal, dtype=float), np.array(loc, dtype=float), width, height)

    scObject.addDynamicEffector(newDrag)

    if rN is None:
        rN = np.array([6571e3, 0.0, 0.0])  # [m]
    if vN is None:
        vN = np.array([0.0, 7500.0, 0.0])  # [m/s]
    if sigmaBN is None:
        sigmaBN = np.array([0.0, 0.0, 0.0])  # [-] MRP

    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    scObject.hub.sigma_BNInit = sigmaBN

    scSim.AddModelToTask(simTaskName, scObject)
    scSim.AddModelToTask(simTaskName, newAtmo)
    scSim.AddModelToTask(simTaskName, newDrag)

    dataLog = scObject.scStateOutMsg.recorder()
    atmoLog = newAtmo.envOutMsgs[0].recorder()
    dragForceLog = newDrag.logger("forceExternal_B")
    dragTorqueLog = newDrag.logger("torqueExternalPntB_B")
    scSim.AddModelToTask(simTaskName, dataLog)
    scSim.AddModelToTask(simTaskName, atmoLog)
    scSim.AddModelToTask(simTaskName, dragForceLog)
    scSim.AddModelToTask(simTaskName, dragTorqueLog)

    return scSim, scObject, newDrag, dataLog, atmoLog, dragForceLog, dragTorqueLog


def runAndCheck(areas, coeffs, normals, locations, widths=None, heights=None,
                 rN=None, vN=None, sigmaBN=None, simulationTimeStep=1e-4):
    (scSim, scObject, newDrag, dataLog, atmoLog,
     dragForceLog, dragTorqueLog) = setupSelfOcclusionSim(areas, coeffs, normals, locations,
                                                            widths, heights, rN, vN, sigmaBN,
                                                            simulationTimeStep)
    if widths is None:
        widths = [-1.0] * len(areas)  # [m], -1 derives the side from the area
    if heights is None:
        heights = [-1.0] * len(areas)  # [m], -1 derives the side from the area

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(simulationTimeStep))
    scSim.ExecuteSimulation()

    # The drag force is computed from the state at the *start* of the (single)
    # integration step. The step is kept extremely short so that the orbital and
    # (for off-center facets) torque-driven attitude dynamics cannot evolve
    # appreciably within it -- otherwise the post-integration state logged here
    # would reflect real dynamics the simplified reference formula never models,
    # rather than letting this be a clean instantaneous check of the drag formula.
    dens = atmoLog.neutralDensity[0]
    vel = dataLog.v_BN_N[0]
    att = dataLog.sigma_BN[0]

    refForce, refTorque = facetDragSelfOcclusionComp(dens, areas, coeffs, normals, locations,
                                                       widths, heights, vel, att)
    force = dragForceLog.forceExternal_B[-1]
    torque = dragTorqueLog.torqueExternalPntB_B[-1]
    return force, torque, refForce, refTorque


def test_facetDragSelfOcclusionInertialVelocity():
    """A single facet with no possible self-shadowing must match the plain flat-plate formula."""
    areas = [1.0]  # [m^2]
    coeffs = [2.2]  # [-]
    normals = [np.array([1.0, 0.0, 0.0])]  # [-]
    locations = [np.array([0.0, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(
        areas, coeffs, normals, locations,
        rN=np.array([6571e3, 0.0, 0.0]),
        vN=np.array([3000.0, 7000.0, 0.0]),
        sigmaBN=np.array([0.0, 0.0, 0.0]))

    assert np.allclose(force, refForce, atol=1e-10)
    assert np.allclose(torque, refTorque, atol=1e-10)
    # Sanity check: a single facet cannot self-shadow, so force must be non-zero here.
    assert np.linalg.norm(force) > 0.0


def test_facetDragSelfOcclusionFullOcclusion():
    """Two identical, directly-stacked facets: the downwind one should be fully shadowed."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [2.2, 2.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    # Both centered on the same line along the flow direction (y), same unit-square footprint.
    locations = [np.array([0.0, 0.5, 0.0]), np.array([0.0, -0.5, 0.0])]  # upwind, downwind

    force, torque, refForce, refTorque = runAndCheck(
        areas, coeffs, normals, locations,
        rN=np.array([6571e3, 0.0, 0.0]),
        vN=np.array([0.0, 7500.0, 0.0]),
        sigmaBN=np.array([0.0, 0.0, 0.0]))

    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    # The upwind facet alone would produce this force; if the downwind facet is fully
    # shadowed, the total force should match the single-facet (upwind-only) case.
    upwindOnlyForce, upwindOnlyTorque, _, _ = runAndCheck(
        [areas[0]], [coeffs[0]], [normals[0]], [locations[0]],
        rN=np.array([6571e3, 0.0, 0.0]),
        vN=np.array([0.0, 7500.0, 0.0]),
        sigmaBN=np.array([0.0, 0.0, 0.0]))

    assert np.allclose(force, upwindOnlyForce, atol=1e-9)
    assert np.allclose(torque, upwindOnlyTorque, atol=1e-9)


def test_facetDragSelfOcclusionPartialOcclusion():
    """Two identical, partially-overlapping facets: exposure should be strictly between 0 and 1."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [2.2, 2.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    # Downwind facet is offset half a facet-width sideways (x) from the upwind one.
    locations = [np.array([0.0, 0.5, 0.0]), np.array([0.5, -0.5, 0.0])]  # upwind, downwind

    force, torque, refForce, refTorque = runAndCheck(
        areas, coeffs, normals, locations,
        rN=np.array([6571e3, 0.0, 0.0]),
        vN=np.array([0.0, 7500.0, 0.0]),
        sigmaBN=np.array([0.0, 0.0, 0.0]))

    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    # Total force should be strictly between the fully-shadowed (upwind-only) and
    # fully-unshadowed (both facets fully exposed) extremes.
    upwindOnlyForce, _, _, _ = runAndCheck(
        [areas[0]], [coeffs[0]], [normals[0]], [locations[0]],
        rN=np.array([6571e3, 0.0, 0.0]),
        vN=np.array([0.0, 7500.0, 0.0]),
        sigmaBN=np.array([0.0, 0.0, 0.0]))
    upwindMag = np.linalg.norm(upwindOnlyForce)
    unshadowedForceMag = 2.0 * upwindMag

    forceMag = np.linalg.norm(force)
    # Relative margins: an absolute epsilon is meaningless here since the force
    # magnitude scales with atmospheric density, which varies over many orders of
    # magnitude with altitude.
    assert forceMag > upwindMag * (1.0 + 1e-6)
    assert forceMag < unshadowedForceMag * (1.0 - 1e-6)


def test_facetDragSelfOcclusionNoOcclusionWhenSeparated():
    """Facets far enough apart that their silhouettes never overlap should not shadow each other."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [2.2, 2.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    # Same depth along the flow, offset well beyond one facet width sideways (x).
    locations = [np.array([0.0, 0.0, 0.0]), np.array([50.0, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(
        areas, coeffs, normals, locations,
        rN=np.array([6571e3, 0.0, 0.0]),
        vN=np.array([0.0, 7500.0, 0.0]),
        sigmaBN=np.array([0.0, 0.0, 0.0]))

    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    upwindOnlyForce, _, _, _ = runAndCheck(
        [areas[0]], [coeffs[0]], [normals[0]], [locations[0]],
        rN=np.array([6571e3, 0.0, 0.0]),
        vN=np.array([0.0, 7500.0, 0.0]),
        sigmaBN=np.array([0.0, 0.0, 0.0]))

    assert np.allclose(force, 2.0 * upwindOnlyForce, atol=1e-9)


def test_facetDragSelfOcclusionAtmosphereRelativeVelocityWhenWindLinked():
    """Verify drag uses atmosphere-relative velocity when a wind message is linked.

    Two simulation steps are needed: step 1 populates windInBuffer via readInputMessages();
    step 2's computeForceTorque uses that cached wind velocity.
    """
    # Kept extremely short (see runAndCheck) so that the orbital/attitude state
    # barely evolves between the two steps -- this test only needs two distinct
    # readInputMessages() calls, not any real dynamics propagation.
    simulationTimeStep = 1e-4  # [s]

    simTaskName = "simTask"
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.CreateNewProcess("sim").addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(simulationTimeStep)))

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"

    newAtmo = exponentialAtmosphere.ExponentialAtmosphere()
    newAtmo.ModelTag = "ExpAtmo"
    newAtmo.baseDensity = 1.217  # [kg/m^3]
    newAtmo.scaleHeight = 8500.0  # [m]
    newAtmo.planetRadius = 6371e3  # [m]
    newAtmo.addSpacecraftToModel(scObject.scStateOutMsg)

    windMdl = zeroWindModel.ZeroWindModel()
    windMdl.ModelTag = "wind"
    windMdl.addSpacecraftToModel(scObject.scStateOutMsg)
    planetStateMsg = messaging.SpicePlanetStateMsg().write(messaging.SpicePlanetStateMsgPayload())
    windMdl.planetPosInMsg.subscribeTo(planetStateMsg)
    omega_earth = np.array([0.0, 0.0, orbitalMotion.OMEGA_EARTH])  # [rad/s]
    windMdl.setPlanetOmega_N(omega_earth)

    newDrag = facetDragSelfOcclusionPolygonClippingEffector.FacetDragSelfOcclusionPolygonClippingEffector()
    newDrag.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    newDrag.ModelTag = "FacetDragSelfOcclusion"
    newDrag.addFacet(1.0, 2.2, np.array([0, 1, 0]), np.array([0, 0, 0.1]))  # area, Cd, normal, CoP_B
    newDrag.atmoDensInMsg.subscribeTo(newAtmo.envOutMsgs[0])
    newDrag.windVelInMsg.subscribeTo(windMdl.envOutMsgs[0])
    scObject.addDynamicEffector(newDrag)

    scObject.hub.r_CN_NInit = np.array([6571e3, 0.0, 0.0])  # [m]
    scObject.hub.v_CN_NInit = np.array([0.0, 7600.0, 0.0])  # [m/s]
    scObject.hub.sigma_BNInit = np.array([0.0, 0.0, 0.0])  # [-] MRP

    scSim.AddModelToTask(simTaskName, scObject)
    scSim.AddModelToTask(simTaskName, newAtmo)
    scSim.AddModelToTask(simTaskName, windMdl)
    scSim.AddModelToTask(simTaskName, newDrag)

    scSim.InitializeSimulation()

    windLog = windMdl.envOutMsgs[0].recorder()
    scLog = scObject.scStateOutMsg.recorder()
    atmoLog = newAtmo.envOutMsgs[0].recorder()
    dragLog = newDrag.logger("forceExternal_B")
    scSim.AddModelToTask(simTaskName, windLog)
    scSim.AddModelToTask(simTaskName, scLog)
    scSim.AddModelToTask(simTaskName, atmoLog)
    scSim.AddModelToTask(simTaskName, dragLog)

    scSim.ConfigureStopTime(macros.sec2nano(2.0 * simulationTimeStep))
    scSim.ExecuteSimulation()

    # windInBuffer/atmoInBuffer are cached at the start of step 1 (index 0). Step 2's
    # computeForceTorque then runs from the state at the start of step 2, i.e. the
    # state produced by integrating step 1 (index 1) -- not the final, post-step-2
    # state (index -1).
    dens_step1 = atmoLog.neutralDensity[0]
    wind_step1 = np.array(windLog.v_air_N[0])
    v_step2 = np.array(scLog.v_BN_N[1])
    sigma_step2 = np.array(scLog.sigma_BN[1])
    drag_force = np.array(dragLog.forceExternal_B[-1])

    expected_force, _ = facetDragSelfOcclusionComp(
        dens_step1, [1.0], [2.2], [np.array([0, 1, 0])], [np.array([0, 0, 0.1])], [-1.0], [-1.0],
        v_step2 - wind_step1, sigma_step2)
    np.testing.assert_allclose(drag_force, expected_force, atol=1e-10)


# --------------------------------------------------------------------------
# Sign-convention tests for the exact depth-plane clipping and shadow-union
# logic in computeExposureFactor(). Every case below is checked in two
# independent ways:
# (1) against facetDragSelfOcclusionComp/exposureFactors, the Python mirror
#     of the exact C++ algorithm (via runAndCheck's built-in refForce);
# (2) against a force magnitude hand-derived directly from the known
#     exponential-atmosphere/flat-plate-drag formulas and the geometrically
#     reasoned exposure fractions, independent of either implementation
#
# All cases use runAndCheck's default orbit (rN=(6571e3,0,0), vN=(0,7500,0),
# sigmaBN=0), so v_hat_B == (0, 1, 0) exactly and the density/speed below are
# the corresponding known constants.
# --------------------------------------------------------------------------

_signTestDensity = 1.217 * np.exp(-(6571e3 - 6371e3) / 8500.0)  # [kg/m^3], baseDensity*exp(-altitude/scaleHeight)
_signTestSpeed = 7500.0  # [m/s]
_signTestCd = 2.2  # [-]


def expectedForceYFromExposureSum(exposureWeightedAreaSum):
    """-0.5 * dens * Cd * vMag^2 * Sum_k(area_k * projTerm_k * exposure_k), derived
    directly from the flat-plate drag formula and the known orbit constants above --
    independent of both the C++ module and its Python mirror."""
    return -0.5 * _signTestDensity * _signTestCd * _signTestSpeed ** 2.0 * exposureWeightedAreaSum


def test_facetDragSelfOcclusionSignConvention_parallelUpstream():
    """Case 1: parallel facets, j clearly upstream -- j must fully shadow i, and i
    must NOT shadow j back (checks the directionality of the depth comparison)."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [_signTestCd, _signTestCd]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i downstream, j upstream

    force, torque, refForce, refTorque = runAndCheck(areas, coeffs, normals, locations)
    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    # exposure_i = 0 (fully shadowed), exposure_j = 1 (unshadowed): total force must
    # equal a single fully-exposed unit facet's force.
    expectedForceY = expectedForceYFromExposureSum(1.0 * 1.0 * 1.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetDragSelfOcclusionSignConvention_parallelDownstream():
    """Case 2: parallel facets, j clearly downstream -- the mirror of case 1: now i is
    upstream and must stay fully exposed, while the downstream j gets fully shadowed."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [_signTestCd, _signTestCd]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]  # i upstream, j downstream

    force, torque, refForce, refTorque = runAndCheck(areas, coeffs, normals, locations)
    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    expectedForceY = expectedForceYFromExposureSum(1.0 * 1.0 * 1.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetDragSelfOcclusionSignConvention_tiltedFacetCrossesDepth():
    """Case 3: a facet tilted 30 deg from the flat target crosses the target's depth
    exactly through the overlap's center, so it shadows exactly the downstream half of
    the overlap and is itself shadowed over exactly half its own silhouette. This
    exercises the pointwise (not center-only) depth comparison for a facet that is
    upstream over part of the overlap and downstream over the rest."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [_signTestCd, _signTestCd]
    tiltAngle = np.pi / 6.0  # 30 deg
    normals = [np.array([0.0, 1.0, 0.0]),  # [-]
               np.array([np.sin(tiltAngle), np.cos(tiltAngle), 0.0])]
    locations = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, coeffs, normals, locations)
    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    # Hand-derived: with a flat target (depth constant across its silhouette) and this
    # tilted facet, the depth-crossing line always passes through the tilted facet's
    # own projected center, so exposure_j = 0.5 exactly regardless of overlap extent.
    # Working out the overlap geometry for this centered case gives
    # exposure_i = 1 - 0.25*sqrt(3); the area*projTerm*exposure sum below comes out to
    # exactly 1.0, i.e. the same total force as a single fully-exposed unit facet.
    cosTilt = np.cos(tiltAngle)
    exposure_i = 1.0 - 0.25 * np.sqrt(3.0)  # [-]
    exposure_j = 0.5  # [-]
    exposureWeightedAreaSum = 1.0 * 1.0 * exposure_i + 1.0 * cosTilt * exposure_j
    assert abs(exposureWeightedAreaSum - 1.0) < 1e-12
    expectedForceY = expectedForceYFromExposureSum(exposureWeightedAreaSum)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetDragSelfOcclusionSignConvention_partialOverlapPartialUpstream():
    """Case 4: the same 30 deg tilted facet as above, shifted sideways so only part of
    its silhouette overlaps the target at all, and only part of that overlap is
    upstream of the target."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [_signTestCd, _signTestCd]
    tiltAngle = np.pi / 6.0  # 30 deg
    normals = [np.array([0.0, 1.0, 0.0]),  # [-]
               np.array([np.sin(tiltAngle), np.cos(tiltAngle), 0.0])]
    locations = [np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, coeffs, normals, locations)
    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    # Hand-derived, with the flow along +y and projected coordinate x: the tilted facet j
    # spans x in [0.3 - cos30/2, 0.3 + cos30/2] = [-0.133, 0.733] and is upstream of the flat
    # target i (y > 0) for x < 0.3. It therefore shadows i over x in [-0.133, 0.3], a width
    # of cos30/2 = sqrt(3)/4, so exposure_i = 1 - sqrt(3)/4. The target shadows the
    # downstream part of j only where they overlap, x in [0.3, 0.5], a width of 0.2 out of
    # cos30, so exposure_j = 1 - 0.2/cos30.
    cosTilt = np.cos(tiltAngle)
    exposure_i = 1.0 - 0.25 * np.sqrt(3.0)  # [-]
    exposure_j = 1.0 - 0.2 / np.cos(tiltAngle)  # [-]
    exposureWeightedAreaSum = 1.0 * 1.0 * exposure_i + 1.0 * cosTilt * exposure_j
    expectedForceY = expectedForceYFromExposureSum(exposureWeightedAreaSum)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetDragSelfOcclusionSignConvention_completeOverlapLargerShadower():
    """Case 5: a larger upstream facet completely contains the (smaller, downstream)
    target's silhouette -- the target must be fully (not just partially) shadowed,
    and the larger facet's own exposure must be unaffected by the smaller one."""
    areas = [1.0, 4.0]  # [m^2]
    coeffs = [_signTestCd, _signTestCd]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i downstream+smaller, j upstream+bigger

    force, torque, refForce, refTorque = runAndCheck(areas, coeffs, normals, locations)
    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    # exposure_i = 0 (fully contained in j's shadow), exposure_j = 1 (unaffected).
    expectedForceY = expectedForceYFromExposureSum(4.0 * 1.0 * 1.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetDragSelfOcclusionSignConvention_upstreamCoefficientsWin():
    """Case 7: two identical, fully overlapping parallel facets with different drag
    coefficients. The flow comes from +y, so only the facet at larger y is exposed and the
    force carries its coefficient alone. Unlike the cases above, whose totals are invariant
    under swapping the upstream and downstream facets, this pins down the direction of the
    depth comparison."""
    upstreamCd = 2.0  # [-]
    downstreamCd = 3.0  # [-]
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [downstreamCd, upstreamCd]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i downstream, j upstream

    force, torque, refForce, refTorque = runAndCheck(areas, coeffs, normals, locations)
    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    expectedForceY = -0.5 * _signTestDensity * upstreamCd * _signTestSpeed ** 2.0 * 1.0  # [N]
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetDragSelfOcclusionSignConvention_zeroProjectedArea():
    """Case 6: a facet edge-on to the flow (zero projected area) must contribute no
    force, and must neither spuriously shadow, nor be treated as shadowing, the other,
    genuinely front-facing facet."""
    areas = [1.0, 1.0]  # [m^2]
    coeffs = [_signTestCd, _signTestCd]
    normals = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i is edge-on to the flow
    locations = [np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, coeffs, normals, locations)
    assert np.allclose(force, refForce, atol=1e-9)
    assert np.allclose(torque, refTorque, atol=1e-9)

    # i contributes nothing regardless of exposure (projTerm_i = 0); j is unaffected.
    expectedForceY = expectedForceYFromExposureSum(1.0 * 1.0 * 1.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


# --------------------------------------------------------------------------
# Tests ported from facetDragEffector/_UnitTest/test_unitFacetDrag.py. Every
# facet configuration below places its panel flush on the corresponding face
# of a cube: the location's component along the facet's own normal is set to
# at least that facet's own half side-length (sqrt(area)/2), with any
# additional offset in the ORIGINAL test's location kept as a perpendicular
# lever arm.
# These confirm the new module's occlusion logic does not introduce spurious
# self-shadowing for geometry where none should occur.
# --------------------------------------------------------------------------

def cubeFaceLocation(normal, area, location):
    """Push `location` out to at least the facet's own half-width along its normal,
    keeping any component of `location` perpendicular to the normal as a lever arm."""
    normal = np.array(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    location = np.array(location, dtype=float)
    half = 0.5 * np.sqrt(area)
    leverArm = location - normal * normal.dot(location)
    return half * normal + leverArm


test_drag = [([1.0, 1.0], np.array([2.0, 2.0]), [np.array([1, 0, 0]), np.array([0, 1, 0])], [np.array([0.1, 0, 0]), np.array([0, 0.1, 0])]),
             ([1.0, 1.0], np.array([2.0, 2.0]), [np.array([1, 0, 0]), np.array([0, 1, 0])], [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]),
             ([1.0, 2.0], np.array([2.0, 4.0]), [np.array([1, 0, 0]), np.array([0, 1, 0])], [np.array([0.1, 0, 0]), np.array([0, 0.1, 0])]),
             ([1.0, 1.0], np.array([2.0, 2.0]), [np.array([1, 0, 0]), np.array([0, 1, 0])], [np.array([0.1, 0, 0]), np.array([0, 0, 0.1])]),
             ([1.0, 1.0], np.array([2.0, 2.0]), [np.array([1, 0, 0]), np.array([0, 0, 1])], [np.array([0.1, 0, 0]), np.array([0, 0, 0.1])]),
             ([1.0, 1.0], np.array([2.0, 2.0]), [np.array([0, 0, -1]), np.array([0, -1, 0])], [np.array([0, 0, 0.1]), np.array([0, 0.1, 0])]),
]
test_drag = [(areas, coeffs, normals, [cubeFaceLocation(n, a, loc) for n, a, loc in zip(normals, areas, locations)])
             for areas, coeffs, normals, locations in test_drag]

@pytest.mark.parametrize("scAreas, scCoeff, B_normals, B_locations", test_drag)
def test_DragCalculation(scAreas, scCoeff, B_normals, B_locations):
    """Cube-face-panel drag check: two orthogonal facets, no self-shadowing possible."""

    ## Simulation initialization
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()

    dynProcess = scSim.CreateNewProcess(simProcessName)
    # Kept extremely short (see runAndCheck above) so gravity drift and any
    # torque-driven attitude change cannot evolve appreciably within the single
    # logged step
    simulationTimeStep = macros.sec2nano(1e-4)  # [s]
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"

    ## Initialize new atmosphere and drag model, add them to task
    newAtmo = exponentialAtmosphere.ExponentialAtmosphere()
    newAtmo.ModelTag = "ExpAtmo"
    newAtmo.addSpacecraftToModel(scObject.scStateOutMsg)

    newDrag = facetDragSelfOcclusionPolygonClippingEffector.FacetDragSelfOcclusionPolygonClippingEffector()
    newDrag.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    newDrag.ModelTag = "FacetDragSelfOcclusion"
    newDrag.atmoDensInMsg.subscribeTo(newAtmo.envOutMsgs[0])

    scObject.addDynamicEffector(newDrag)

    try:
        for i in range(0, len(scAreas)):
            newDrag.addFacet(scAreas[i], scCoeff[i], B_normals[i], B_locations[i])
    except:
        pytest.fail("ERROR: FacetDragSelfOcclusion unit test failed while setting facet parameters.")

    # clear prior gravitational body and SPICE setup definitions
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()

    planet.isCentralBody = True          # ensure this is the central gravitational body
    # attach gravity model to spacecraft
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

    # setup orbit and simulation time
    r_eq = 6371 * 1000.0  # [m]
    refBaseDens = 1.217  # [kg/m^3]
    refScaleHeight = 8500.0  # [m]

    # Set base density, equatorial radius, scale height in Atmosphere
    newAtmo.baseDensity = refBaseDens
    newAtmo.scaleHeight = refScaleHeight
    newAtmo.planetRadius = r_eq

    rN = np.array([r_eq + 200.0e3, 0, 0])  # [m]
    vN = np.array([0, 7.788e3, 0])  # [m/s]
    sig_BN = np.array([0, 0, 0])  # [-] MRP
    # initialize Spacecraft States with the initialization variables
    scObject.hub.r_CN_NInit = rN  # m - r_CN_N
    scObject.hub.v_CN_NInit = vN  # m - v_CN_N
    scObject.hub.sigma_BNInit = sig_BN

    simulationTime = simulationTimeStep

    # add BSK objects to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)
    scSim.AddModelToTask(simTaskName, newAtmo)
    scSim.AddModelToTask(simTaskName, newDrag)

    # setup logging
    dataLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, dataLog)
    atmoLog = newAtmo.envOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, atmoLog)
    newDragLog = newDrag.logger(["forceExternal_B", "torqueExternalPntB_B"])
    scSim.AddModelToTask(simTaskName, newDragLog)

    # initialize Simulation
    scSim.InitializeSimulation()

    # configure a simulation stop time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # Retrieve logged data
    dragDataForce_B = simHelpers.addTimeColumn(newDragLog.times(), newDragLog.forceExternal_B)
    dragTorqueData = simHelpers.addTimeColumn(newDragLog.times(), newDragLog.torqueExternalPntB_B)
    velData = dataLog.v_BN_N
    attData = dataLog.sigma_BN
    densData = atmoLog.neutralDensity
    np.set_printoptions(precision=16)

    def checkFacetDragForce(dens, area, coeff, facet_dir, sigma_BN, inertial_vel):
        dcm = rbk.MRP2C(sigma_BN)
        vMag = np.linalg.norm(inertial_vel)
        v_hat_B = dcm.dot(inertial_vel) / vMag
        projArea = area * (facet_dir.dot(v_hat_B))
        if projArea > 0:
            drag_force = -0.5 * dens * projArea * coeff * vMag ** 2.0 * v_hat_B
        else:
            drag_force = np.zeros([3, ])
        return drag_force

    # Compare to expected values
    test_val_force = np.zeros([3, ])
    test_val_torque = np.zeros([3, ])
    for i in range(len(scAreas)):
        val_force_i = checkFacetDragForce(densData[0], scAreas[i], scCoeff[i], B_normals[i], attData[0], velData[0])
        test_val_force += val_force_i
        test_val_torque += np.cross(B_locations[i], val_force_i)

    assert len(densData) > 0, "FAILED:  ExpAtmo failed to pull any logged data"
    np.testing.assert_allclose(dragDataForce_B[1, 1:4], test_val_force, atol=1e-06)
    np.testing.assert_allclose(dragTorqueData[1, 1:4], test_val_torque, atol=1e-06)


test_shadow = [([1.0, 1.0], np.array([2.0, 2.0]), [np.array([0, 0, -1]), np.array([0, -1, 0])], [np.array([0, 0, 0.1]), np.array([0, 0.1, 0])]),
               ([1.0, 1.0], np.array([2.0, 4.0]), [np.array([0, 0, -1]), np.array([0, -1, 0])], [np.array([0, 0, 0.1]), np.array([0, 0.1, 0])]),
               ([1.0, 1.0], np.array([2.0, 2.0]), [np.array([0, 0, -1]), np.array([0, -1, 0])], [np.array([0, 0, 0.4]), np.array([0, 0.4, 0])]),
]
test_shadow = [(areas, coeffs, normals, [cubeFaceLocation(n, a, loc) for n, a, loc in zip(normals, areas, locations)])
               for areas, coeffs, normals, locations in test_shadow]

@pytest.mark.parametrize("scAreas, scCoeff, B_normals, B_locations", test_shadow)
def test_ShadowCalculation(scAreas, scCoeff, B_normals, B_locations):
    """Cube-face panels both facing away from the flow: drag/torque must be exactly zero."""

    ## Simulation initialization
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()

    dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(10.)  # [s]
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"

    simpleNavObj = simpleNav.SimpleNav()
    simpleNavObj.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Initialize new atmosphere and drag model, add them to task
    newAtmo = exponentialAtmosphere.ExponentialAtmosphere()
    newAtmo.ModelTag = "ExpAtmo"
    newAtmo.addSpacecraftToModel(scObject.scStateOutMsg)

    newDrag = facetDragSelfOcclusionPolygonClippingEffector.FacetDragSelfOcclusionPolygonClippingEffector()
    newDrag.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    newDrag.ModelTag = "FacetDragSelfOcclusion"
    newDrag.atmoDensInMsg.subscribeTo(newAtmo.envOutMsgs[0])

    scObject.addDynamicEffector(newDrag)

    try:
        for ind in range(0, len(scAreas)):
            newDrag.addFacet(scAreas[ind], scCoeff[ind], B_normals[ind], B_locations[ind])
    except:
        pytest.fail("ERROR: FacetDragSelfOcclusion unit test failed while setting facet parameters.")

    # clear prior gravitational body and SPICE setup definitions
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()

    planet.isCentralBody = True  # ensure this is the central gravitational body
    # attach gravity model to spacecraft
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

    # setup orbit and simulation time
    r_eq = 6371 * 1000.0  # [m]
    refBaseDens = 1.217  # [kg/m^3]
    refScaleHeight = 8500.0  # [m]

    # Set base density, equatorial radius, scale height in Atmosphere
    newAtmo.baseDensity = refBaseDens
    newAtmo.scaleHeight = refScaleHeight
    newAtmo.planetRadius = r_eq

    rN = np.array([r_eq + 200.0e3, 0, 0])  # [m]
    vN = np.array([0, 7.788e3, 0])  # [m/s]
    sig_BN = np.array([0, 0, 0])  # [-] MRP

    # initialize Spacecraft States with the initialization variables
    scObject.hub.r_CN_NInit = rN  # m - r_CN_N
    scObject.hub.v_CN_NInit = vN  # m - v_CN_N
    scObject.hub.sigma_BNInit = sig_BN

    simulationTime = macros.sec2nano(10.)

    # add BSK objects to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)
    scSim.AddModelToTask(simTaskName, newAtmo)
    scSim.AddModelToTask(simTaskName, newDrag)

    # setup logging
    dataLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, dataLog)
    atmoLog = newAtmo.envOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, atmoLog)
    newDragLog = newDrag.logger(["forceExternal_B", "torqueExternalPntB_B"])
    scSim.AddModelToTask(simTaskName, newDragLog)

    # initialize Simulation
    scSim.InitializeSimulation()

    # configure a simulation stop time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # Retrieve logged data
    dragDataForce_B = simHelpers.addTimeColumn(newDragLog.times(), newDragLog.forceExternal_B)
    dragTorqueData = simHelpers.addTimeColumn(newDragLog.times(), newDragLog.torqueExternalPntB_B)
    densData = atmoLog.neutralDensity
    np.set_printoptions(precision=16)

    assert len(densData) > 0, "FAILED:  ExpAtmo failed to pull any logged data"
    for ind in range(1, len(densData)):
        np.testing.assert_allclose(dragDataForce_B[ind, 1:4], [0, 0, 0], atol=1e-11)
        np.testing.assert_allclose(dragTorqueData[ind, 1:4], [0, 0, 0], atol=1e-11)


if __name__ == "__main__":
    test_facetDragSelfOcclusionInertialVelocity()
    test_facetDragSelfOcclusionFullOcclusion()
    test_facetDragSelfOcclusionPartialOcclusion()
    test_facetDragSelfOcclusionNoOcclusionWhenSeparated()
    test_facetDragSelfOcclusionAtmosphereRelativeVelocityWhenWindLinked()
    test_facetDragSelfOcclusionSignConvention_parallelUpstream()
    test_facetDragSelfOcclusionSignConvention_parallelDownstream()
    test_facetDragSelfOcclusionSignConvention_tiltedFacetCrossesDepth()
    test_facetDragSelfOcclusionSignConvention_partialOverlapPartialUpstream()
    test_facetDragSelfOcclusionSignConvention_completeOverlapLargerShadower()
    test_facetDragSelfOcclusionSignConvention_zeroProjectedArea()
    test_DragCalculation(*test_drag[0])
    test_ShadowCalculation(*test_shadow[0])
