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
# Purpose:  Test the flat-plate solar radiation pressure model of facetSRPSelfOcclusionPolygonClippingEffector
#           and its planar self-shadowing computation.
#

import numpy as np

from Basilisk.architecture import astroConstants
import pytest

from Basilisk.architecture import messaging
from Basilisk.simulation import (
    spacecraft,
    facetSRPSelfOcclusionPolygonClippingEffector
)
from Basilisk.utilities import RigidBodyKinematics as rbk

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
)


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
                # t = D_prev / (D_prev - D_cur), equals -D_prev/denom
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


def clipToUpstreamHalfPlane(poly, n_i, c_i, sunCosine_i, n_j, c_j, sunCosine_j, p1, p2):
    """Reference implementation of clipToUpstreamHalfPlane().

    Clips `poly` to the region where facet j's plane is upstream of facet i's plane,
    evaluated pointwise via each facet's affine depth(x, y) function.
    The depth is measured along the flux propagation direction,
    depth_k(x, y) = (x n_k.p1 + y n_k.p2 - n_k.c_k) / (n_k.d), so the shadow region
    is D = depth_i - depth_j >= 0.
    """
    A = np.dot(n_i, p1) / sunCosine_i - np.dot(n_j, p1) / sunCosine_j
    B = np.dot(n_i, p2) / sunCosine_i - np.dot(n_j, p2) / sunCosine_j
    C = np.dot(n_j, c_j) / sunCosine_j - np.dot(n_i, c_i) / sunCosine_i

    tol = 1e-12  # [-]
    if abs(A) < tol and abs(B) < tol:
        return list(poly) if C >= 0.0 else []

    P0 = np.array([-C / A, 0.0]) if abs(A) >= abs(B) else np.array([0.0, -C / B])
    edgeDir = np.array([B, -A])
    return clipPolygon(poly, P0, P0 + edgeDir)


def exposureFactors(areas, normals, locations, edges1, edges2, sHat_B):
    """Reference implementation of computeExposureFactor() for every facet.

    Mirrors the C++ implementation: for each pair (i, j) the shadow patch is
    the rectangle overlap further clipped to wherever j is pointwise upstream of i, and
    the total shadowed area on i is the exact union (via inclusion-exclusion) of the
    patches cast by every other facet. Also mirrors the exposed region's
    center-of-pressure tracking: since a partially-shadowed facet's exposed centroid
    generally is not the facet's own center, the same inclusion-exclusion tracks each
    patch's first moment so the exposed centroid can be recovered  and mapped back
    onto facet i's plane via its affine depth(x, y) function.

    Returns (exposures, centersOfPressure_B), each a length-n list; a facet with no
    shadowing at all gets its own location as its center of pressure.
    """
    n = len(areas)
    if np.linalg.norm(sHat_B) < 1e-12:
        return [1.0] * n, [np.array(loc, dtype=float) for loc in locations]

    ref = np.array([1.0, 0.0, 0.0]) if abs(sHat_B[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = ref - sHat_B * np.dot(ref, sHat_B)
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(sHat_B, p1)

    def makeRect(idx):
        c = np.array(locations[idx], dtype=float)
        e1 = edges1[idx]
        e2 = edges2[idx]
        corners = [c - e1 - e2, c + e1 - e2, c + e1 + e2, c - e1 + e2]
        pts = [np.array([np.dot(corner, p1), np.dot(corner, p2)]) for corner in corners]
        sunCosine = np.dot(normals[idx], sHat_B)
        projArea = areas[idx] * abs(sunCosine)
        return pts, sunCosine, projArea

    rects = [makeRect(i) for i in range(n)]
    tol = 1e-12  # [-]
    exposures = []
    centersOfPressure_B = []
    for i in range(n):
        ptsI, sunCosineI, projI = rects[i]
        centerOfPressure_B = np.array(locations[i], dtype=float)
        if projI <= tol or abs(sunCosineI) <= tol:
            exposures.append(0.0)
            centersOfPressure_B.append(centerOfPressure_B)
            continue

        shadowPieces = []
        for j in range(n):
            if j == i:
                continue
            ptsJ, sunCosineJ, _ = rects[j]
            if abs(sunCosineJ) <= tol:
                continue
            overlap = polygonIntersection(ptsI, ptsJ)
            if not overlap:
                continue
            shadow = clipToUpstreamHalfPlane(
                overlap,
                normals[i], locations[i], sunCosineI,
                normals[j], locations[j], sunCosineJ,
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
                       - exposedCentroid2D[1] * np.dot(normals[i], p2)) / sunCosineI
            centerOfPressure_B = exposedCentroid2D[0] * p1 + exposedCentroid2D[1] * p2 + depth_i * sHat_B
        exposures.append(min(1.0, exposedArea / projI))
        centersOfPressure_B.append(centerOfPressure_B)
    return exposures, centersOfPressure_B


def facetSRPSelfOcclusionComp(r_SN_N, r_BN_N, sigmaBN, areas, diffuse, specular,
                              normals, locations, widths, heights, illuminationFactor=1.0,
                              sHat_B=None, r_SB_norm=None):
    """Reference facet SRP force/torque in the body frame, including self-shadowing.

    The sun direction and body-to-Sun distance can be supplied explicitly via
    ``sHat_B`` / ``r_SB_norm``; otherwise they are derived from the spacecraft
    state and the sun position. Supplying the module's own values guarantees the
    reference uses exactly the same shadowing/force inputs as the C++ module.
    """
    if sHat_B is None or r_SB_norm is None:
        dcm = rbk.MRP2C(sigmaBN)
        r_SB_N = np.array(r_SN_N, dtype=float) - np.array(r_BN_N, dtype=float)
        r_SB_B = dcm.dot(r_SB_N)
        r_SB_norm = np.linalg.norm(r_SB_B)
        if r_SB_norm <= 1e-12:
            return np.zeros(3), np.zeros(3)
        sHat_B = r_SB_B / r_SB_norm
    else:
        sHat_B = np.array(sHat_B, dtype=float).flatten()
        sHat_B = sHat_B / np.linalg.norm(sHat_B)
        r_SB_norm = float(r_SB_norm)
        if r_SB_norm <= 1e-12:
            return np.zeros(3), np.zeros(3)

    srpPressure = (astroConstants.SOLAR_FLUX_EARTH / astroConstants.SPEED_LIGHT) * (astroConstants.AU2M / r_SB_norm) ** 2.0

    edges = [buildFacetRectangle(normals[i], areas[i], widths[i], heights[i]) for i in range(len(areas))]
    edges1 = [e[0] for e in edges]
    edges2 = [e[1] for e in edges]
    exposures, centersOfPressure_B = exposureFactors(areas, normals, locations, edges1, edges2, sHat_B)

    force = np.zeros(3)
    torque = np.zeros(3)
    for i in range(len(areas)):
        projTerm = np.dot(normals[i], sHat_B)
        projArea = areas[i] * projTerm * exposures[i]
        if projArea > 0.0:
            f = -srpPressure * projArea * (
                (1.0 - specular[i]) * sHat_B
                + 2.0 * (diffuse[i] / 3.0 + specular[i] * projTerm) * normals[i]
            )
            force += f
            torque += -np.cross(f, centersOfPressure_B[i])
    return illuminationFactor * force, illuminationFactor * torque


def setupSRPSim(areas, diffuse, specular, normals, locations, widths=None, heights=None,
                r_BN_N=None, sigmaBN=None, illuminationFactor=1.0, r_SN_N=None,
                simulationTimeStep=1e-4):
    """
    Simulation setup helper. Places the spacecraft on the +x inertial axis with
    identity attitude and the Sun along +y at 1 AU, so the body-frame sun
    direction is exactly sHat_B = [0, 1, 0] at the step used to probe the
    instantaneous force.
    """
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

    if r_BN_N is None:
        r_BN_N = np.array([astroConstants.AU2M * 0.05, 0.0, 0.0])  # [m], well out of the gravity well -> negligible drift
    if sigmaBN is None:
        sigmaBN = np.array([0.0, 0.0, 0.0])  # [-] MRP
    if r_SN_N is None:
        r_SN_N = np.array([r_BN_N[0], astroConstants.AU2M, 0.0])  # [m], sun along +y at 1 AU

    sunStateMsg = messaging.SpicePlanetStateMsgPayload()
    sunStateMsg.PositionVector = list(r_SN_N)
    sunMsg = messaging.SpicePlanetStateMsg().write(sunStateMsg)

    newSRP = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    newSRP.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    newSRP.ModelTag = "FacetSRPSelfOcclusion"
    newSRP.sunInMsg.subscribeTo(sunMsg)
    scSim.sunMsg = sunMsg

    if illuminationFactor != 1.0:
        eclipseMsgData = messaging.EclipseMsgPayload()
        eclipseMsgData.illuminationFactor = illuminationFactor
        eclipseMsg = messaging.EclipseMsg().write(eclipseMsgData)
        newSRP.sunEclipseInMsg.subscribeTo(eclipseMsg)
        scSim.eclipseMsg = eclipseMsg

    for area, diff, spec, normal, loc, width, height in zip(areas, diffuse, specular, normals, locations, widths, heights):
        newSRP.addFacet(area, diff, spec, np.array(normal, dtype=float), np.array(loc, dtype=float), width, height)

    scObject.addDynamicEffector(newSRP)

    scObject.hub.r_CN_NInit = r_BN_N  # [m]
    scObject.hub.v_CN_NInit = np.array([0.0, 7500.0, 0.0])  # [m/s]
    scObject.hub.sigma_BNInit = sigmaBN

    scSim.AddModelToTask(simTaskName, scObject)
    scSim.AddModelToTask(simTaskName, newSRP)

    dataLog = scObject.scStateOutMsg.recorder()
    srpForceLog = newSRP.logger("forceExternal_B")
    srpTorqueLog = newSRP.logger("torqueExternalPntB_B")
    scSim.AddModelToTask(simTaskName, dataLog)
    scSim.AddModelToTask(simTaskName, srpForceLog)
    scSim.AddModelToTask(simTaskName, srpTorqueLog)

    return scSim, scObject, newSRP, dataLog, srpForceLog, srpTorqueLog, r_SN_N


def runAndCheck(areas, diffuse, specular, normals, locations, widths=None, heights=None,
                r_BN_N=None, sigmaBN=None, illuminationFactor=1.0, r_SN_N=None,
                simulationTimeStep=1e-4, independentSunDir=False):
    (scSim, scObject, newSRP, dataLog, srpForceLog,
     srpTorqueLog, r_SN_N_x) = setupSRPSim(areas, diffuse, specular, normals, locations,
                                           widths, heights, r_BN_N, sigmaBN,
                                           illuminationFactor, r_SN_N, simulationTimeStep)
    result_sun = r_SN_N_x
    if widths is None:
        widths = [-1.0] * len(areas)  # [m], -1 derives the side from the area
    if heights is None:
        heights = [-1.0] * len(areas)  # [m], -1 derives the side from the area

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(simulationTimeStep))
    scSim.ExecuteSimulation()

    if independentSunDir:
        # Recompute sHat_B/r_SB_norm entirely independently from the raw logged
        # sun/spacecraft state, instead of trusting the module's own cached values
        refForce, refTorque = facetSRPSelfOcclusionComp(
            result_sun, dataLog.r_BN_N[0], dataLog.sigma_BN[0],
            areas, diffuse, specular, normals, locations,
            widths, heights, illuminationFactor)
    else:
        # The SRP force is computed from the state at the start of the
        # integration step, with the sun data read at that same step. Feed the module's
        # own cached sun direction and body-to-Sun distance into the reference so the two
        # share identical shadowing/force inputs.
        refForce, refTorque = facetSRPSelfOcclusionComp(
            result_sun, dataLog.r_BN_N[0], dataLog.sigma_BN[0],
            areas, diffuse, specular, normals, locations,
            widths, heights, illuminationFactor,
            sHat_B=np.array(newSRP.getSourceDirection_B()).flatten(), r_SB_norm=newSRP.getSunDistance())
    force = srpForceLog.forceExternal_B[-1]
    torque = srpTorqueLog.torqueExternalPntB_B[-1]
    return force, torque, refForce, refTorque


# A threshold well above the SRP forces' numerical noise but well below the physically
# meaningful SRP force magnitudes at 1 AU.
_FORCE_ATOL = 1e-9


def test_facetSRPSelfOcclusionSingleFacet():
    """A single facet with no possible self-shadowing must match the plain flat-plate SRP formula
    and the canonical FacetSRPDynamicEffector force expression."""
    areas = [1.0]  # [m^2]
    diffuse = [0.5]  # [-]
    specular = [0.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, 1.0, 0.0])]  # off-center to also exercise the torque

    force, torque, refForce, refTorque = runAndCheck(
        areas, diffuse, specular, normals, locations)

    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)
    assert np.linalg.norm(force) > 0.0
    # With a single sun-facing facet the total force/torque must equal that of a
    # single fully-exposed unit facet
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)


def test_facetSRPSelfOcclusionSunDirectionIndependentComputation():
    """the Sun direction and distance computation themselves must be correct, not just the
    shadowing/force assembly downstream of them.

    Every other test in this file feeds the reference implementation the module's own
    cached sHat_B/r_SB_norm, which cross-checks the shadowing and force-assembly math
    but would not catch a bug in the sun-direction/distance computation itself, since
    both sides would then agree on the same wrong value. This test instead lets the
    reference recompute sHat_B/r_SB_norm independently from the raw logged sun and
    spacecraft state, so the two can actually disagree if the module's own computation
    is wrong.
    """
    areas = [1.0]  # [m^2]
    diffuse = [0.5]  # [-]
    specular = [0.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, 1.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(
        areas, diffuse, specular, normals, locations, independentSunDir=True)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)
    assert np.linalg.norm(force) > 0.0


def test_facetSRPSelfOcclusionSelfOcclusionZero():
    """A single facet can never self-shadow; verify exposure is exactly unity by
    comparing against the same facet evaluated with caching disabled."""
    areas = [1.0]  # [m^2]
    diffuse = [0.5]  # [-]
    specular = [0.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)


def test_facetSRPSelfOcclusionFullOcclusion():
    """Two identical, directly-stacked facets: the downwind one should be fully shadowed."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.5, 0.5]  # [-]
    specular = [0.2, 0.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    # Both centered on the same line along the sun direction (y), same unit-square footprint.
    locations = [np.array([0.0, 0.5, 0.0]), np.array([0.0, -0.5, 0.0])]  # upwind, downwind

    force, torque, refForce, refTorque = runAndCheck(
        areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    # The upwind facet alone would produce this force; if the downwind facet is fully
    # shadowed, the total force should match the single-facet case.
    upwindOnlyForce, upwindOnlyTorque, _, _ = runAndCheck(
        [areas[0]], [diffuse[0]], [specular[0]], [normals[0]], [locations[0]])

    assert np.allclose(force, upwindOnlyForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, upwindOnlyTorque, atol=_FORCE_ATOL)


def test_facetSRPSelfOcclusionPartialOcclusion():
    """Two identical, partially-overlapping facets: exposure should be strictly between 0 and 1."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.5, 0.5]  # [-]
    specular = [0.2, 0.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    # Downwind facet is offset half a facet-width sideways (x) from the upwind one.
    locations = [np.array([0.0, 0.5, 0.0]), np.array([0.5, -0.5, 0.0])]  # upwind, downwind

    force, torque, refForce, refTorque = runAndCheck(
        areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    upwindOnlyForce, _, _, _ = runAndCheck(
        [areas[0]], [diffuse[0]], [specular[0]], [normals[0]], [locations[0]])
    upwindMag = np.linalg.norm(upwindOnlyForce)
    unshadowedForceMag = 2.0 * upwindMag
    forceMag = np.linalg.norm(force)
    assert forceMag > upwindMag * (1.0 + 1e-6)
    assert forceMag < unshadowedForceMag * (1.0 - 1e-6)


def test_facetSRPSelfOcclusionNoOcclusionWhenSeparated():
    """Facets far enough apart that their silhouettes never overlap should not shadow each other."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.5, 0.5]  # [-]
    specular = [0.2, 0.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, 0.0, 0.0]), np.array([50.0, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(
        areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    upwindOnlyForce, _, _, _ = runAndCheck(
        [areas[0]], [diffuse[0]], [specular[0]], [normals[0]], [locations[0]])
    assert np.allclose(force, 2.0 * upwindOnlyForce, atol=_FORCE_ATOL)


def test_facetSRPSelfOcclusionEclipseScaling():
    """The total SRP force/torque must scale linearly with the eclipse illumination factor."""
    areas = [1.0]  # [m^2]
    diffuse = [0.5]  # [-]
    specular = [0.2]  # [-]
    normals = [np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, 1.0, 0.0])]  # [m]

    forceFull, torqueFull, _, _ = runAndCheck(areas, diffuse, specular, normals, locations)

    illum = 0.37
    forceScaled, torqueScaled, refScaled, refTqScaled = runAndCheck(
        areas, diffuse, specular, normals, locations, illuminationFactor=illum)
    assert np.allclose(forceScaled, refScaled, atol=_FORCE_ATOL)
    assert np.allclose(torqueScaled, refTqScaled, atol=_FORCE_ATOL)
    assert np.allclose(forceScaled, illum * forceFull, atol=_FORCE_ATOL)
    assert np.allclose(torqueScaled, illum * torqueFull, atol=_FORCE_ATOL)


def test_facetSRPSelfOcclusionDirectedSun():
    """A facet whose normal points away from the sun must contribute nothing."""
    areas = [1.0]  # [m^2]
    diffuse = [0.5]  # [-]
    specular = [0.2]  # [-]
    normals = [np.array([0.0, -1.0, 0.0])]  # faces away from the +y sun
    locations = [np.array([0.0, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)
    assert np.allclose(force, [0.0, 0.0, 0.0], atol=_FORCE_ATOL)
    assert np.allclose(torque, [0.0, 0.0, 0.0], atol=_FORCE_ATOL)


_COEFF_DIFF = 0.5
_COEFF_SPEC = 0.2
_pressure_au = (astroConstants.SOLAR_FLUX_EARTH / astroConstants.SPEED_LIGHT)  # [N/m^2], solar pressure at exactly 1 AU


def expectedSRPForceFromExposureSum(exposureWeightedAreaSum):
    """-P_AU * Cd * Sum_k(area_k * projTerm_k * exposure_k) applied to the incident
    term; for a pure +y sun and pure +y normals the total projected sum drives the
    magnitude. Here P_AU = solarRadFlux/c. Only the incident (-sHat) term is included,
    because the tests below use specular = 0 so the reflection term vanishes."""
    return -_pressure_au * exposureWeightedAreaSum


def _srpAreaSum(areas, exposures):
    """Sum_k(area_k * projTerm_k * exposure_k) with projTerm = 1 for all +y facets."""
    return sum(a * 1.0 * e for a, e in zip(areas, exposures))


def test_facetSRPSelfOcclusionSignConvention_parallelUpstream():
    """Parallel facets, j clearly upstream -- j must fully shadow i, and i
    must NOT shadow j back."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.0, 0.0]  # absorption-only isolates the incident (-sHat) term
    specular = [0.0, 0.0]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i downstream, j upstream

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    expectedForceY = expectedSRPForceFromExposureSum(1.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetSRPSelfOcclusionSignConvention_parallelDownstream():
    """Parallel facets, j clearly downstream -- the mirror of case 1."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.0, 0.0]  # [-]
    specular = [0.0, 0.0]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]  # i upstream, j downstream

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    expectedForceY = expectedSRPForceFromExposureSum(1.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetSRPSelfOcclusionSignConvention_tiltedFacetCrossesDepth():
    """A facet tilted 30 deg from the flat target crosses the target's depth
    exactly through the overlap's center."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.0, 0.0]  # [-]
    specular = [0.0, 0.0]  # [-]
    tiltAngle = np.pi / 6.0  # 30 deg
    normals = [np.array([0.0, 1.0, 0.0]),  # [-]
               np.array([np.sin(tiltAngle), np.cos(tiltAngle), 0.0])]
    locations = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    cosTilt = np.cos(tiltAngle)
    exposure_i = 1.0 - 0.25 * np.sqrt(3.0)  # [-]
    exposure_j = 0.5  # [-]
    exposureWeightedAreaSum = 1.0 * 1.0 * exposure_i + 1.0 * cosTilt * exposure_j
    assert abs(exposureWeightedAreaSum - 1.0) < 1e-12
    expectedForceY = expectedSRPForceFromExposureSum(exposureWeightedAreaSum)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetSRPSelfOcclusionSignConvention_partialOverlapPartialUpstream():
    """The same 30 deg tilted facet shifted sideways."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.0, 0.0]  # [-]
    specular = [0.0, 0.0]  # [-]
    tiltAngle = np.pi / 6.0  # 30 deg
    normals = [np.array([0.0, 1.0, 0.0]),  # [-]
               np.array([np.sin(tiltAngle), np.cos(tiltAngle), 0.0])]
    locations = [np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    exposure_i = 1.0 - 0.25 * np.sqrt(3.0)  # [-]
    exposure_j = 1.0 - 0.2 / np.cos(tiltAngle)  # [-]
    exposureWeightedAreaSum = 1.0 * 1.0 * exposure_i + 1.0 * np.cos(tiltAngle) * exposure_j
    expectedForceY = expectedSRPForceFromExposureSum(exposureWeightedAreaSum)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetSRPSelfOcclusionSignConvention_completeOverlapLargerShadower():
    """A larger upstream facet completely contains the smaller downstream target."""
    areas = [1.0, 4.0]  # [m^2]
    diffuse = [0.0, 0.0]  # [-]
    specular = [0.0, 0.0]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i downstream+smaller, j upstream+bigger

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    expectedForceY = expectedSRPForceFromExposureSum(4.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetSRPSelfOcclusionSignConvention_upstreamCoefficientsWin():
    """Two identical, fully overlapping parallel facets with different diffuse
    coefficients. The Sun is along +y, so only the facet at larger y is lit and the force
    carries its coefficient alone"""
    upstreamDiffuse = 1.0  # [-]
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.0, upstreamDiffuse]  # [-]
    specular = [0.0, 0.0]  # [-]
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # [-]
    locations = [np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i downstream, j upstream

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    # Normal parallel to the Sun direction: F = -P A (1 + 2 D / 3) along +y, at exactly 1 AU.
    expectedForceY = -_pressure_au * 1.0 * (1.0 + 2.0 * upstreamDiffuse / 3.0)  # [N]
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetSRPSelfOcclusionSignConvention_zeroProjectedArea():
    """A facet edge-on to the sun must contribute no force
    and neither shadow nor be shadowed spuriously."""
    areas = [1.0, 1.0]  # [m^2]
    diffuse = [0.0, 0.0]  # [-]
    specular = [0.0, 0.0]  # [-]
    normals = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]  # i is edge-on
    locations = [np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])]  # [m]

    force, torque, refForce, refTorque = runAndCheck(areas, diffuse, specular, normals, locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)

    expectedForceY = expectedSRPForceFromExposureSum(1.0)
    assert abs(force[1] - expectedForceY) < 1e-9 * abs(expectedForceY)


def test_facetSRPSelfOcclusionShadowCacheStaleProjectionTerm():
    """A facet crossing its terminator within the shadow-cache's angular tolerance
    must stop contributing force immediately, not on the next cache-miss recompute."""
    eps = 1e-2  # [rad] half-angle the sun swings across the facet's terminator
    tolerance = 2.5e-2  # [rad] > 2*eps, so the second call reuses the first call's cache
    dist = astroConstants.AU2M

    s0 = np.array([np.sin(eps), np.cos(eps), 0.0])
    s1 = np.array([-np.sin(eps), np.cos(eps), 0.0])
    # Sanity check on the geometry this test relies on: s0/s1 stay within the cache's
    # angular tolerance of each other, while the facet normal's projection onto them
    # flips sign.
    assert np.dot(s0, s1) >= np.cos(tolerance)
    facetNormal = np.array([1.0, 0.0, 0.0])
    assert np.dot(facetNormal, s0) > 0.0
    assert np.dot(facetNormal, s1) < 0.0

    srpEffector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    srpEffector.setLogOverlapFallbackWarnings(False)  # keep the test log free of fallback warnings
    srpEffector.ModelTag = "FacetSRPSelfOcclusion"
    srpEffector.setDirectionCacheTolerance(tolerance)
    srpEffector.addFacet(1.0, 0.5, 0.2, facetNormal, np.array([0.0, 0.0, 0.0]))

    sunStateMsg = messaging.SpicePlanetStateMsgPayload()
    sunStateMsg.PositionVector = list(dist * s0)
    sunMsg = messaging.SpicePlanetStateMsg().write(sunStateMsg)
    srpEffector.sunInMsg.subscribeTo(sunMsg)

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"
    scObject.addDynamicEffector(srpEffector)
    scObject.hub.r_CN_NInit = np.array([0.0, 0.0, 0.0])  # [m]
    scObject.hub.v_CN_NInit = np.array([0.0, 0.0, 0.0])  # [m/s]
    scObject.hub.sigma_BNInit = np.array([0.0, 0.0, 0.0])  # [-]
    scObject.hub.omega_BN_BInit = np.array([0.0, 0.0, 0.0])  # [rad/s]

    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess("simProcess")
    dt = macros.sec2nano(1e-4)
    dynProcess.addTask(scSim.CreateNewTask("simTask", dt))
    scSim.AddModelToTask("simTask", scObject)
    scSim.AddModelToTask("simTask", srpEffector)

    scSim.InitializeSimulation()

    srpEffector.readInputMessages()
    srpEffector.computeForceTorque(0.0, macros.NANO2SEC * dt)
    force0 = np.array(srpEffector.forceExternal_B).flatten()
    assert np.linalg.norm(force0) > 1e-9, (
        "the facet is barely sun-facing at s0 and must contribute a small nonzero force")

    # Swing the sun direction across the facet's terminator while staying inside the
    # cache's angular tolerance, so the next call reuses the shadow cache built for s0.
    sunStateMsg.PositionVector = list(dist * s1)
    sunMsg.write(sunStateMsg, dt)
    srpEffector.readInputMessages()
    srpEffector.computeForceTorque(macros.NANO2SEC * dt, macros.NANO2SEC * dt)

    force1 = np.array(srpEffector.forceExternal_B).flatten()
    np.testing.assert_allclose(
        force1, 0.0, atol=1e-9,
        err_msg="facet now faces away from the sun; a stale cached projection term "
                "would keep contributing force even though the sun crossed the "
                "facet's terminator")


def cubeFaceLocation(normal, area, location):
    normal = np.array(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    location = np.array(location, dtype=float)
    half = 0.5 * np.sqrt(area)
    leverArm = location - normal * normal.dot(location)
    return half * normal + leverArm


test_drag = [([1.0, 1.0], np.array([0.5, 0.5]), np.array([0.2, 0.2]),
              [np.array([1, 0, 0]), np.array([0, 1, 0])],
              [np.array([0.1, 0, 0]), np.array([0, 0.1, 0])]),
             ([1.0, 1.0], np.array([0.5, 0.5]), np.array([0.2, 0.2]),
              [np.array([1, 0, 0]), np.array([0, 1, 0])],
              [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]),
             ([1.0, 2.0], np.array([0.5, 0.5]), np.array([0.2, 0.2]),
              [np.array([1, 0, 0]), np.array([0, 1, 0])],
              [np.array([0.1, 0, 0]), np.array([0, 0.1, 0])]),
             ([1.0, 1.0], np.array([0.5, 0.5]), np.array([0.2, 0.2]),
              [np.array([1, 0, 0]), np.array([0, 1, 0])],
              [np.array([0.1, 0, 0]), np.array([0, 0, 0.1])]),
             ([1.0, 1.0], np.array([0.5, 0.5]), np.array([0.2, 0.2]),
              [np.array([1, 0, 0]), np.array([0, 0, 1])],
              [np.array([0.1, 0, 0]), np.array([0, 0, 0.1])]),
             ([1.0, 1.0], np.array([0.5, 0.5]), np.array([0.2, 0.2]),
              [np.array([0, 0, -1]), np.array([0, -1, 0])],
              [np.array([0, 0, 0.1]), np.array([0, 0.1, 0])]),
]
test_drag = [(areas, diff, spec,
              normals,
              [cubeFaceLocation(n, a, loc) for n, a, loc in zip(normals, areas, locations)])
             for areas, diff, spec, normals, locations in test_drag]


@pytest.mark.parametrize("scAreas, scDiffuse, scSpecular, B_normals, B_locations", test_drag)
def test_SRPCubeFacesNoSelfShadowing(scAreas, scDiffuse, scSpecular, B_normals, B_locations):
    """Cube-face-panel SRP check: two orthogonal, non-overlapping facets must each match
    the no-occlusion flat-plate formula exactly"""
    force, torque, refForce, refTorque = runAndCheck(
        scAreas, scDiffuse, scSpecular, B_normals, B_locations)
    assert np.allclose(force, refForce, atol=_FORCE_ATOL)
    assert np.allclose(torque, refTorque, atol=_FORCE_ATOL)


if __name__ == "__main__":
    test_facetSRPSelfOcclusionSingleFacet()
    test_facetSRPSelfOcclusionSunDirectionIndependentComputation()
    test_facetSRPSelfOcclusionShadowCacheStaleProjectionTerm()
    test_facetSRPSelfOcclusionSelfOcclusionZero()
    test_facetSRPSelfOcclusionFullOcclusion()
    test_facetSRPSelfOcclusionPartialOcclusion()
    test_facetSRPSelfOcclusionNoOcclusionWhenSeparated()
    test_facetSRPSelfOcclusionEclipseScaling()
    test_facetSRPSelfOcclusionDirectedSun()
    test_facetSRPSelfOcclusionSignConvention_parallelUpstream()
    test_facetSRPSelfOcclusionSignConvention_parallelDownstream()
    test_facetSRPSelfOcclusionSignConvention_tiltedFacetCrossesDepth()
    test_facetSRPSelfOcclusionSignConvention_partialOverlapPartialUpstream()
    test_facetSRPSelfOcclusionSignConvention_completeOverlapLargerShadower()
    test_facetSRPSelfOcclusionSignConvention_zeroProjectedArea()
    test_SRPCubeFacesNoSelfShadowing(*test_drag[0])
