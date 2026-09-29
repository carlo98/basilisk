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
# Purpose:  Configuration validation of facetSRPSelfOcclusionPolygonClippingEffector.
#

import numpy as np
import pytest
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.simulation import facetSRPSelfOcclusionPolygonClippingEffector
from Basilisk.simulation import spacecraft, spinningBodyOneDOFStateEffector
from Basilisk.utilities import SimulationBaseClass, macros


def addTestFacet(effector, area=1.0, nHat_B=(1.0, 0.0, 0.0), width=-1.0, height=-1.0):
    """Add a facet with default flat-plate solar radiation pressure coefficients."""
    nHat_B = np.array(nHat_B, dtype=float)
    r_CopB_B = np.array([0.0, 0.0, 0.0])  # [m]
    effector.addFacet(area, 0.5, 0.2, nHat_B, r_CopB_B, width, height)


@pytest.mark.parametrize("attachToBranch", [False, True], ids=["hub", "branch"])
@pytest.mark.parametrize("validationPath", ["attachment", "reset"])
def test_facetSRPSelfOcclusionPolygonClippingEffector_missingSunInMsg(attachToBranch, validationPath):
    """Hub, branch, and direct Reset paths must reject a missing ``sunInMsg``."""
    simulation = SimulationBaseClass.SimBaseClass()
    process = simulation.CreateNewProcess("process")
    timeStep = macros.sec2nano(0.1)  # [ns]
    process.addTask(simulation.CreateNewTask("task", timeStep))

    spacecraftObject = spacecraft.Spacecraft()
    spacecraftObject.hub.mHub = 100.0  # [kg]
    spacecraftObject.hub.IHubPntBc_B = [[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]]  # [kg m^2]
    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    addTestFacet(effector)
    if attachToBranch:
        spinningBody = spinningBodyOneDOFStateEffector.SpinningBodyOneDOFStateEffector()
        spinningBody.mass = 1.0  # [kg]
        spinningBody.sHat_S = [[1.0], [0.0], [0.0]]  # [-]
        spinningBody.dcm_S0B = [[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]]  # [-]
        spinningBody.IPntSc_S = [[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]]  # [kg m^2]
        spinningBody.addDynamicEffector(effector)
        spacecraftObject.addStateEffector(spinningBody)
    else:
        spacecraftObject.addDynamicEffector(effector)
    simulation.AddModelToTask("task", spacecraftObject)
    simulation.AddModelToTask("task", effector)

    with pytest.raises(BasiliskError, match="sunInMsg"):
        if validationPath == "reset":
            effector.Reset(0)
        else:
            simulation.InitializeSimulation()


def test_facetSRPSelfOcclusionPolygonClippingEffector_nonPositiveArea():
    """addFacet must reject a non-positive facet area."""
    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    with pytest.raises(BasiliskError, match="non-positive facet area"):
        addTestFacet(effector, area=0.0)
    with pytest.raises(BasiliskError, match="non-positive facet area"):
        addTestFacet(effector, area=-1.0)


def test_facetSRPSelfOcclusionPolygonClippingEffector_nonUnitNormal():
    """addFacet must reject a facet normal that is not a unit vector."""
    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    with pytest.raises(BasiliskError, match="non-unit normal"):
        addTestFacet(effector, nHat_B=(2.0, 0.0, 0.0))


def test_facetSRPSelfOcclusionPolygonClippingEffector_inconsistentWidthHeight():
    """addFacet must reject a mix of one explicit and one defaulted dimension, and a
    width*height inconsistent with the supplied area."""
    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    with pytest.raises(BasiliskError, match="both be left at the default"):
        addTestFacet(effector, width=1.0)
    with pytest.raises(BasiliskError, match="inconsistent with area"):
        addTestFacet(effector, width=2.0, height=2.0)


def test_facetSRPSelfOcclusionPolygonClippingEffector_negativeCacheTolerance():
    """setDirectionCacheTolerance must reject a negative angle."""
    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    with pytest.raises(BasiliskError, match="negative"):
        effector.setDirectionCacheTolerance(-1e-3)  # [rad]


@pytest.mark.parametrize("diffuseCoeff, specularCoeff", [(-0.1, 0.2), (1.1, 0.2), (0.5, -0.1), (0.5, 1.1)])
def test_facetSRPSelfOcclusionPolygonClippingEffector_coefficientRange(diffuseCoeff, specularCoeff):
    """addFacet must reject optical coefficients outside [0, 1]."""
    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    with pytest.raises(BasiliskError, match="outside"):
        effector.addFacet(1.0, diffuseCoeff, specularCoeff, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))


def test_facetSRPSelfOcclusionPolygonClippingEffector_overlapFallbackWarningsSwitch():
    """The overlap-fallback warning switch defaults to True and returns the value set."""
    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    assert effector.getLogOverlapFallbackWarnings()
    effector.setLogOverlapFallbackWarnings(False)
    assert not effector.getLogOverlapFallbackWarnings()
    effector.setLogOverlapFallbackWarnings(True)
    assert effector.getLogOverlapFallbackWarnings()
