/*
 ISC License

 Copyright (c) 2026, PIC4SeR & AVS Lab, Politecnico di Torino & Argotec S.R.L., University of Colorado Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

#include "simulation/dynamics/_GeneralModuleFiles/facetSelfOcclusionEffectorBase.h"
#include "architecture/utilities/avsEigenMRP.h"

#include <algorithm>
#include <cmath>

using facetSelfOcclusion::FacetProjection;
using facetSelfOcclusion::FacetVisibility;
using facetSelfOcclusion::ProjectedFacet;
using facetSelfOcclusion::geometryTolerance;

/*! Constructor */
FacetSelfOcclusionEffectorBase::FacetSelfOcclusionEffectorBase()
{
    this->isAttachableToStateEffector = true;
}

/*! Set the angle within which the visibility of the previous source direction is reused.

 @param tolerance [rad] cache tolerance angle; 0 recomputes the visibility at every force evaluation
 */
void FacetSelfOcclusionEffectorBase::setDirectionCacheTolerance(double tolerance)
{
    if (tolerance < 0.0) {
        this->bskLogger.bskError("facetSelfOcclusionEffector.setDirectionCacheTolerance was given a negative "
                                 "tolerance %f.",
                                 tolerance);
    }
    this->directionCacheTolerance = tolerance;
    this->invalidateVisibilityCache();
}

/*! Validate and store the rectangle of one facet.

 Each facet is a planar rectangle centered at ``r_CopB_B`` with normal ``nHat_B``. ``width`` and
 ``height`` must either both be left at the sentinel value ``-1.0``, in which case the facet is a
 square of side ``sqrt(area)``, or both be given explicitly, in which case ``width * height`` must
 equal ``area`` so the rectangle stays consistent with the physical facet area. Flux base classes call
 this method from their ``addFacet()`` and store the facet coefficients when it succeeds.

 @param area [m^2] facet area
 @param nHat_B [-] facet unit normal in B frame components
 @param r_CopB_B [m] facet center location relative to B in B frame components
 @param width [m] rectangle width, or -1.0
 @param height [m] rectangle height, or -1.0
 @return true if the facet was stored
 */
bool FacetSelfOcclusionEffectorBase::addFacetGeometry(double area,
                                                      const Eigen::Vector3d& nHat_B,
                                                      const Eigen::Vector3d& r_CopB_B,
                                                      double width,
                                                      double height)
{
    if (area <= 0.0) {
        this->bskLogger.bskError("facetSelfOcclusionEffector was given a non-positive facet area %f.", area);
    }
    if (nHat_B.norm() < geometryTolerance) {
        this->bskLogger.bskError("facetSelfOcclusionEffector was given a zero-norm normal.");
    }
    if (!nHat_B.isUnitary(1e-6)) {  // [-] unit-norm tolerance
        this->bskLogger.bskError("facetSelfOcclusionEffector was given a non-unit normal %f, %f, %f.",
                                 nHat_B[0],
                                 nHat_B[1],
                                 nHat_B[2]);
    }
    bool widthIsDefault = (width == -1.0);
    bool heightIsDefault = (height == -1.0);
    if (widthIsDefault != heightIsDefault) {
        this->bskLogger.bskError("facetSelfOcclusionEffector requires width and height to either both be given "
                                 "explicitly, or both be left at the default -1.0 (deriving a square side from "
                                 "area); got %f, %f.",
                                 width,
                                 height);
    }
    if (!widthIsDefault) {
        if (width <= 0.0 || height <= 0.0) {
            this->bskLogger.bskError("facetSelfOcclusionEffector was given a non-positive facet width/height %f, %f.",
                                     width,
                                     height);
        }
        double geometricArea = width * height;  // [m^2]
        if (std::fabs(geometricArea - area) > 1e-9 * std::max(1.0, area)) {  // [-] relative tolerance
            this->bskLogger.bskError("facetSelfOcclusionEffector was given width*height (%f) inconsistent with "
                                     "area (%f).",
                                     geometricArea,
                                     area);
        }
    }
    this->facetGeometry.areas.push_back(area);
    this->facetGeometry.nHat_B.push_back(nHat_B.normalized());
    this->facetGeometry.r_CopB_B.push_back(r_CopB_B);
    this->facetGeometry.halfEdge1_B.push_back(Eigen::Vector3d::Zero());
    this->facetGeometry.halfEdge2_B.push_back(Eigen::Vector3d::Zero());
    this->facetWidths.push_back(width);
    this->facetHeights.push_back(height);
    this->numFacets++;
    this->geometryBuilt = false;
    this->invalidateVisibilityCache();
    return true;
}

/*! Link the effector to the hub states. The hub attitude is linked here, and the translational state
 the flux model needs is linked by linkInSourceStates().

 The state names are only set when the effector is attached to the hub. When it is attached to a
 state effector they stay empty and linkInProperties() provides the parent frame instead.

 @param states dynamic parameter manager
 */
void FacetSelfOcclusionEffectorBase::linkInStates(DynParamManager& states)
{
    if (this->isAttachedToHub()) {
        this->hubSigma = states.getStateObject(this->stateNameOfSigma);
        if (this->hubSigma == nullptr) {
            this->bskLogger.bskError("facetSelfOcclusionEffector.linkInStates could not find the hub attitude state.");
        }
        this->linkInSourceStates(states);
    }
    this->validateConfiguration();
}

/*! Link the effector to its parent frame properties, used when attached to a state effector. The
 parent attitude is linked here, and the translational property the flux model needs is linked by
 linkInSourceProperties().

 @param properties dynamic parameter manager
 */
void FacetSelfOcclusionEffectorBase::linkInProperties(DynParamManager& properties)
{
    this->inertialAttitudeProperty = properties.getPropertyReference(this->propName_inertialAttitude);
    this->linkInSourceProperties(properties);
    this->validateConfiguration();
}

/*! @return true when the effector is attached to the hub rather than to a state effector */
bool FacetSelfOcclusionEffectorBase::isAttachedToHub() const
{
    return !this->stateNameOfSigma.empty();
}

/*! Compute the DCM from the inertial frame to the effector frame B from the hub attitude state, or
 from the parent attitude property when attached to a state effector.

 @return [-] DCM [BN]
 */
Eigen::Matrix3d FacetSelfOcclusionEffectorBase::computeDcm_BN() const
{
    Eigen::MRPd sigma_BN = this->isAttachedToHub() ? Eigen::MRPd(this->hubSigma->getState().data())
                                                   : Eigen::MRPd(this->inertialAttitudeProperty->data());
    return sigma_BN.toRotationMatrix().transpose();
}

/*! Build the in-plane rectangle half-edges of every facet.

 The facet plane is spanned by an orthonormal basis (e1, e2) built from the facet normal. The stored
 edges are the half-extents 0.5 * width * e1 and 0.5 * height * e2, so the corners are the center
 plus or minus both half-edges.
 */
void FacetSelfOcclusionEffectorBase::buildFacetRectangles()
{
    if (this->geometryBuilt) {
        return;
    }
    for (uint64_t i = 0; i < this->numFacets; ++i) {
        const Eigen::Vector3d& nHat = this->facetGeometry.nHat_B[i];
        Eigen::Vector3d ref = (std::fabs(nHat[0]) < 0.9) ? Eigen::Vector3d(1.0, 0.0, 0.0)
                                                         : Eigen::Vector3d(0.0, 1.0, 0.0);
        Eigen::Vector3d e1 = ref - nHat * ref.dot(nHat);
        e1.normalize();
        Eigen::Vector3d e2 = nHat.cross(e1);

        double side = std::sqrt(this->facetGeometry.areas[i]);  // [m]
        double width = (this->facetWidths[i] < 0.0) ? side : this->facetWidths[i];  // [m]
        double height = (this->facetHeights[i] < 0.0) ? side : this->facetHeights[i];  // [m]
        this->facetGeometry.halfEdge1_B[i] = 0.5 * width * e1;
        this->facetGeometry.halfEdge2_B[i] = 0.5 * height * e2;
    }
    this->geometryBuilt = true;
}

/*! Orthographically project every facet rectangle onto the plane normal to the source direction.

 Besides the projected corners, this computes each facet's projected area, signed source cosine,
 affine depth plane and projected bounding box, so that occlusion algorithms can share them.

 @param sourceDirection_B [-] unit vector from B toward the flux source
 @param projection output projection
 */
void FacetSelfOcclusionEffectorBase::projectFacets(const Eigen::Vector3d& sourceDirection_B,
                                                   FacetProjection& projection) const
{
    projection.dHat_B = sourceDirection_B;
    Eigen::Vector3d ref = (std::fabs(sourceDirection_B[0]) < 0.9) ? Eigen::Vector3d(1.0, 0.0, 0.0)
                                                                  : Eigen::Vector3d(0.0, 1.0, 0.0);
    projection.p1_B = ref - sourceDirection_B * ref.dot(sourceDirection_B);
    projection.p1_B.normalize();
    projection.p2_B = sourceDirection_B.cross(projection.p1_B);
    const Eigen::Vector3d& p1 = projection.p1_B;
    const Eigen::Vector3d& p2 = projection.p2_B;

    projection.facets.clear();
    projection.facets.reserve(this->numFacets);
    for (uint64_t idx = 0; idx < this->numFacets; ++idx) {
        const Eigen::Vector3d& center = this->facetGeometry.r_CopB_B[idx];
        const Eigen::Vector3d& e1 = this->facetGeometry.halfEdge1_B[idx];
        const Eigen::Vector3d& e2 = this->facetGeometry.halfEdge2_B[idx];
        const Eigen::Vector3d& nHat = this->facetGeometry.nHat_B[idx];
        const Eigen::Vector3d corners[4] = {center - e1 - e2, center + e1 - e2, center + e1 + e2, center - e1 + e2};

        ProjectedFacet facet;
        facet.pts.reserve(4);
        for (const auto& corner : corners) {
            facet.pts.emplace_back(corner.dot(p1), corner.dot(p2));
        }
        facet.sourceCosine = nHat.dot(sourceDirection_B);
        facet.projectedArea = this->facetGeometry.areas[idx] * std::fabs(facet.sourceCosine);

        facet.minX = facet.maxX = facet.pts[0].x();
        facet.minY = facet.maxY = facet.pts[0].y();
        for (size_t k = 1; k < facet.pts.size(); ++k) {
            facet.minX = std::min(facet.minX, facet.pts[k].x());
            facet.maxX = std::max(facet.maxX, facet.pts[k].x());
            facet.minY = std::min(facet.minY, facet.pts[k].y());
            facet.maxY = std::max(facet.maxY, facet.pts[k].y());
        }

        // Depth, measured along the flux propagation direction -dHat, of the point on the facet plane
        // that projects to (x, y): depth(x, y) = (x (n.p1) + y (n.p2) - n.c) / (n.dHat). The point
        // with the smallest depth is the closest to the source. Undefined for an edge-on facet.
        if (std::fabs(facet.sourceCosine) > geometryTolerance) {
            facet.depthA = nHat.dot(p1) / facet.sourceCosine;
            facet.depthB = nHat.dot(p2) / facet.sourceCosine;
            facet.depthC = -nHat.dot(center) / facet.sourceCosine;
        }
        projection.facets.push_back(std::move(facet));
    }
}

/*! Force the next force evaluation to recompute the facet visibility. Derived classes call this
 whenever a parameter that changes the visibility result is modified. */
void FacetSelfOcclusionEffectorBase::invalidateVisibilityCache()
{
    this->visibilityCacheValid = false;
}

/*! Compute the total force and torque about B, including facet self-occlusion.

 Each facet facing the source contributes the force of the flux model on its visible projected area,
 applied at the center of pressure of its visible region. The visibility only depends on the source
 direction, so it is reused while the direction stays within the cache tolerance; the facet cosines,
 the pressure and the total force scale are recomputed at every call.

 @param integTime [s] integration time
 @param timeStep [s] integration time step
 */
void FacetSelfOcclusionEffectorBase::computeForceTorque(double integTime, double timeStep)
{
    this->forceExternal_B.setZero();
    this->torqueExternalPntB_B.setZero();

    double fluxPressure = 0.0;  // [Pa]
    if (!this->updateSourceDirection(this->dHat_B, fluxPressure)) {
        return;
    }

    this->buildFacetRectangles();

    bool reuseCache = false;
    if (this->directionCacheTolerance > 0.0 && this->visibilityCacheValid
        && this->visibilityCache.size() == this->numFacets) {
        reuseCache = (this->cachedDirection_B.dot(this->dHat_B) >= std::cos(this->directionCacheTolerance));
    }
    if (!reuseCache) {
        FacetProjection projection;
        this->projectFacets(this->dHat_B, projection);
        this->visibilityCache.assign(this->numFacets, FacetVisibility());
        this->computeFacetVisibility(projection, this->visibilityCache);
        this->cachedDirection_B = this->dHat_B;
        this->visibilityCacheValid = true;
    }

    Eigen::Vector3d totalForce_B = Eigen::Vector3d::Zero();
    Eigen::Vector3d totalTorquePntB_B = Eigen::Vector3d::Zero();
    for (uint64_t i = 0; i < this->numFacets; ++i) {
        // Recomputed at every call rather than cached: the cosine can cross zero well within the
        // cache tolerance, and the force law may depend on it.
        double cosTheta = this->facetGeometry.nHat_B[i].dot(this->dHat_B);
        if (cosTheta <= 0.0) {
            continue;
        }
        double exposure = this->visibilityCache[i].exposure;
        if (exposure <= geometryTolerance) {
            continue;
        }
        double visibleArea = this->facetGeometry.areas[i] * cosTheta * exposure;  // [m^2]
        Eigen::Vector3d facetForce_B = this->computeFacetForce(i, visibleArea, cosTheta, this->dHat_B, fluxPressure);
        totalForce_B += facetForce_B;
        totalTorquePntB_B += this->visibilityCache[i].r_CopB_B.cross(facetForce_B);
    }

    double scale = this->totalForceScale();  // [-]
    this->forceExternal_B = scale * totalForce_B;
    this->torqueExternalPntB_B = scale * totalTorquePntB_B;
}
