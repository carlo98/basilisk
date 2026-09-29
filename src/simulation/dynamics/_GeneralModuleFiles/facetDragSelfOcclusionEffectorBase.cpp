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

#include "simulation/dynamics/_GeneralModuleFiles/facetDragSelfOcclusionEffectorBase.h"

using facetSelfOcclusion::geometryTolerance;

/*! Check that the atmospheric density input message is linked.

 @param currentSimNanos [ns] current simulation time
 */
void FacetDragSelfOcclusionEffectorBase::Reset(uint64_t currentSimNanos)
{
    this->validateConfiguration();
}

/*! Read the input messages. Their content is held constant over the following integration step.

 @param currentSimNanos [ns] current simulation time
 */
void FacetDragSelfOcclusionEffectorBase::UpdateState(uint64_t currentSimNanos)
{
    this->readInputMessages();
}

/*! Read the atmosphere and wind input messages into the module buffers.

 A linked but never written message is not treated as valid data: without a valid atmosphere sample
 the effector applies no force, and without a valid wind sample the air is at rest.
 */
void FacetDragSelfOcclusionEffectorBase::readInputMessages()
{
    this->atmoDataValid = this->atmoDensInMsg.isLinked() && this->atmoDensInMsg.isWritten();
    if (this->atmoDataValid) {
        this->atmoInBuffer = this->atmoDensInMsg();
    }
    this->windDataValid = this->windVelInMsg.isLinked() && this->windVelInMsg.isWritten();
    if (this->windDataValid) {
        this->windInBuffer = this->windVelInMsg();
    } else {
        this->windInBuffer = {};
    }
}

/*! Emit an error if the atmospheric density input message is not linked. */
void FacetDragSelfOcclusionEffectorBase::validateConfiguration()
{
    if (!this->atmoDensInMsg.isLinked()) {
        this->bskLogger.bskError("facetDragSelfOcclusionEffector.atmoDensInMsg was not linked.");
    }
}

/*! Add a flat-plate drag facet. See FacetSelfOcclusionEffectorBase::addFacetGeometry() for the
 rectangle definition.

 @param area [m^2] facet area
 @param dragCoeff [-] facet drag coefficient
 @param nHat_B [-] facet unit normal in B frame components
 @param r_CopB_B [m] facet center location relative to B in B frame components
 @param width [m] rectangle width; -1.0 (default) derives a square side from area
 @param height [m] rectangle height; -1.0 (default) derives a square side from area
 */
void FacetDragSelfOcclusionEffectorBase::addFacet(double area,
                                                  double dragCoeff,
                                                  Eigen::Vector3d nHat_B,
                                                  Eigen::Vector3d r_CopB_B,
                                                  double width,
                                                  double height)
{
    if (this->addFacetGeometry(area, nHat_B, r_CopB_B, width, height)) {
        this->dragCoeffs.push_back(dragCoeff);
    }
}

/*! Link the hub inertial velocity state.

 @param states dynamic parameter manager
 */
void FacetDragSelfOcclusionEffectorBase::linkInSourceStates(DynParamManager& states)
{
    this->hubVelocity = states.getStateObject(this->stateNameOfVelocity);
    if (this->hubVelocity == nullptr) {
        this->bskLogger.bskError("facetDragSelfOcclusionEffector.linkInStates could not find the hub velocity state.");
    }
}

/*! Link the parent frame inertial velocity property.

 @param properties dynamic parameter manager
 */
void FacetDragSelfOcclusionEffectorBase::linkInSourceProperties(DynParamManager& properties)
{
    this->inertialVelocityProperty = properties.getPropertyReference(this->propName_inertialVelocity);
}

/*! Compute the atmosphere-relative velocity direction and the dynamic pressure 0.5 rho v^2.

 The relative velocity is v_B = [BN] (v_BN_N - v_air_N), where the air velocity is only subtracted
 when the wind message is linked and written.

 @param dHat_B [-] output unit vector along the relative velocity
 @param fluxPressure [Pa] output dynamic pressure
 @return true when a valid atmosphere sample and a nonzero relative velocity are available
 */
bool FacetDragSelfOcclusionEffectorBase::updateSourceDirection(Eigen::Vector3d& dHat_B, double& fluxPressure)
{
    dHat_B.setZero();
    fluxPressure = 0.0;  // [Pa]
    Eigen::Matrix3d dcm_BN = this->computeDcm_BN();
    Eigen::Vector3d v_BN_N = this->isAttachedToHub() ? Eigen::Vector3d(this->hubVelocity->getState())
                                                     : Eigen::Vector3d(*this->inertialVelocityProperty);
    if (this->windDataValid) {
        v_BN_N -= Eigen::Map<Eigen::Vector3d>(this->windInBuffer.v_air_N);
    }
    this->v_B = dcm_BN * v_BN_N;
    double vNorm = this->v_B.norm();  // [m/s]
    if (!this->atmoDataValid || vNorm <= geometryTolerance) {
        return false;
    }
    dHat_B = this->v_B / vNorm;
    fluxPressure = 0.5 * this->atmoInBuffer.neutralDensity * vNorm * vNorm;
    return fluxPressure > 0.0;
}

/*! Flat-plate drag force of one facet, F = -q Cd A_visible dHat_B.

 @param facetIndex facet index
 @param visibleArea [m^2] visible projected area of the facet
 @param cosTheta [-] cosine between the facet normal and the source direction
 @param dHat_B [-] unit vector along the relative velocity
 @param fluxPressure [Pa] dynamic pressure
 @return [N] facet force in B frame components
 */
Eigen::Vector3d FacetDragSelfOcclusionEffectorBase::computeFacetForce(size_t facetIndex, double visibleArea,
                                                                      double cosTheta, const Eigen::Vector3d& dHat_B,
                                                                      double fluxPressure) const
{
    return -fluxPressure * this->dragCoeffs[facetIndex] * visibleArea * dHat_B;
}
