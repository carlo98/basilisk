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

#include "simulation/dynamics/_GeneralModuleFiles/facetSRPSelfOcclusionEffectorBase.h"
#include "architecture/utilities/astroConstants.h"

using facetSelfOcclusion::geometryTolerance;

/*! Constructor */
FacetSRPSelfOcclusionEffectorBase::FacetSRPSelfOcclusionEffectorBase()
{
    this->eclipseInBuffer.illuminationFactor = 1.0;  // [-]
}

/*! Check that the Sun ephemeris input message is linked and restore full illumination.

 @param currentSimNanos [ns] current simulation time
 */
void FacetSRPSelfOcclusionEffectorBase::Reset(uint64_t currentSimNanos)
{
    this->validateConfiguration();
    this->eclipseInBuffer.illuminationFactor = 1.0;  // [-]
}

/*! Read the input messages. Their content is held constant over the following integration step.

 @param currentSimNanos [ns] current simulation time
 */
void FacetSRPSelfOcclusionEffectorBase::UpdateState(uint64_t currentSimNanos)
{
    this->readInputMessages();
}

/*! Read the Sun ephemeris and eclipse input messages into the module buffers.

 A linked but never written Sun message is not treated as valid data: without a valid Sun sample the
 effector applies no force. Full illumination is assumed unless the eclipse message is linked and
 written.
 */
void FacetSRPSelfOcclusionEffectorBase::readInputMessages()
{
    this->sunDataValid = this->sunInMsg.isLinked() && this->sunInMsg.isWritten();
    if (this->sunDataValid) {
        this->sunInBuffer = this->sunInMsg();
    }
    this->eclipseInBuffer = {};
    this->eclipseInBuffer.illuminationFactor = 1.0;  // [-]
    if (this->sunEclipseInMsg.isLinked() && this->sunEclipseInMsg.isWritten()) {
        this->eclipseInBuffer = this->sunEclipseInMsg();
    }
}

/*! Emit an error if the Sun ephemeris input message is not linked. */
void FacetSRPSelfOcclusionEffectorBase::validateConfiguration()
{
    if (!this->sunInMsg.isLinked()) {
        this->bskLogger.bskError("facetSRPSelfOcclusionEffector.sunInMsg was not linked.");
    }
}

/*! Add a flat-plate SRP facet. See FacetSelfOcclusionEffectorBase::addFacetGeometry() for the
 rectangle definition.

 @param area [m^2] facet area
 @param diffuseCoeff [-] facet diffuse reflection coefficient, in [0, 1]
 @param specularCoeff [-] facet specular reflection coefficient, in [0, 1]
 @param nHat_B [-] facet unit normal in B frame components
 @param r_CopB_B [m] facet center location relative to B in B frame components
 @param width [m] rectangle width; -1.0 (default) derives a square side from area
 @param height [m] rectangle height; -1.0 (default) derives a square side from area
 */
void FacetSRPSelfOcclusionEffectorBase::addFacet(double area,
                                                 double diffuseCoeff,
                                                 double specularCoeff,
                                                 Eigen::Vector3d nHat_B,
                                                 Eigen::Vector3d r_CopB_B,
                                                 double width,
                                                 double height)
{
    if (diffuseCoeff < 0.0 || diffuseCoeff > 1.0 || specularCoeff < 0.0 || specularCoeff > 1.0) {
        this->bskLogger.bskError("facetSRPSelfOcclusionEffector.addFacet was given optical coefficients outside "
                                 "[0, 1]: diffuse %f, specular %f.",
                                 diffuseCoeff,
                                 specularCoeff);
    }
    if (this->addFacetGeometry(area, nHat_B, r_CopB_B, width, height)) {
        this->diffuseCoeffs.push_back(diffuseCoeff);
        this->specularCoeffs.push_back(specularCoeff);
    }
}

/*! Link the hub inertial position state.

 @param states dynamic parameter manager
 */
void FacetSRPSelfOcclusionEffectorBase::linkInSourceStates(DynParamManager& states)
{
    this->hubPosition = states.getStateObject(this->stateNameOfPosition);
    if (this->hubPosition == nullptr) {
        this->bskLogger.bskError("facetSRPSelfOcclusionEffector.linkInStates could not find the hub position state.");
    }
}

/*! Link the parent frame inertial position property.

 @param properties dynamic parameter manager
 */
void FacetSRPSelfOcclusionEffectorBase::linkInSourceProperties(DynParamManager& properties)
{
    this->inertialPositionProperty = properties.getPropertyReference(this->propName_inertialPosition);
}

/*! Compute the Sun direction and the solar radiation pressure P = (F_sun / c) (1 AU / r)^2, with the
 solar flux at 1 AU SOLAR_FLUX_EARTH, the speed of light SPEED_LIGHT and the astronomical unit AU2M
 from astroConstants.h.

 @param dHat_B [-] output unit vector from B toward the Sun
 @param fluxPressure [Pa] output solar radiation pressure
 @return true when a valid Sun sample at a nonzero distance is available
 */
bool FacetSRPSelfOcclusionEffectorBase::updateSourceDirection(Eigen::Vector3d& dHat_B, double& fluxPressure)
{
    dHat_B.setZero();
    fluxPressure = 0.0;  // [Pa]
    if (!this->sunDataValid) {
        this->r_SB_norm = 0.0;  // [m]
        return false;
    }
    Eigen::Matrix3d dcm_BN = this->computeDcm_BN();
    Eigen::Vector3d r_BN_N = this->isAttachedToHub() ? Eigen::Vector3d(this->hubPosition->getState())
                                                     : Eigen::Vector3d(*this->inertialPositionProperty);
    Eigen::Vector3d r_SB_B = dcm_BN * (Eigen::Map<Eigen::Vector3d>(this->sunInBuffer.PositionVector) - r_BN_N);
    this->r_SB_norm = r_SB_B.norm();
    if (this->r_SB_norm <= geometryTolerance) {
        return false;
    }
    dHat_B = r_SB_B / this->r_SB_norm;
    double distanceRatio = AU2M / this->r_SB_norm;  // [-]
    fluxPressure = (SOLAR_FLUX_EARTH / SPEED_LIGHT) * distanceRatio * distanceRatio;
    return fluxPressure > 0.0;
}

/*! Flat-plate SRP force of one facet,
 F = -P A_visible ((1 - S) sHat + 2 (D / 3 + S cos(theta)) nHat).

 @param facetIndex facet index
 @param visibleArea [m^2] visible projected area of the facet
 @param cosTheta [-] cosine between the facet normal and the Sun direction
 @param dHat_B [-] unit vector from B toward the Sun
 @param fluxPressure [Pa] solar radiation pressure
 @return [N] facet force in B frame components
 */
Eigen::Vector3d FacetSRPSelfOcclusionEffectorBase::computeFacetForce(size_t facetIndex, double visibleArea,
                                                                     double cosTheta, const Eigen::Vector3d& dHat_B,
                                                                     double fluxPressure) const
{
    double specularCoeff = this->specularCoeffs[facetIndex];
    double diffuseCoeff = this->diffuseCoeffs[facetIndex];
    return -fluxPressure * visibleArea
           * ((1.0 - specularCoeff) * dHat_B
              + 2.0 * (diffuseCoeff / 3.0 + specularCoeff * cosTheta) * this->facetGeometry.nHat_B[facetIndex]);
}

/*! @return [-] eclipse illumination factor scaling the total SRP force and torque */
double FacetSRPSelfOcclusionEffectorBase::totalForceScale() const
{
    return this->eclipseInBuffer.illuminationFactor;
}
