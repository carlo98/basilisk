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

#ifndef FACET_SRP_SELF_OCCLUSION_EFFECTOR_BASE_H
#define FACET_SRP_SELF_OCCLUSION_EFFECTOR_BASE_H

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "architecture/messaging/messaging.h"
#include "architecture/msgPayloadDefC/EclipseMsgPayload.h"
#include "architecture/msgPayloadDefC/SpicePlanetStateMsgPayload.h"
#include "simulation/dynamics/_GeneralModuleFiles/dynParamManager.h"
#include "simulation/dynamics/_GeneralModuleFiles/facetSelfOcclusionEffectorBase.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"

/*! @brief Abstract faceted flat-plate solar radiation pressure (SRP) effector with planar self-occlusion.

 Provides the SRP flux model of the self-occlusion effectors: the Sun ephemeris and eclipse input
 messages, the Sun direction used as source direction, the solar radiation pressure at the current
 Sun distance, the per-facet flat-plate absorption and diffuse/specular reflection force, and the
 eclipse scaling. Concrete modules only provide the occlusion algorithm.
 */
class FacetSRPSelfOcclusionEffectorBase : public FacetSelfOcclusionEffectorBase {
public:
    FacetSRPSelfOcclusionEffectorBase();
    virtual ~FacetSRPSelfOcclusionEffectorBase() = default;

    void Reset(uint64_t currentSimNanos) override;
    void UpdateState(uint64_t currentSimNanos) override;
    void readInputMessages();

    void addFacet(double area,
                  double diffuseCoeff,
                  double specularCoeff,
                  Eigen::Vector3d nHat_B,
                  Eigen::Vector3d r_CopB_B,
                  double width = -1.0,
                  double height = -1.0);

    /*! @brief Get the spacecraft-to-Sun distance computed at the last force evaluation.
        @return [m] spacecraft-to-Sun distance */
    double getSunDistance() const { return this->r_SB_norm; }

    ReadFunctor<SpicePlanetStateMsgPayload> sunInMsg;        //!< Sun ephemeris input message
    ReadFunctor<EclipseMsgPayload> sunEclipseInMsg;          //!< (optional) Sun eclipse input message

protected:
    void validateConfiguration() override;
    void linkInSourceStates(DynParamManager& states) override;
    void linkInSourceProperties(DynParamManager& properties) override;
    bool updateSourceDirection(Eigen::Vector3d& dHat_B, double& fluxPressure) override;
    Eigen::Vector3d computeFacetForce(size_t facetIndex, double visibleArea, double cosTheta,
                                      const Eigen::Vector3d& dHat_B, double fluxPressure) const override;
    double totalForceScale() const override;

private:
    std::vector<double> diffuseCoeffs;                       //!< [-] facet diffuse reflection coefficients
    std::vector<double> specularCoeffs;                      //!< [-] facet specular reflection coefficients
    double r_SB_norm = 0.0;                                  //!< [m] spacecraft-to-Sun distance
    SpicePlanetStateMsgPayload sunInBuffer{};                //!< Sun ephemeris input buffer
    EclipseMsgPayload eclipseInBuffer{};                     //!< eclipse input buffer
    bool sunDataValid = false;                               //!< true only if sunInMsg is linked and written
    StateData *hubPosition = nullptr;                        //!< [m] hub inertial position state
    Eigen::MatrixXd *inertialPositionProperty = nullptr;     //!< [m] parent frame origin inertial position
};

#endif
