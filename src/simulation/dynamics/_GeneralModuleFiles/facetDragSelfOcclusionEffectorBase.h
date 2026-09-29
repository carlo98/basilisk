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

#ifndef FACET_DRAG_SELF_OCCLUSION_EFFECTOR_BASE_H
#define FACET_DRAG_SELF_OCCLUSION_EFFECTOR_BASE_H

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "architecture/messaging/messaging.h"
#include "architecture/msgPayloadDefC/AtmoPropsMsgPayload.h"
#include "architecture/msgPayloadDefC/WindMsgPayload.h"
#include "simulation/dynamics/_GeneralModuleFiles/dynParamManager.h"
#include "simulation/dynamics/_GeneralModuleFiles/facetSelfOcclusionEffectorBase.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"

/*! @brief Abstract faceted flat-plate atmospheric drag effector with planar self-occlusion.

 Provides the drag flux model of the self-occlusion effectors: the atmosphere and wind input
 messages, the atmosphere-relative velocity used as source direction, the dynamic pressure and the
 per-facet flat-plate drag force. Concrete modules only provide the occlusion algorithm.
 */
class FacetDragSelfOcclusionEffectorBase : public FacetSelfOcclusionEffectorBase {
public:
    FacetDragSelfOcclusionEffectorBase() = default;
    virtual ~FacetDragSelfOcclusionEffectorBase() = default;

    void Reset(uint64_t currentSimNanos) override;
    void UpdateState(uint64_t currentSimNanos) override;
    void readInputMessages();

    void addFacet(double area,
                  double dragCoeff,
                  Eigen::Vector3d nHat_B,
                  Eigen::Vector3d r_CopB_B,
                  double width = -1.0,
                  double height = -1.0);

    /*! @brief Get the atmosphere-relative spacecraft velocity computed at the last force evaluation.
        @return [m/s] relative velocity in B frame components */
    Eigen::Vector3d getRelativeVelocity_B() const { return this->v_B; }

    ReadFunctor<AtmoPropsMsgPayload> atmoDensInMsg;          //!< atmospheric density input message
    ReadFunctor<WindMsgPayload> windVelInMsg;                //!< (optional) wind velocity input message

protected:
    void validateConfiguration() override;
    void linkInSourceStates(DynParamManager& states) override;
    void linkInSourceProperties(DynParamManager& properties) override;
    bool updateSourceDirection(Eigen::Vector3d& dHat_B, double& fluxPressure) override;
    Eigen::Vector3d computeFacetForce(size_t facetIndex, double visibleArea, double cosTheta,
                                      const Eigen::Vector3d& dHat_B, double fluxPressure) const override;

private:
    std::vector<double> dragCoeffs;                          //!< [-] facet drag coefficients
    Eigen::Vector3d v_B = Eigen::Vector3d::Zero();           //!< [m/s] atmosphere-relative velocity
    AtmoPropsMsgPayload atmoInBuffer{};                      //!< atmosphere input buffer
    WindMsgPayload windInBuffer{};                           //!< wind input buffer
    bool atmoDataValid = false;                              //!< true only if atmoDensInMsg is linked and written
    bool windDataValid = false;                              //!< true only if windVelInMsg is linked and written
    StateData *hubVelocity = nullptr;                        //!< [m/s] hub inertial velocity state
    Eigen::MatrixXd *inertialVelocityProperty = nullptr;     //!< [m/s] parent frame origin inertial velocity
};

#endif
