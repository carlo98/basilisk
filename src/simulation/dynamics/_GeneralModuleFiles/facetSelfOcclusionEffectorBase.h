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

#ifndef FACET_SELF_OCCLUSION_EFFECTOR_BASE_H
#define FACET_SELF_OCCLUSION_EFFECTOR_BASE_H

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"
#include "simulation/dynamics/_GeneralModuleFiles/dynParamManager.h"
#include "simulation/dynamics/_GeneralModuleFiles/dynamicEffector.h"
#include "simulation/dynamics/_GeneralModuleFiles/facetSelfOcclusionGeometry.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"

/*! @brief Abstract faceted flat-plate dynamic effector with planar self-occlusion.

 This base class owns what is common to every faceted self-occlusion effector, whatever the flux
 (atmospheric drag or solar radiation pressure) and whatever the occlusion algorithm: the facet
 rectangles, their projection onto the plane normal to the source direction, the source-direction
 visibility cache, the effector attitude and the force and torque accumulation.

 Two levels of derived classes complete it:

 - a flux base class, such as FacetDragSelfOcclusionEffectorBase or FacetSRPSelfOcclusionEffectorBase,
   provides the input messages, the source direction and pressure, the per-facet force law and the
   facet coefficients;
 - a concrete module provides the occlusion algorithm through computeFacetVisibility(), usually by
   delegating to a reusable algorithm class such as PolygonClippingOcclusion.

 The B frame quantities are expressed in the frame this effector is attached to. That is the hub
 body frame in the usual case, and the parent segment frame when this effector is attached to a
 state effector through effector branching.
 */
class FacetSelfOcclusionEffectorBase : public SysModel, public DynamicEffector {
public:
    FacetSelfOcclusionEffectorBase();
    virtual ~FacetSelfOcclusionEffectorBase() = default;

    void linkInStates(DynParamManager& states) override;
    void linkInProperties(DynParamManager& properties) override;
    void computeForceTorque(double integTime, double timeStep) override;

    /*! @brief Get the number of facets added to this effector.
        @return number of facets */
    uint64_t getNumFacets() const { return this->numFacets; }
    void setDirectionCacheTolerance(double tolerance);
    /*! @brief Get the source-direction visibility cache tolerance.
        @return [rad] cache tolerance angle */
    double getDirectionCacheTolerance() const { return this->directionCacheTolerance; }
    /*! @brief Get the unit vector from B toward the flux source computed at the last force evaluation.
        @return [-] source direction in B frame components; zero if no valid source was available */
    Eigen::Vector3d getSourceDirection_B() const { return this->dHat_B; }

    BSKLogger bskLogger;                                     //!< BSK Logging

protected:
    bool addFacetGeometry(double area, const Eigen::Vector3d& nHat_B, const Eigen::Vector3d& r_CopB_B,
                          double width, double height);
    void invalidateVisibilityCache();
    bool isAttachedToHub() const;
    Eigen::Matrix3d computeDcm_BN() const;

    /*! @brief Emit an error if the configuration cannot be evaluated, e.g. a required input message
        is not linked. Called when the effector is linked to the spacecraft. */
    virtual void validateConfiguration() = 0;

    /*! @brief Link the hub translational state the flux model needs (position or velocity).
        @param states dynamic parameter manager */
    virtual void linkInSourceStates(DynParamManager& states) = 0;

    /*! @brief Link the parent-frame translational property the flux model needs (position or velocity).
        @param properties dynamic parameter manager */
    virtual void linkInSourceProperties(DynParamManager& properties) = 0;

    /*! @brief Compute the source direction and the flux pressure.
        @param dHat_B [-] output unit vector from B toward the flux source
        @param fluxPressure [Pa] output flux pressure
        @return true when a valid source direction and a positive pressure are available */
    virtual bool updateSourceDirection(Eigen::Vector3d& dHat_B, double& fluxPressure) = 0;

    /*! @brief Compute the force of one facet facing the source.
        @param facetIndex facet index
        @param visibleArea [m^2] visible projected area of the facet
        @param cosTheta [-] cosine between the facet normal and the source direction, positive
        @param dHat_B [-] unit vector from B toward the flux source
        @param fluxPressure [Pa] flux pressure
        @return [N] facet force in B frame components */
    virtual Eigen::Vector3d computeFacetForce(size_t facetIndex, double visibleArea, double cosTheta,
                                              const Eigen::Vector3d& dHat_B, double fluxPressure) const = 0;

    /*! @brief Scale factor applied to the total force and torque, e.g. the eclipse illumination factor.
        @return [-] total force scale factor */
    virtual double totalForceScale() const { return 1.0; }

    /*! @brief Resolve the visible fraction and center of pressure of every facet.

        Implementations must treat every facet as a potential occluder, whichever way its own normal
        points. Only the entries of facets with a positive source cosine are read by the force law.
        The visibility vector is sized to the number of facets and default-initialized (exposure 0)
        before this method is called.

        @param projection facet projection for the current source direction
        @param visibility per-facet visibility output
     */
    virtual void computeFacetVisibility(const facetSelfOcclusion::FacetProjection& projection,
                                        std::vector<facetSelfOcclusion::FacetVisibility>& visibility) = 0;

    facetSelfOcclusion::FacetGeometry facetGeometry;          //!< facet rectangles
    uint64_t numFacets = 0;                                  //!< [-] number of facets

private:
    void buildFacetRectangles();
    void projectFacets(const Eigen::Vector3d& sourceDirection_B, facetSelfOcclusion::FacetProjection& projection) const;

    double directionCacheTolerance = 1e-5;                   //!< [rad] visibility cache reuse angle, 0 disables caching
    std::vector<double> facetWidths;                         //!< [m] per-facet rectangle width, -1 derives it from area
    std::vector<double> facetHeights;                        //!< [m] per-facet rectangle height, -1 derives it from area
    bool geometryBuilt = false;                              //!< true once the rectangle half-edges are built

    std::vector<facetSelfOcclusion::FacetVisibility> visibilityCache;  //!< per-facet visibility for cachedDirection_B
    Eigen::Vector3d cachedDirection_B = Eigen::Vector3d::Zero();  //!< [-] source direction of visibilityCache
    bool visibilityCacheValid = false;                       //!< true once visibilityCache is usable

    Eigen::Vector3d dHat_B = Eigen::Vector3d::Zero();        //!< [-] unit vector from B toward the flux source

    StateData *hubSigma = nullptr;                           //!< [-] hub MRP inertial attitude state
    Eigen::MatrixXd *inertialAttitudeProperty = nullptr;     //!< [-] parent frame MRP inertial attitude
};

#endif
