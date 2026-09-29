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

#ifndef FACET_SRP_SELF_OCCLUSION_POLYGON_CLIPPING_EFFECTOR_H
#define FACET_SRP_SELF_OCCLUSION_POLYGON_CLIPPING_EFFECTOR_H

#include <vector>

#include "simulation/dynamics/_GeneralModuleFiles/facetSRPSelfOcclusionEffectorBase.h"
#include "simulation/dynamics/_GeneralModuleFiles/facetSelfOcclusionGeometry.h"
#include "simulation/dynamics/_GeneralModuleFiles/polygonClippingOcclusion.h"

/*! @brief Faceted flat-plate solar radiation pressure (SRP) effector with planar self-occlusion computed exactly by polygon clipping */
class FacetSRPSelfOcclusionPolygonClippingEffector final : public FacetSRPSelfOcclusionEffectorBase {
public:
    FacetSRPSelfOcclusionPolygonClippingEffector() = default;
    ~FacetSRPSelfOcclusionPolygonClippingEffector() = default;

    void setLogOverlapFallbackWarnings(bool logWarnings);
    /*! @brief Get whether the overlap-fallback warning is logged.
        @return true if the warning is logged */
    bool getLogOverlapFallbackWarnings() const { return this->occlusion.getLogOverlapFallbackWarnings(); }

private:
    void computeFacetVisibility(const facetSelfOcclusion::FacetProjection& projection,
                                std::vector<facetSelfOcclusion::FacetVisibility>& visibility) override;

    PolygonClippingOcclusion occlusion;  //!< occlusion algorithm
};

#endif
