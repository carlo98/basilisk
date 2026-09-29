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

#ifndef POLYGON_CLIPPING_OCCLUSION_H
#define POLYGON_CLIPPING_OCCLUSION_H

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "architecture/utilities/bskLogging.h"
#include "simulation/dynamics/_GeneralModuleFiles/facetSelfOcclusionGeometry.h"

/*! @brief Exact facet self-occlusion by polygon clipping.

 For every facet facing the source, the shadow cast by each other facet is computed with
 Sutherland-Hodgman polygon clipping, restricted to where the other facet is upstream, and the
 union of these shadows is evaluated with inclusion-exclusion. Beyond a fixed number of
 overlapping shadowers the union falls back to a bounded-cost grid quadrature.

 This class holds no state and is shared by the drag and SRP polygon-clipping effectors.
 */
class PolygonClippingOcclusion {
public:
    void computeVisibility(const facetSelfOcclusion::FacetGeometry& geometry,
                           const facetSelfOcclusion::FacetProjection& projection,
                           std::vector<facetSelfOcclusion::FacetVisibility>& visibility,
                           BSKLogger& bskLogger) const;

    /*! @brief Enable or disable the warning logged when too many facets overlap on one facet for the
        exact union, and the grid-quadrature fallback is used instead.
        @param logWarnings true to log the warning (default), false to suppress it */
    void setLogOverlapFallbackWarnings(bool logWarnings) { this->logOverlapFallbackWarnings = logWarnings; }
    /*! @brief Get whether the overlap-fallback warning is logged.
        @return true if the warning is logged */
    bool getLogOverlapFallbackWarnings() const { return this->logOverlapFallbackWarnings; }

private:
    double computeExposureFactor(size_t i,
                                 const facetSelfOcclusion::FacetGeometry& geometry,
                                 const facetSelfOcclusion::FacetProjection& projection,
                                 Eigen::Vector3d& r_CopB_B,
                                 BSKLogger& bskLogger) const;

    bool logOverlapFallbackWarnings = true;                  //!< log a warning when the grid-quadrature fallback is used
};

#endif
