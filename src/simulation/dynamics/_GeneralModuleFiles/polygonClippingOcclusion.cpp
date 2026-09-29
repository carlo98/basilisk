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

#include "simulation/dynamics/_GeneralModuleFiles/polygonClippingOcclusion.h"

#include <algorithm>
#include <cmath>

using facetSelfOcclusion::FacetGeometry;
using facetSelfOcclusion::FacetProjection;
using facetSelfOcclusion::FacetVisibility;
using facetSelfOcclusion::ProjectedFacet;
using facetSelfOcclusion::cross2;
using facetSelfOcclusion::depthAt;
using facetSelfOcclusion::geometryTolerance;
using facetSelfOcclusion::pointInConvexPolygon;

namespace {

/*! Maximum number of overlapping upstream shadowers on one facet for which the exact O(2^k)
 inclusion-exclusion union is evaluated; beyond this a grid quadrature is used instead. */
constexpr size_t maxExactShadowUnionFacets = 8;

/*! Shadow patch area, relative to the shadowed facet projected area, below which the patch is
 treated as numerical noise (e.g. two facets that only share an edge in projection) and dropped. */
constexpr double minRelativeShadowArea = 1e-9;

/*! Grid points per axis of the grid-quadrature fallback. */
constexpr int quadratureGridSize = 41;

/*! Clip polygon ``subject`` against the half-plane on the left of the directed edge
 ``edgeStart`` -> ``edgeEnd`` (Sutherland-Hodgman). */
std::vector<Eigen::Vector2d> clipPolygon(const std::vector<Eigen::Vector2d>& subject,
                                         const Eigen::Vector2d& edgeStart,
                                         const Eigen::Vector2d& edgeEnd)
{
    std::vector<Eigen::Vector2d> output;
    if (subject.empty()) {
        return output;
    }
    Eigen::Vector2d edgeDir = edgeEnd - edgeStart;
    for (size_t i = 0; i < subject.size(); ++i) {
        Eigen::Vector2d cur = subject[i];
        Eigen::Vector2d prev = subject[(i + subject.size() - 1) % subject.size()];

        bool curInside = cross2(edgeDir, cur - edgeStart) >= 0.0;
        bool prevInside = cross2(edgeDir, prev - edgeStart) >= 0.0;

        Eigen::Vector2d hit;
        bool hasHit = false;
        if (curInside != prevInside) {
            Eigen::Vector2d seg = cur - prev;
            double denom = cross2(edgeDir, seg);
            if (std::fabs(denom) > geometryTolerance) {
                // denom = D_cur - D_prev, so the crossing fraction D_prev / (D_prev - D_cur) is -D_prev / denom
                double t = -cross2(edgeDir, prev - edgeStart) / denom;
                t = std::max(0.0, std::min(1.0, t));
                hit = prev + t * seg;
                hasHit = true;
            }
        }
        if (curInside) {
            if (!prevInside && hasHit) {
                output.push_back(hit);
            }
            output.push_back(cur);
        } else if (prevInside && hasHit) {
            output.push_back(hit);
        }
    }
    return output;
}

/*! Intersection of two convex polygons with consistent winding. */
std::vector<Eigen::Vector2d> polygonIntersection(const std::vector<Eigen::Vector2d>& clipWindow,
                                                 const std::vector<Eigen::Vector2d>& subject)
{
    std::vector<Eigen::Vector2d> clip = subject;
    for (size_t i = 0; i < clipWindow.size() && !clip.empty(); ++i) {
        size_t j = (i + 1) % clipWindow.size();
        clip = clipPolygon(clip, clipWindow[i], clipWindow[j]);
    }
    return clip;
}

/*! Intersection of every convex polygon in ``polys``. */
std::vector<Eigen::Vector2d> intersectAll(const std::vector<std::vector<Eigen::Vector2d>>& polys)
{
    if (polys.empty()) {
        return {};
    }
    std::vector<Eigen::Vector2d> result = polys[0];
    for (size_t k = 1; k < polys.size() && !result.empty(); ++k) {
        result = polygonIntersection(result, polys[k]);
    }
    return result;
}

/*! Area and first moment (centroid times area) of a polygon. */
struct AreaMoment {
    double area = 0.0;                                   //!< [m^2] polygon area
    Eigen::Vector2d moment = Eigen::Vector2d::Zero();    //!< [m^3] centroid times area
};

/*! Area and first moment of a polygon from the shoelace formula, independent of the winding. */
AreaMoment polygonAreaAndMoment(const std::vector<Eigen::Vector2d>& poly)
{
    if (poly.size() < 3) {
        return AreaMoment();
    }
    double signedArea = 0.0;
    double mx = 0.0;
    double my = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        size_t j = (i + 1) % poly.size();
        double cross = poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
        signedArea += cross;
        mx += (poly[i].x() + poly[j].x()) * cross;
        my += (poly[i].y() + poly[j].y()) * cross;
    }
    signedArea *= 0.5;
    AreaMoment result;
    if (std::fabs(signedArea) < 1e-15) {  // [m^2]
        return result;
    }
    Eigen::Vector2d centroid(mx / (6.0 * signedArea), my / (6.0 * signedArea));
    result.area = std::fabs(signedArea);
    result.moment = centroid * result.area;
    return result;
}

/*! Clip ``poly`` to the region where facet j is upstream of (closer to the source than) facet i.

 Depths are measured along the flux propagation direction, so j is upstream where depth_j <= depth_i.
 Both facet depth planes are affine in the projected coordinates, so D(x, y) = depth_i - depth_j is
 affine too and the shadow region D >= 0 is a single half-plane. */
std::vector<Eigen::Vector2d> clipToUpstreamHalfPlane(const std::vector<Eigen::Vector2d>& poly,
                                                     double depthA_i, double depthB_i, double depthC_i,
                                                     double depthA_j, double depthB_j, double depthC_j)
{
    double A = depthA_i - depthA_j;
    double B = depthB_i - depthB_j;
    double C = depthC_i - depthC_j;

    if (std::fabs(A) < geometryTolerance && std::fabs(B) < geometryTolerance) {
        // Constant depth offset: j shadows i everywhere they overlap, or nowhere.
        return (C >= 0.0) ? poly : std::vector<Eigen::Vector2d>();
    }

    // Any point on the line D = 0, solved on the more numerically stable axis.
    Eigen::Vector2d P0 = (std::fabs(A) >= std::fabs(B)) ? Eigen::Vector2d(-C / A, 0.0) : Eigen::Vector2d(0.0, -C / B);
    Eigen::Vector2d edgeDir(B, -A);  // cross2(edgeDir, point - P0) == D(point)
    return clipPolygon(poly, P0, P0 + edgeDir);
}

}  // namespace

/*! Compute the exact visibility of every facet facing the source.

 A facet facing away from the source never receives force, so its visibility is not computed. It
 still acts as an occluder of the other facets inside computeExposureFactor().

 @param geometry facet rectangles
 @param projection facet projection for the current source direction
 @param visibility per-facet visibility output
 @param bskLogger logger used to warn when the grid-quadrature fallback is used
 */
void PolygonClippingOcclusion::computeVisibility(const FacetGeometry& geometry,
                                                 const FacetProjection& projection,
                                                 std::vector<FacetVisibility>& visibility,
                                                 BSKLogger& bskLogger) const
{
    for (size_t i = 0; i < geometry.areas.size(); ++i) {
        if (projection.facets[i].sourceCosine <= 0.0) {
            continue;
        }
        visibility[i].exposure = this->computeExposureFactor(i, geometry, projection, visibility[i].r_CopB_B, bskLogger);
    }
}

/*! Compute the exposed (unshadowed) fraction of facet ``i`` and the center of pressure of its
 exposed region.

 For every other facet j, the patch of facet i's silhouette that j shadows is computed exactly as
 the overlap of the two projected rectangles, clipped to where j is pointwise upstream of i. The
 shadowed area is the area of the union of these patches, computed with inclusion-exclusion for up
 to maxExactShadowUnionFacets patches and with a grid quadrature beyond that. The first moment of
 every patch is tracked alongside its area, so that the exposed-region centroid can be recovered
 and mapped back onto the facet plane.

 @param i facet index
 @param geometry facet rectangles
 @param projection facet projection for the current source direction
 @param r_CopB_B [m] output center of pressure of the exposed region
 @param bskLogger logger used to warn when the grid-quadrature fallback is used
 @return [-] exposed fraction of the facet projected area, in [0, 1]
 */
double PolygonClippingOcclusion::computeExposureFactor(size_t i,
                                                       const FacetGeometry& geometry,
                                                       const FacetProjection& projection,
                                                       Eigen::Vector3d& r_CopB_B,
                                                       BSKLogger& bskLogger) const
{
    const Eigen::Vector3d& p1 = projection.p1_B;
    const Eigen::Vector3d& p2 = projection.p2_B;
    const ProjectedFacet& facet_i = projection.facets[i];
    r_CopB_B = geometry.r_CopB_B[i];
    if (facet_i.projectedArea <= geometryTolerance || std::fabs(facet_i.sourceCosine) <= geometryTolerance) {
        return 0.0;
    }

    std::vector<std::vector<Eigen::Vector2d>> shadowPieces;
    for (size_t j = 0; j < geometry.areas.size(); ++j) {
        if (j == i) {
            continue;
        }
        const ProjectedFacet& facet_j = projection.facets[j];
        if (std::fabs(facet_j.sourceCosine) <= geometryTolerance) {
            // An edge-on facet has a zero-area silhouette and an undefined depth plane.
            continue;
        }
        std::vector<Eigen::Vector2d> overlap = polygonIntersection(facet_i.pts, facet_j.pts);
        if (overlap.empty()) {
            continue;
        }
        std::vector<Eigen::Vector2d> shadow = clipToUpstreamHalfPlane(overlap,
                                                                      facet_i.depthA, facet_i.depthB, facet_i.depthC,
                                                                      facet_j.depthA, facet_j.depthB, facet_j.depthC);
        if (!shadow.empty()) {
            double shadowArea = polygonAreaAndMoment(shadow).area;  // [m^2]
            if (shadowArea > minRelativeShadowArea * facet_i.projectedArea) {
                shadowPieces.push_back(std::move(shadow));
            }
        }
    }

    if (shadowPieces.empty()) {
        // Fully exposed: the facet center is already the exact center of pressure.
        return 1.0;
    }

    double shadowedArea = 0.0;  // [m^2]
    Eigen::Vector3d exposedPointSum_B = Eigen::Vector3d::Zero();  // [m] grid path only
    int exposedSampleCount = 0;  // grid path only
    bool haveExposedCentroid2D = false;
    Eigen::Vector2d exposedCentroid2D = Eigen::Vector2d::Zero();

    if (shadowPieces.size() > maxExactShadowUnionFacets) {
        if (this->logOverlapFallbackWarnings) {
            bskLogger.bskLog(BSK_WARNING,
                             "polygonClippingOcclusion: %d facets simultaneously shadow one "
                             "facet; using an approximate grid-quadrature union instead of the exact computation.",
                             static_cast<int>(shadowPieces.size()));
        }
        const Eigen::Vector3d& center = geometry.r_CopB_B[i];
        const Eigen::Vector3d& e1 = geometry.halfEdge1_B[i];
        const Eigen::Vector3d& e2 = geometry.halfEdge2_B[i];
        int covered = 0;
        int total = 0;
        for (int a = 0; a < quadratureGridSize; ++a) {
            double u = -1.0 + 2.0 * (a + 0.5) / quadratureGridSize;
            for (int b = 0; b < quadratureGridSize; ++b) {
                double v = -1.0 + 2.0 * (b + 0.5) / quadratureGridSize;
                Eigen::Vector3d point_B = center + u * e1 + v * e2;
                Eigen::Vector2d point2D(point_B.dot(p1), point_B.dot(p2));
                ++total;
                bool isShadowed = false;
                for (const auto& piece : shadowPieces) {
                    if (pointInConvexPolygon(point2D, piece)) {
                        isShadowed = true;
                        break;
                    }
                }
                if (isShadowed) {
                    ++covered;
                } else {
                    exposedPointSum_B += point_B;
                    ++exposedSampleCount;
                }
            }
        }
        shadowedArea = facet_i.projectedArea * (static_cast<double>(covered) / static_cast<double>(total));
    } else {
        // Exact union by inclusion-exclusion over every nonempty subset of shadow pieces.
        size_t numPieces = shadowPieces.size();
        Eigen::Vector2d shadowedMoment2D = Eigen::Vector2d::Zero();
        for (size_t mask = 1; mask < (size_t(1) << numPieces); ++mask) {
            std::vector<std::vector<Eigen::Vector2d>> subset;
            int bitCount = 0;
            for (size_t bit = 0; bit < numPieces; ++bit) {
                if (mask & (size_t(1) << bit)) {
                    subset.push_back(shadowPieces[bit]);
                    ++bitCount;
                }
            }
            AreaMoment areaMoment = polygonAreaAndMoment(intersectAll(subset));
            double sign = (bitCount % 2 == 1) ? 1.0 : -1.0;
            shadowedArea += sign * areaMoment.area;
            shadowedMoment2D += sign * areaMoment.moment;
        }

        double exposedAreaCheck = facet_i.projectedArea - shadowedArea;  // [m^2]
        if (exposedAreaCheck > geometryTolerance) {
            Eigen::Vector2d fullCentroid2D(geometry.r_CopB_B[i].dot(p1),
                                           geometry.r_CopB_B[i].dot(p2));
            Eigen::Vector2d fullMoment2D = fullCentroid2D * facet_i.projectedArea;
            exposedCentroid2D = (fullMoment2D - shadowedMoment2D) / exposedAreaCheck;
            haveExposedCentroid2D = true;
        }
    }

    double exposedArea = std::max(0.0, facet_i.projectedArea - shadowedArea);  // [m^2]
    if (exposedArea > geometryTolerance) {
        if (haveExposedCentroid2D) {
            // Map the exposed centroid back onto facet i's plane with its affine depth function.
            double depth = depthAt(facet_i, exposedCentroid2D.x(), exposedCentroid2D.y());  // [m]
            r_CopB_B = exposedCentroid2D.x() * p1 + exposedCentroid2D.y() * p2 - depth * projection.dHat_B;
        } else if (exposedSampleCount > 0) {
            r_CopB_B = exposedPointSum_B / static_cast<double>(exposedSampleCount);
        }
        // Otherwise no exposed sample was captured by the grid fallback; keep the facet center.
    }
    return std::min(1.0, exposedArea / facet_i.projectedArea);
}
