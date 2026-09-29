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

#ifndef FACET_SELF_OCCLUSION_GEOMETRY_H
#define FACET_SELF_OCCLUSION_GEOMETRY_H

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

/*! @brief Geometry shared by the facet self-occlusion effectors and their occlusion algorithms.

 Every facet is a planar rectangle in the effector frame B. For a given source direction dHat_B,
 which points from the spacecraft toward the incoming flux, every rectangle is orthographically
 projected onto the plane normal to dHat_B. The depth of a projected point is measured along the
 flux propagation direction -dHat_B, so that the smallest depth is the closest to the source.
 */
namespace facetSelfOcclusion {

/*! @brief Facet rectangles in the effector frame B */
struct FacetGeometry {
    std::vector<double> areas;                           //!< [m^2] facet areas
    std::vector<Eigen::Vector3d> nHat_B;                 //!< [-] facet unit normals
    std::vector<Eigen::Vector3d> r_CopB_B;               //!< [m] facet center locations relative to B
    std::vector<Eigen::Vector3d> halfEdge1_B;            //!< [m] in-plane rectangle half-edge along width
    std::vector<Eigen::Vector3d> halfEdge2_B;            //!< [m] in-plane rectangle half-edge along height
};

/*! @brief One facet rectangle orthographically projected onto the plane normal to the source direction */
struct ProjectedFacet {
    std::vector<Eigen::Vector2d> pts;                    //!< [m] projected corners in (p1, p2) coordinates
    double projectedArea = 0.0;                          //!< [m^2] facet area times |sourceCosine|
    double sourceCosine = 0.0;                           //!< [-] signed nHat_B . dHat_B
    /*! Affine depth plane depth(x, y) = depthA*x + depthB*y + depthC, measured along the flux
     * propagation direction -dHat_B, so that the point on the facet plane projecting to (x, y) is
     * x*p1 + y*p2 - depth*dHat_B. At a point covered by several facets the smallest depth is the one
     * closest to the source, i.e. the visible (upstream) one. Left at zero for an edge-on facet
     * (|sourceCosine| below tolerance). */
    double depthA = 0.0;                                 //!< [-] depth slope along p1
    double depthB = 0.0;                                 //!< [-] depth slope along p2
    double depthC = 0.0;                                 //!< [m] depth offset
    double minX = 0.0;                                   //!< [m] projected bounding box minimum along p1
    double maxX = 0.0;                                   //!< [m] projected bounding box maximum along p1
    double minY = 0.0;                                   //!< [m] projected bounding box minimum along p2
    double maxY = 0.0;                                   //!< [m] projected bounding box maximum along p2
};

/*! @brief All facets projected for one source direction */
struct FacetProjection {
    Eigen::Vector3d dHat_B = Eigen::Vector3d::Zero();    //!< [-] source direction the projection was built for
    Eigen::Vector3d p1_B = Eigen::Vector3d::Zero();      //!< [-] first in-plane projection axis
    Eigen::Vector3d p2_B = Eigen::Vector3d::Zero();      //!< [-] second in-plane projection axis
    std::vector<ProjectedFacet> facets;                  //!< per-facet projection, same order as the facets
};

/*! @brief Visibility of one facet for one source direction */
struct FacetVisibility {
    double exposure = 0.0;                               //!< [-] visible fraction of the facet projected area
    Eigen::Vector3d r_CopB_B = Eigen::Vector3d::Zero();  //!< [m] center of pressure of the visible region
};

/*! Tolerance on cosines, norms and projected areas used by the self-occlusion geometry. */
constexpr double geometryTolerance = 1e-12;  // [-]

/*! @brief 2D cross product.
    @param u first vector
    @param v second vector
    @return u_x v_y - u_y v_x */
inline double cross2(const Eigen::Vector2d& u, const Eigen::Vector2d& v)
{
    return u[0] * v[1] - u[1] * v[0];
}

/*! @brief Test whether a point lies inside, or on the boundary of, a convex polygon with consistent winding.
    @param point 2D point
    @param poly convex polygon vertices
    @return true if the point is inside or on the boundary */
inline bool pointInConvexPolygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& poly)
{
    if (poly.size() < 3) {
        return false;
    }
    bool sawPositive = false;
    bool sawNegative = false;
    for (size_t i = 0; i < poly.size(); ++i) {
        size_t j = (i + 1) % poly.size();
        double cross = cross2(poly[j] - poly[i], point - poly[i]);
        if (cross > geometryTolerance) {
            sawPositive = true;
        }
        if (cross < -geometryTolerance) {
            sawNegative = true;
        }
        if (sawPositive && sawNegative) {
            return false;
        }
    }
    return true;
}

/*! @brief Evaluate a projected facet's affine depth plane.
    @param facet projected facet
    @param x [m] projected coordinate along p1
    @param y [m] projected coordinate along p2
    @return [m] depth along the flux propagation direction -dHat_B; smaller is closer to the source */
inline double depthAt(const ProjectedFacet& facet, double x, double y)
{
    return facet.depthA * x + facet.depthB * y + facet.depthC;
}

}  // namespace facetSelfOcclusion

#endif
