Executive Summary
-----------------
Faceted, flat-plate solar radiation pressure (SRP) dynamic effector with planar self-occlusion between the
spacecraft's own facets, computed by polygon clipping.

It is the self-occluding counterpart of :ref:`facetSRPDynamicEffector`. The facet definition, the force law,
the input messages and their timing are described in :ref:`facetSRPSelfOcclusionEffectorBase`. The self-occlusion model,
the visibility cache and the attachment options are described in :ref:`facetSelfOcclusionEffectorBase`.

Module Assumptions and Limitations
----------------------------------
- The assumptions of :ref:`facetSelfOcclusionEffectorBase` and :ref:`facetSRPSelfOcclusionEffectorBase` apply.
- The exact union of the shadows on one facet is evaluated by inclusion-exclusion, whose cost grows as
  :math:`2^k` with the number :math:`k` of facets shadowing it. Above 8 shadowing facets the union
  falls back to a grid quadrature with 41 samples per facet edge, and a warning is logged. The result
  is then approximate. ``setLogOverlapFallbackWarnings(False)`` turns this warning off (default
  ``True``); the result is the same either way.
- The visibility of every facet facing the source is computed against every other facet, so the cost
  per visibility update grows at least as :math:`N^2` with the number of facets :math:`N`. The
  visibility cache of :ref:`facetSelfOcclusionEffectorBase` avoids most of these updates.

Message Connection Descriptions
-------------------------------
Connect the inputs from Python using ``subscribeTo()``. The module writes no output message; its
force and torque are applied to the body it is attached to.

.. bsk-module-io:: facetSRPSelfOcclusionPolygonClippingEffector
    :caption: Module I/O Messages

    input sunInMsg SpicePlanetStateMsgPayload
        Sun inertial position.
    input sunEclipseInMsg EclipseMsgPayload
        (optional) Sun illumination factor scaling the SRP force and torque.

Detailed Module Description
---------------------------
For every facet :math:`i` facing the source, each other facet :math:`j` that is not edge-on to the
source is tested as a shadower:

#. The projected rectangles of :math:`i` and :math:`j` are intersected with Sutherland-Hodgman
   clipping.
#. The overlap is clipped to the half-plane where :math:`j` is upstream of :math:`i`. Both depth
   functions are affine, so :math:`z_i(x, y) - z_j(x, y) \ge 0` is a single half-plane. This also
   handles a tilted facet that is upstream over part of the overlap and downstream over the rest.
#. Shadow pieces smaller than :math:`10^{-9}` of the facet projected area are discarded as round-off.

The shadowed area is the area of the union of the pieces. Summing the piece areas would double count
regions shadowed by several facets, so the union is computed by inclusion-exclusion over every subset
of pieces, intersecting each subset with the same clipping routine. The first moment of every subset
is accumulated with its area, so that the centroid of the exposed region is

.. math::

    \mathbf{c}_{\text{exp}} = \frac{A_\text{proj}\, \mathbf{c}_\text{proj} - \mathbf{M}_\text{shadow}}
        {A_\text{proj} - A_\text{shadow}}

in projected coordinates. It is mapped back onto the facet plane with the facet depth function to
give the center of pressure. When two facets have exactly the same depth over their overlap, each is
treated as shadowed by the other.

User Guide
----------
The following example adds a large plate partly shadowed by a smaller plate upstream of it, with the
Sun along +x in the body frame.

.. code-block:: python

    from Basilisk.simulation import facetSRPSelfOcclusionPolygonClippingEffector

    effector = facetSRPSelfOcclusionPolygonClippingEffector.FacetSRPSelfOcclusionPolygonClippingEffector()
    effector.ModelTag = "selfOcclusionSRP"
    # effector.setLogOverlapFallbackWarnings(False)  # optional: silence the overlap-fallback warning
    # area [m^2], diffuse [-], specular [-], normal [-], location [m]
    effector.addFacet(10.0, 0.5, 0.2, [1, 0, 0], [0, 0, 0])
    effector.addFacet(1.0, 0.5, 0.2, [1, 0, 0], [0.5, 0, 0], 1.0, 1.0)  # optional width, height [m]
    effector.sunInMsg.subscribeTo(sunSpiceMsg)
    effector.sunEclipseInMsg.subscribeTo(eclipseObject.eclipseOutMsgs[0])  # optional
    scObject.addDynamicEffector(effector)
    scSim.AddModelToTask(taskName, effector)

``setDirectionCacheTolerance()`` sets the visibility cache angle in radians; see
:ref:`facetSelfOcclusionEffectorBase`.
