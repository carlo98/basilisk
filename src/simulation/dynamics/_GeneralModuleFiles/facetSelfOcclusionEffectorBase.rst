Executive Summary
-----------------
Abstract base class for faceted, flat-plate dynamic effectors that account for planar self-occlusion
between the spacecraft's own facets.

The flat-plate models of :ref:`facetDragDynamicEffector` and :ref:`facetSRPDynamicEffector` use the
full projected area of every facet facing the flux source. That is exact for a convex body, but
overestimates the force on a non-convex spacecraft, where one facet can sit between another facet
and the flow or the Sun. It also applies each facet force at the facet center, while on a partially
shadowed facet the force acts at the centroid of the visible region.

The self-occlusion effectors are organized in three layers, so that neither the flux model nor the
occlusion algorithm is implemented more than once:

.. code-block:: text

    FacetSelfOcclusionEffectorBase                  facets, projection, visibility cache, force loop
    ├── FacetDragSelfOcclusionEffectorBase          atmosphere and wind inputs, drag force law
    │   └── facetDragSelfOcclusionPolygonClippingEffector  ── PolygonClippingOcclusion
    └── FacetSRPSelfOcclusionEffectorBase           Sun and eclipse inputs, SRP force law
        └── facetSRPSelfOcclusionPolygonClippingEffector   ── PolygonClippingOcclusion

- This class owns the facet rectangles, their projection onto the plane normal to the source
  direction, the visibility cache, the effector attitude, and the accumulation of the facet forces
  and torques.
- The flux base classes :ref:`facetDragSelfOcclusionEffectorBase` and
  :ref:`facetSRPSelfOcclusionEffectorBase` provide the input messages, the source direction and
  pressure, the facet coefficients with ``addFacet()``, and the per-facet force law.
- The concrete modules provide the occlusion algorithm. Each algorithm is a plain C++ class in this
  folder, shared by the drag and SRP modules that use it. :ref:`facetDragSelfOcclusionPolygonClippingEffector`
  and :ref:`facetSRPSelfOcclusionPolygonClippingEffector` use ``PolygonClippingOcclusion``, exact
  polygon clipping with inclusion-exclusion.

Self-Occlusion Model
--------------------
Every facet is a planar rectangle centered at ``r_CopB_B`` with unit normal ``nHat_B``. Its
in-plane edges are taken from the optional ``width`` and ``height`` arguments of ``addFacet()``.
If both are left at their default value of ``-1``, the facet is a square of side
:math:`\sqrt{A}`. If both are given, ``width * height`` must equal the facet area.

At each force evaluation every facet rectangle is orthographically projected onto the plane normal to
the source direction :math:`\hat{\mathbf{d}}_B`, using an orthonormal in-plane basis
:math:`(\mathbf{p}_1, \mathbf{p}_2)`. The source direction points from the spacecraft toward the
incoming flux, so the flux propagates along :math:`-\hat{\mathbf{d}}_B`. Each facet is planar, so
the depth, measured along the propagation direction, of the facet point that projects to
:math:`(x, y)` is an affine function:

.. math::

    z_i(x, y) = \frac{x\, (\hat{\mathbf{n}}_i \cdot \mathbf{p}_1) + y\, (\hat{\mathbf{n}}_i \cdot \mathbf{p}_2)
        - \hat{\mathbf{n}}_i \cdot \mathbf{c}_i}{\hat{\mathbf{n}}_i \cdot \hat{\mathbf{d}}_B}

where :math:`\mathbf{c}_i` is the facet center, and the facet point itself is
:math:`x \mathbf{p}_1 + y \mathbf{p}_2 - z_i \hat{\mathbf{d}}_B`. Where several silhouettes overlap,
the facet with the smallest depth is the closest to the source, and is the visible (upstream) one.
A facet only receives force when it faces the source
(:math:`\hat{\mathbf{n}}_i \cdot \hat{\mathbf{d}}_B > 0`), but every facet, whichever way it faces,
is an occluder of the other facets.

The occlusion algorithm returns, for each facet, the exposure

.. math::

    \eta_i = \frac{A_{i,\text{visible}}}{A_i\, |\hat{\mathbf{n}}_i \cdot \hat{\mathbf{d}}_B|}

and the center of pressure :math:`\mathbf{r}_{\text{cp},i}`, the centroid of the visible region
mapped back onto the facet plane.

Force and Torque
----------------
With :math:`\cos\theta_i = \hat{\mathbf{n}}_i \cdot \hat{\mathbf{d}}_B`, each facet with
:math:`\cos\theta_i > 0` and a nonzero exposure contributes the force :math:`\mathbf{F}_i` of the flux
model on its visible projected area :math:`A_{i,\text{exp}} = A_i \cos\theta_i\, \eta_i`. The force
laws are given in :ref:`facetDragSelfOcclusionEffectorBase` and
:ref:`facetSRPSelfOcclusionEffectorBase`. The torque about point B is
:math:`\sum_i \mathbf{r}_{\text{cp},i} \times \mathbf{F}_i`. The flux model may scale the total force
and torque, which the SRP model uses for the eclipse illumination factor.

On a convex body no facet is shadowed, every :math:`\eta_i = 1` and every
:math:`\mathbf{r}_{\text{cp},i}` is the facet center, so the model reduces to
:ref:`facetDragDynamicEffector` and to a non-articulated :ref:`facetSRPDynamicEffector`.

Visibility Cache
----------------
The visibility only depends on the source direction, while its cost grows with the number of facets.
``setDirectionCacheTolerance()`` (default :math:`10^{-5}` rad) reuses the previous visibility
whenever the source direction has rotated less than this angle since it was last computed. The facet
cosines, the pressure and the total force scale are still recomputed at every force evaluation. Set
the tolerance to 0 to recompute the visibility at every evaluation. Adding a facet or changing an
algorithm parameter invalidates the cache.

Attaching to a State Effector
-----------------------------
The effector can be attached to the hub with ``scObject.addDynamicEffector()``, or to a state
effector that supports the branching described in :ref:`bskPrinciples-11`. When it is attached to a
state effector, the B frame of these pages is the parent segment frame, whose inertial attitude and
position or velocity properties are used in place of the hub states.

Module Assumptions and Limitations
----------------------------------
- Facets are rigidly fixed in the frame the effector is attached to. Unlike
  :ref:`facetSRPDynamicEffector`, individually articulated facets are not supported; attach a
  separate effector to an articulated state effector instead.
- Occlusion is only computed between facets of the same effector.
- Facets are opaque, flat rectangles. Multiple reflections and shadowing of reflected radiation are
  not modeled.

Adding a New Occlusion Algorithm
--------------------------------
Implement the algorithm once, as a plain class in ``_GeneralModuleFiles`` with the same method as the
existing ones:

.. code-block:: cpp

    void computeVisibility(const facetSelfOcclusion::FacetGeometry& geometry,
                           const facetSelfOcclusion::FacetProjection& projection,
                           std::vector<facetSelfOcclusion::FacetVisibility>& visibility,
                           BSKLogger& bskLogger);

Then add one thin module per flux model, deriving from ``FacetDragSelfOcclusionEffectorBase`` or
``FacetSRPSelfOcclusionEffectorBase``, that holds the algorithm and overrides
``computeFacetVisibility()`` with a single call to it.

- ``geometry`` holds the facet areas, normals, centers and rectangle half-edges.
- ``projection`` holds the source direction, the projection basis and, for every facet, the projected
  corners, the projected area, the signed source cosine, the affine depth-plane coefficients and the
  projected bounding box.
- ``visibility`` is sized to the number of facets and initialized to zero exposure before the call.
  Only the entries of facets facing the source are read.

The data types and the shared helpers ``cross2``, ``pointInConvexPolygon`` and ``depthAt`` are in
``facetSelfOcclusionGeometry.h``. A module whose parameters change the visibility result must call
``invalidateVisibilityCache()`` when they are modified.
