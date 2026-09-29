Executive Summary
-----------------
Abstract base class of the faceted, flat-plate atmospheric drag effectors with planar self-occlusion, such as
:ref:`facetDragSelfOcclusionPolygonClippingEffector`.

It derives from :ref:`facetSelfOcclusionEffectorBase`, which describes the facet geometry, the
self-occlusion model, the visibility cache and the attachment options. This class adds the drag flux
model: the atmosphere and wind input messages, the relative velocity used as source direction, the
facet drag coefficients and the drag force law. The concrete modules only add the occlusion
algorithm.

Adding Facets
-------------
``addFacet(area, dragCoeff, nHat_B, r_CopB_B, width=-1, height=-1)`` adds a facet of area ``area``
[m^2], drag coefficient ``dragCoeff`` [-], unit normal ``nHat_B`` and center ``r_CopB_B`` [m] in the
effector frame. The optional ``width`` and ``height`` [m] define the rectangle, as described in
:ref:`facetSelfOcclusionEffectorBase`.

Source Direction and Drag Force
-------------------------------
The source direction is the direction of the atmosphere-relative velocity

.. math::

    \mathbf{v}_B = [BN] (\mathbf{v}_{B/N} - \mathbf{v}_{\text{air}}), \qquad
    \hat{\mathbf{d}}_B = \frac{\mathbf{v}_B}{v}

where the air velocity is read from ``windVelInMsg`` when it is linked and written (for example from
:ref:`zeroWindModel`), and is zero otherwise. With the atmospheric density :math:`\rho` from
``atmoDensInMsg`` and the facet drag coefficient :math:`C_{D,i}`, the force on the visible projected
area :math:`A_{i,\text{exp}}` of a facet is

.. math::

    \mathbf{F}_i = -\frac{1}{2} \rho v^2\, C_{D,i}\, A_{i,\text{exp}}\, \hat{\mathbf{d}}_B

Without a valid atmosphere sample, or at zero relative speed, the effector applies no force.
``getRelativeVelocity_B()`` returns the relative velocity of the last force evaluation.

Input Message Timing
--------------------
``atmoDensInMsg`` and ``windVelInMsg`` are read only during ``UpdateState()``. The spacecraft calls
``computeForceTorque()`` during its integration, before the environment models are updated, so the
force always uses the message values of the previous time step, held constant over each integration
step. Add the effector to a task so that ``UpdateState()`` is called.

Initialization and Reset
------------------------
Attaching the effector to the hub or to a state effector, and ``Reset()``, check that
``atmoDensInMsg`` is linked. ``Reset()`` does not clear the facets or the visibility cache.
