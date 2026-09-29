Executive Summary
-----------------
Abstract base class of the faceted, flat-plate solar radiation pressure (SRP) effectors with planar
self-occlusion, such as :ref:`facetSRPSelfOcclusionPolygonClippingEffector`.

It derives from :ref:`facetSelfOcclusionEffectorBase`, which describes the facet geometry, the
self-occlusion model, the visibility cache and the attachment options. This class adds the SRP flux
model: the Sun ephemeris and eclipse input messages, the Sun direction used as source direction, the
facet optical coefficients, the SRP force law and the eclipse scaling. The concrete modules only add
the occlusion algorithm.

Adding Facets
-------------
``addFacet(area, diffuseCoeff, specularCoeff, nHat_B, r_CopB_B, width=-1, height=-1)`` adds a facet of
area ``area`` [m^2], diffuse and specular reflection coefficients ``diffuseCoeff`` and
``specularCoeff`` [-], each in [0, 1], unit normal ``nHat_B`` and center ``r_CopB_B`` [m] in the
effector frame. The optional ``width`` and ``height`` [m] define the rectangle, as described in
:ref:`facetSelfOcclusionEffectorBase`. The coefficients have the same meaning as in
:ref:`facetSRPDynamicEffector`.

Source Direction and SRP Force
------------------------------
The source direction is the Sun direction
:math:`\hat{\mathbf{d}}_B = [BN] (\mathbf{r}_{S/N} - \mathbf{r}_{B/N}) / r`, using the Sun position
from ``sunInMsg``, where :math:`r` is the spacecraft-to-Sun distance. With the facet diffuse and
specular coefficients :math:`D_i` and :math:`S_i` and
:math:`\cos\theta_i = \hat{\mathbf{n}}_i \cdot \hat{\mathbf{d}}_B`, the force on the visible projected
area :math:`A_{i,\text{exp}}` of a facet is

.. math::

    \mathbf{F}_i = -P\, A_{i,\text{exp}}
        \left[ (1 - S_i)\, \hat{\mathbf{d}}_B
              + 2 \left(\frac{D_i}{3} + S_i \cos\theta_i\right) \hat{\mathbf{n}}_i \right],
    \qquad
    P = \frac{F_\odot}{c} \left(\frac{1\ \text{AU}}{r}\right)^2

where :math:`F_\odot` is the solar radiation flux at 1 AU and :math:`c` is the speed of light. They are
the Basilisk constants ``SOLAR_FLUX_EARTH`` and ``SPEED_LIGHT``, and 1 AU is ``AU2M``, all
from ``architecture/utilities/astroConstants.h``. :ref:`facetSRPDynamicEffector` uses its own
1368 W/m^2 flux, so for the same unshadowed geometry its forces are different.

The total force and torque are scaled by the ``illuminationFactor`` of ``sunEclipseInMsg`` when it is
linked and written, and by 1 otherwise. Without a valid Sun sample the
effector applies no force. ``getSunDistance()`` returns the Sun distance of the last force
evaluation.

Input Message Timing
--------------------
``sunInMsg`` and ``sunEclipseInMsg`` are read only during ``UpdateState()``. The spacecraft calls
``computeForceTorque()`` during its integration, before the environment models are updated, so the
force always uses the message values of the previous time step, held constant over each integration
step. Add the effector to a task so that ``UpdateState()`` is called.

Initialization and Reset
------------------------
Attaching the effector to the hub or to a state effector, and ``Reset()``, check that ``sunInMsg`` is
linked. ``Reset()`` also restores full illumination until the next eclipse message is read. It does
not clear the facets or the visibility cache.
