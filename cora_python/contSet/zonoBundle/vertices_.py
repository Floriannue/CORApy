"""
vertices_ - Returns potential vertices of a zonotope bundle
   WARNING: Do not use this function for high order zonotope bundles as
   the computational complexity grows exponentially!

Syntax:
    V = vertices_(zB)

Inputs:
    zB - zonoBundle object

Outputs:
    V - matrix storing the vertices 

Example: 
    Z1 = zonotope([1;1], [1 1; -1 1]);
    Z2 = zonotope([-1;1], [1 0; 0 1]);
    zB = zonoBundle({Z1,Z2});
    V = vertices(zB);
 
    figure; hold on;
    plot(zB);
    plot(V(1,:),V(2,:),'k.','MarkerSize',12);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/vertices, polytope

Authors:       Matthias Althoff
Written:       18-August-2016 
Last update:   ---
Last revision: 27-March-2023 (MW, rename vertices_)
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, TYPE_CHECKING
import numpy as np

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


def vertices_(zB: 'ZonoBundle', *args: Any) -> np.ndarray:
    """
    Returns potential vertices of a zonotope bundle.
    """
    if zB.dim() == 2:
        return _aux_vertices2D(zB)

    # obtain polytope
    P = zB.polytope()

    # obtain vertices (input check in polytope-function)
    V = P.vertices_()

    # return correct dimension
    if V.size == 0:
        V = np.zeros((zB.dim(), 0))

    return V


def _aux_vertices2D(zB: 'ZonoBundle') -> np.ndarray:
    tol = 1e-7
    
    def _fallback_polytope():
        P = zB.polytope()
        Vp = P.vertices_()
        if Vp.size == 0:
            return np.zeros((2, 0))
        return Vp
    
    # Single zonotope: return its convex hull vertices directly
    if zB.parallelSets == 1 and len(zB.Z) == 1:
        V0 = zB.Z[0].vertices_()
        if V0.size == 0:
            return _fallback_polytope()
        try:
            hull = ConvexHull(V0.T)
            return V0[:, hull.vertices]
        except Exception:
            return V0

    polygons = []
    for i in range(zB.parallelSets):
        # delete zero generators
        Z = zB.Z[i].compact_('zeros', tol)

        # get polygon from vertices
        V = Z.vertices_()
        if V.size == 0:
            return _fallback_polytope()

        # Ensure proper vertex ordering for polygon construction
        try:
            hull = ConvexHull(V.T)
            V_ordered = V[:, hull.vertices]
        except Exception:
            V_ordered = V

        pgon = ShapelyPolygon(V_ordered.T)
        if not pgon.is_valid:
            # Fix invalid polygon to match MATLAB's robust intersection behavior
            pgon = pgon.buffer(0)
        if pgon.is_empty:
            return _fallback_polytope()
        # Ensure we keep a single convex polygon for intersection
        if pgon.geom_type != 'Polygon':
            pgon = pgon.convex_hull
        if pgon.is_empty or pgon.geom_type != 'Polygon':
            return _fallback_polytope()
        polygons.append(pgon)

    if len(polygons) == 0:
        return np.zeros((2, 0))

    # intersect polygons
    pint = polygons[0]
    for i in range(1, len(polygons)):
        try:
            pint = pint.intersection(polygons[i])
        except Exception:
            # Fix invalid geometries and retry intersection
            pint = pint.buffer(0)
            other = polygons[i].buffer(0)
            pint = pint.intersection(other)

    if pint.is_empty:
        return _fallback_polytope()

    if pint.geom_type == 'Polygon':
        coords = np.array(pint.exterior.coords)
        if coords.shape[0] <= 1:
            return _fallback_polytope()
        return coords[:-1].T

    if pint.geom_type in ['MultiPolygon', 'GeometryCollection']:
        merged = unary_union(pint)
        if merged.is_empty:
            return _fallback_polytope()
        if merged.geom_type != 'Polygon':
            merged = merged.convex_hull
        if merged.is_empty or merged.geom_type != 'Polygon':
            return _fallback_polytope()
        coords = np.array(merged.exterior.coords)
        if coords.shape[0] <= 1:
            return _fallback_polytope()
        return coords[:-1].T

    return np.zeros((2, 0))

