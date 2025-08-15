"""
isequal - Check equality of two polytopes (MATLAB parity style)

Returns True if both sets are equal within tolerance. Implements via mutual
containment after normalizing/compacting constraints.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope


def isequal(P: 'Polytope', Q: 'Polytope', tol: float = 1e-10) -> bool:
    # Fast emptiness and dimension checks
    if P.representsa_('emptySet', tol) and Q.representsa_('emptySet', tol):
        return True
    if P.dim() != Q.dim():
        return False

    # Mutual containment using exact method
    r1, _, _ = Q.contains_(P, 'exact', tol)
    r2, _, _ = P.contains_(Q, 'exact', tol)
    return bool(r1) and bool(r2)


