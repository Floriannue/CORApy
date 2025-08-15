"""
le - Subset test for polytopes (MATLAB parity)

Implements P <= Q as (Q contains P).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope


def le(P: 'Polytope', Q: 'Polytope', tol: float = 1e-10) -> bool:
    res, _, _ = Q.contains_(P, 'exact', tol)
    return bool(res)


