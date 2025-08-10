"""
priv_minkDiffEllipsoid - Inner/outer approximation of Minkowski difference between two ellipsoids

Syntax:
   E = priv_minkDiffEllipsoid(E1,E2,L,mode)
"""

from __future__ import annotations

import numpy as np

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_minkDiffEllipsoid(E1: Ellipsoid, E2: Ellipsoid, L: np.ndarray, mode: str) -> Ellipsoid:
    # Rank/degeneracy cases
    if E1.rank() == 0 and E2.rank() == 0:
        return Ellipsoid(np.zeros((E1.dim(), E1.dim())), E1.q - E2.q)
    if E1.rank() == 0 and E2.rank() > 0:
        return Ellipsoid.empty(E1.dim())
    if E1.rank() > 0 and E2.rank() == 0:
        return Ellipsoid(E1.Q, E1.q - E2.q)

    # Containment check required by MATLAB algorithm
    if not E1.isBigger(E2):
        return Ellipsoid.empty(E1.dim())

    n = E1.dim()
    TOL = min(E1.TOL, E2.TOL)
    if np.allclose(E1.Q, E2.Q, atol=TOL, rtol=0):
        return Ellipsoid(np.zeros((n, n)), E1.q - E2.q)

    # Bad directions filter or generate defaults
    def is_bad_dir(Ldir: np.ndarray) -> np.ndarray:
        return E1.isBadDir(E2, Ldir)

    if L.size == 0:
        N = 2 * n
        if n == 1:
            L = np.array([[-1.0, 1.0]]).reshape(1, 2)
            mask = ~is_bad_dir(L)
            if isinstance(mask, np.ndarray):
                mask = mask.astype(bool).ravel()
            L = L[:, mask]
        else:
            dirs = []
            counter = 0
            while len(dirs) < N and counter < 10000:
                v = np.random.randn(n, 1)
                v = v / (np.linalg.norm(v) + 1e-16)
                if not is_bad_dir(v):
                    dirs.append(v)
                counter += 1
            if len(dirs) < N:
                basis = np.eye(n)
                dirs = [basis[:, [i]] for i in range(n)] + [-basis[:, [i]] for i in range(n)]
            L = np.concatenate(dirs, axis=1)[:, :N]
    else:
        mask = ~is_bad_dir(L)
        # Ensure 1D boolean mask for column selection
        if isinstance(mask, np.ndarray):
            mask = mask.astype(bool).ravel()
        L = L[:, mask]
        if L.size == 0:
            raise CORAerror('CORA:emptySet')

    # Per-direction Minkowski difference
    from .priv_lminkDiff import priv_lminkDiff
    cells = priv_lminkDiff(E1, E2, L, mode)
    if len(cells) == 1:
        return cells[0]
    if mode == 'outer':
        from cora_python.contSet.ellipsoid.and_ import and_ as ellipsoid_and
        return ellipsoid_and(cells[0], cells[1:], 'outer')
    else:
        from cora_python.contSet.ellipsoid.or_ import or_ as ellipsoid_or
        return ellipsoid_or(cells[0], cells[1:], 'inner')

