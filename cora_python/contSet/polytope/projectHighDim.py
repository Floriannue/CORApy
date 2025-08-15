"""
projectHighDim - Embed a polytope into a higher-dimensional space

Mirrors MATLAB @polytope/projectHighDim:
- N: target ambient dimension
- proj: list of dimensions in the target space where the original coordinates are placed
  (length equals original dimension). Remaining dimensions become equality-fixed to zero.
Works for both H-rep and V-rep.
"""

import numpy as np
from typing import Sequence, TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
	from .polytope import Polytope


def projectHighDim(P: 'Polytope', N: int, proj: Sequence[int] | None = None) -> 'Polytope':
	from .polytope import Polytope

	n = P.dim()
	if N < n:
		raise CORAerror('CORA:wrongInput', 'Dimension of higher-dimensional space must be larger than or equal to original dimension')

	if proj is None:
		proj = list(range(1, n + 1))
	proj = list(proj)

	if len(proj) != n:
		raise CORAerror('CORA:wrongInput', 'Number of dimensions in higher-dimensional space must match original dimension')
	if max(proj) > N or min(proj) < 1:
		raise CORAerror('CORA:wrongInput', 'Projection indices must be within 1..N')

	# Map from original coords (1..n) to new coords (1..N)
	# Build selection matrix S (N x n) with ones at (proj[i]-1, i)
	S = np.zeros((N, n))
	for i, p in enumerate(proj):
		S[p - 1, i] = 1.0

	# If P is in V-rep, map vertices directly: V_high = S @ V
	if P.isVRep:
		V = P.V
		V_high = S @ V
		return Polytope(V_high)

	# Otherwise, map H-representation: x in R^n, y in R^N with y = S x and remaining coords fixed to 0
	P.constraints()
	A, b, Ae, be = P.A, P.b, P.Ae, P.be

	# Inequalities: A (S^T y) <= b -> (A S^T) y <= b
	AS = A @ S.T if A.size > 0 else np.zeros((0, N))
	b_out = b

	# Equalities from original: Ae (S^T y) = be -> (Ae S^T) y = be
	Ae_out = Ae @ S.T if Ae.size > 0 else np.zeros((0, N))
	be_out = be

	# Additional equalities to fix non-projected coordinates to zero
	fixed = np.ones(N, dtype=bool)
	fixed[np.array(proj) - 1] = False
	if np.any(fixed):
		E_fix = np.eye(N)[fixed]
		Ae_out = np.vstack([Ae_out, E_fix]) if Ae_out.size > 0 else E_fix
		be_fix = np.zeros((E_fix.shape[0], 1))
		be_out = np.vstack([be_out, be_fix]) if be_out.size > 0 else be_fix

	return Polytope(AS, b_out, Ae_out, be_out)
