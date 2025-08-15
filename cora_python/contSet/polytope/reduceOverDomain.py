"""
reduceOverDomain - Reduce inequality constraints over a domain (MATLAB parity)

Reduces a list of similar halfspaces A x <= b to a single inequality that
inner- or outer-approximates them over a given domain (interval). Equality
constraints are not supported.
"""

import numpy as np
from typing import TYPE_CHECKING, Literal

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
	from .polytope import Polytope
	from cora_python.contSet.interval.interval import Interval


def reduceOverDomain(P: 'Polytope', dom: 'Interval', type: Literal['inner','outer']='outer') -> 'Polytope':
	from cora_python.contSet.interval.interval import Interval
	from .polytope import Polytope
	from .private.priv_normalizeConstraints import priv_normalizeConstraints

	if type not in ('inner', 'outer'):
		raise CORAerror('CORA:wrongInput', 'type must be "inner" or "outer"')

	# Convert domain to interval if needed (outer approximation)
	if not isinstance(dom, Interval):
		dom = Interval(dom)

	# Normalize constraints; equality constraints not supported for reduction
	A, b, Ae, be = priv_normalizeConstraints(P.A, P.b, P.Ae, P.be, 'A')
	if Ae.size > 0:
		raise CORAerror('CORA:wrongInput', 'Equality constraints not supported in reduceOverDomain')

	if A.size == 0:
		# Nothing to reduce
		return Polytope(A, b)

	# Intervals for normal vector components and offsets across constraints
	# For each column i: take min/max across rows j
	A_min = np.min(A, axis=0).reshape(-1, 1)
	A_max = np.max(A, axis=0).reshape(-1, 1)
	b_min = np.min(b)
	b_max = np.max(b)

	# Build interval objects
	A_int = Interval(A_min, A_max)  # each component has an interval
	b_int = Interval(np.array([[b_min]]), np.array([[b_max]]))

	# Reduced normal: center of A interval
	A_red = A_int.center()
	# Reduced offset depends on inner/outer
	if type == 'outer':
		# MATLAB: b_reduced = b - (A - A_reduced)' * dom
		b_red = b_int - (A_int - A_red).T * dom
	else:
		# Inner: conservative counterpart
		b_red = b_int + (A_int - A_red).T * dom

	# b_red is an interval; for outer we take its upper bound, for inner its lower bound
	if type == 'outer':
		b_scalar = b_red.sup.flatten()[0]
	else:
		b_scalar = b_red.inf.flatten()[0]

	return Polytope(A_red.T, np.array([[b_scalar]]))
