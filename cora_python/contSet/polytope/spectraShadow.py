"""
spectraShadow - Converts a polytope to a spectrahedral shadow (full MATLAB parity)

Implements the MATLAB logic from @polytope/spectraShadow.m:
- Build constraints Cx <= d where C stacks inequalities and equalities as in MATLAB
- Construct block-diagonal A0 and diagonal Ai for each variable as in the paper
- Instantiate SpectraShadow(A) or SpectraShadow.Inf(n) if no constraints
- Set additional properties (bounded, emptySet, fullDim, center)
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope


def spectraShadow(P: 'Polytope') -> SpectraShadow:
    # Deduce ambient dimension
    n = P.dim()

    # Ensure constraints are available (computes if needed)
    P.constraints()

    # Read constraints Cx <= d with equalities duplicated as +/-
    A = P.A; b = P.b
    Ae = P.Ae; be = P.be

    # Normalize shapes
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if be.ndim == 1:
        be = be.reshape(-1, 1)

    C_parts = []
    d_parts = []
    if A.size > 0:
        C_parts.append(A)
        d_parts.append(b)
    if Ae.size > 0:
        C_parts.append(Ae)
        d_parts.append(be)
        C_parts.append(-Ae)
        d_parts.append(-be)

    if len(C_parts) == 0:
        # No constraints -> fullspace spectrahedral shadow
        A_empty = np.zeros((0, 0))
        c = np.zeros((n, 1))
        G = np.eye(n)
        SpS = SpectraShadow(A_empty, c, G)
        # Properties
        SpS._bounded_val = False
        SpS._emptySet_val = P.representsa_('emptySet', 1e-10)
        SpS._fullDim_val = P.isFullDim()
        SpS.center.val = P.center()
        return SpS

    C = np.vstack(C_parts) if len(C_parts) > 1 else C_parts[0]
    d = np.vstack(d_parts) if len(d_parts) > 1 else d_parts[0]

    # Build A0 and Ai per MATLAB code
    # Start with empty blocks
    k_total = 0
    A0_blocks = []
    Ai_blocks = [[] for _ in range(n)]

    for j in range(C.shape[0]):
        # Append 1x1 block [d_j] to A0
        A0_blocks.append(np.array([[d[j, 0]]]))
        # Append -C(j,i) to Ai diagonal positions
        for i in range(n):
            Ai_blocks[i].append(np.array([[-C[j, i]]]))
        k_total += 1

    # Assemble block diagonal A0
    if len(A0_blocks) == 1:
        A0 = A0_blocks[0]
    else:
        A0 = np.block([[A0_blocks[i] if i == j else np.zeros((A0_blocks[i].shape[0], A0_blocks[j].shape[1]))
                        for j in range(len(A0_blocks))] for i in range(len(A0_blocks))])

    # Assemble Ai as block-diagonal matrices of same size as A0
    Ai_mats = []
    for i in range(n):
        blocks = Ai_blocks[i]
        if len(blocks) == 1:
            Ai_mats.append(blocks[0])
        else:
            Ai_mats.append(np.block([[blocks[p] if p == q else np.zeros((blocks[p].shape[0], blocks[q].shape[1]))
                                      for q in range(len(blocks))] for p in range(len(blocks))]))

    # Concatenate A = [A0 A1 ... An]
    A_concat = np.concatenate([A0] + Ai_mats, axis=1)

    # Instantiate spectraShadow
    SpS = SpectraShadow(A_concat)

        # Set additional properties from P (MATLAB parity)
    SpS._bounded_val = P.isBounded()
    SpS._emptySet_val = P.representsa_('emptySet', 1e-10)
    SpS._fullDim_val = P.isFullDim()
    # For equalities-only single point, set center directly (no try/except)
    if A.size == 0 and Ae.size > 0 and Ae.shape[0] == n:
        if np.linalg.matrix_rank(Ae) == n:
            x_eq = np.linalg.solve(Ae, be).reshape(-1, 1)
            SpS.center.val = x_eq
            SpS._emptySet_val = False
            SpS._bounded_val = True
            SpS._fullDim_val = False
        else:
            SpS.center.val = P.center()
    else:
        SpS.center.val = P.center()

    return SpS


