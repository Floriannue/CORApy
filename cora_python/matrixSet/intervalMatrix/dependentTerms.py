"""
dependentTerms - computes exact Taylor terms of an interval matrix square
   and an interval matrix exponential

Syntax:
    [intSq, intH] = dependentTerms(intMat, r)

Inputs:
    intMat - intervalMatrix object
    r - time step increment

Outputs:
    intSq - exact square matrix
    intH - exact Taylor terms up to second order

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING
from cora_python.contSet.interval.interval import Interval

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def dependentTerms(intMat: 'IntervalMatrix', r: float) -> Tuple['IntervalMatrix', 'IntervalMatrix']:
    """
    Computes exact Taylor terms of an interval matrix square and exponential
    
    Args:
        intMat: intervalMatrix object
        r: time step increment
        
    Returns:
        intSq: Exact square interval matrix
        intH: Exact Taylor terms up to second order (interval matrix)
    """
    # Load data from object structure
    # MATLAB: A=obj.int;
    A = intMat.int
    # MATLAB: n=dim(obj,1);
    from .dim import dim
    n = dim(intMat)[0]
    
    # Initialize the square of A (sq), the first term of the interval exponential (H)
    # MATLAB: sq=0*A;
    # MATLAB: H=0*A;
    from cora_python.contSet.interval.interval import Interval
    sq = Interval(np.zeros((n, n)), np.zeros((n, n)))
    H = Interval(np.zeros((n, n)), np.zeros((n, n)))
    I = np.eye(n)
    
    # Get diagonal elements of A (diagA)
    # MATLAB: diagA=interval(zeros(n),zeros(n)); %initialize
    diagA = Interval(np.zeros((n, n)), np.zeros((n, n)))
    # MATLAB: for i=1:n
    for i in range(n):
        # MATLAB: diagA(i,i)=A(i,i);
        diagA.inf[i, i] = A.inf[i, i]
        diagA.sup[i, i] = A.sup[i, i]
    
    # Compute elements of sq and H
    # MATLAB: for i=1:n
    for i in range(n):
        # i neq j
        # Auxiliary value s
        # MATLAB: s=aux_sum(A,i);
        s = _aux_sum(A, i, n)
        
        # Auxiliary value b
        # MATLAB: b=A(i,:); b(i)=0;
        b = Interval(A.inf[i, :].copy(), A.sup[i, :].copy())
        b.inf[i] = 0.0
        b.sup[i] = 0.0
        
        # Auxiliary matrix C
        # MATLAB: C=I*A(i,i)+diagA;
        A_ii = Interval(A.inf[i, i], A.sup[i, i])
        C = I * A_ii + diagA
        
        # Compute non-diagonal elements of sq
        # MATLAB: sq(i,:)=b*C+s;
        sq_row = b * C + s
        sq.inf[i, :] = sq_row.inf
        sq.sup[i, :] = sq_row.sup
        
        # Compute non-diagonal elements of H
        # MATLAB: H(i,:)=b*(I*r+0.5*C*r^2)+0.5*r^2*s;
        H_row = b * (I * r + 0.5 * C * r ** 2) + 0.5 * r ** 2 * s
        H.inf[i, :] = H_row.inf
        H.sup[i, :] = H_row.sup
        
        # i=j
        # Compute diagonal elements of sq
        # MATLAB: sq(i,i)=sq(i,i)+A(i,i)^2;
        A_ii_sq = A_ii ** 2
        sq.inf[i, i] = sq.inf[i, i] + A_ii_sq.inf
        sq.sup[i, i] = sq.sup[i, i] + A_ii_sq.sup
        
        # Auxiliary values for H
        # MATLAB: a_inf=infimum(A(i,i));
        # MATLAB: a_sup=supremum(A(i,i));
        a_inf = A_ii.infimum()
        a_sup = A_ii.supremum()
        
        # Compute diagonal elements for H
        # MATLAB: kappa=max(a_inf*r+0.5*a_inf^2*r^2, a_sup*r+0.5*a_sup^2*r^2);
        kappa = max(a_inf * r + 0.5 * a_inf ** 2 * r ** 2, 
                    a_sup * r + 0.5 * a_sup ** 2 * r ** 2)
        # MATLAB: H(i,i)=H(i,i)+interval(aux_g(A(i,i),r),kappa);
        aux_g_val = _aux_g(A_ii, r)
        H.inf[i, i] = H.inf[i, i] + aux_g_val
        H.sup[i, i] = H.sup[i, i] + kappa
    
    # Write as interval matrices
    # MATLAB: intSq=intervalMatrix(r^2*center(sq),r^2*rad(sq));
    # MATLAB: intH=intervalMatrix(center(H)+I,rad(H));
    from .intervalMatrix import IntervalMatrix
    intSq = IntervalMatrix(r ** 2 * sq.center(), r ** 2 * sq.rad())
    intH = IntervalMatrix(H.center() + I, H.rad())
    
    return intSq, intH


def _aux_g(a: Interval, r: float) -> float:
    """
    Auxiliary function for computing diagonal elements of H
    """
    from cora_python.contSet.interval.isIntersecting_ import isIntersecting_
    from cora_python.contSet.interval.interval import Interval
    
    # MATLAB: if isIntersecting(interval(-1/r,-1/r),a)
    if isIntersecting_(Interval(-1.0 / r, -1.0 / r), a):
        # MATLAB: res=-0.5;
        return -0.5
    else:
        # MATLAB: a_inf=infimum(a);
        # MATLAB: a_sup=supremum(a);
        a_inf = a.infimum()
        a_sup = a.supremum()
        # MATLAB: res=min(a_inf*r+0.5*a_inf^2*r^2, a_sup*r+0.5*a_sup^2*r^2);
        return min(a_inf * r + 0.5 * a_inf ** 2 * r ** 2,
                   a_sup * r + 0.5 * a_sup ** 2 * r ** 2)


def _aux_sum(A: Interval, i: int, n: int) -> Interval:
    """
    Sum function: s=0.5 \sum_{k:k\neq i,k\neq j} a_{ik}a_{kj}t^2
    """
    # MATLAB: A = A .* (1-eye(n));
    # Create mask to zero diagonal
    mask = 1 - np.eye(n)
    A_masked = Interval(A.inf * mask, A.sup * mask)
    
    # MATLAB: s=A(i,:)*A;
    # Compute row i times matrix A
    A_i = Interval(A_masked.inf[i, :], A_masked.sup[i, :])
    s = A_i * A_masked
    
    return s
