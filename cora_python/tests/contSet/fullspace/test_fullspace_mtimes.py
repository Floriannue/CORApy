"""
test_fullspace_mtimes - unit test function of mtimes

Syntax:
   res = test_fullspace_mtimes

Inputs:
   -

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger
Written:       05-April-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace import Fullspace
from cora_python.contSet.interval import Interval

def test_fullspace_mtimes():
    """Test the mtimes method of fullspace"""
    # init fullspace
    n = 3
    fs = Fullspace(n)
    
    # multiplication with scalar
    p = 2
    fs_ = fs * p
    assert fs == fs_
    fs_ = p * fs
    assert fs == fs_
    
    # multiplication with full-rank square matrix (still fullspace)
    M = np.array([[2, 1, 0], [-1, -1, 2], [1, 2, 1]])
    fs_ = M @ fs
    assert fs == fs_
    
    # multiplication as projection
    M = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0]])
    fs_ = M @ fs
    expected = Interval(np.array([-np.inf, -np.inf, 0]), np.array([np.inf, np.inf, 0]))
    assert fs_ == expected
    
    # multiplication as projection onto higher-dimensional space
    M = np.vstack([np.eye(n), np.array([[1, 0, 0]])])
    fs_ = M @ fs
    assert fs_ == Fullspace(n + 1)
    

# ------------------------------ END OF CODE ------------------------------ 