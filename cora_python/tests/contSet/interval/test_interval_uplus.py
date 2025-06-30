"""
Test file for interval uplus method - translated from MATLAB

Authors: Dmitry Grebenyuk, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-January-2016 (MATLAB)
Last update: 04-December-2023 (MW, add empty and unbounded cases) (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.uplus import uplus
from cora_python.contSet.interval.isequal import isequal
from cora_python.contSet.interval.representsa_ import representsa_


def test_interval_uplus():
    """Test interval uplus method - translated from MATLAB"""
    
    tol = 1e-9
    
    # empty
    I = Interval.empty(2)
    I_uplus = uplus(I)
    assert representsa_(I_uplus, 'emptySet')
    
    # bounded
    I = Interval([-5, -4, -3, 0, 0, 5], [-2, 0, 2, 0, 5, 8])
    I_uplus = uplus(I)
    assert isequal(I_uplus, I, tol)
    
    # unbounded
    I = Interval([-np.inf, 2], [1, np.inf])
    I_uplus = uplus(I)
    assert isequal(I_uplus, I, tol)
    
    # test that uplus returns the same object (identity)
    I = Interval([1, 2], [3, 4])
    I_uplus = uplus(I)
    assert I_uplus is I  # Should be the same object
    
    # test with point interval
    I = Interval([1, 2], [1, 2])
    I_uplus = uplus(I)
    assert isequal(I_uplus, I, tol)


if __name__ == "__main__":
    test_interval_uplus()
    print("All uplus tests passed!") 