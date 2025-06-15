"""
Test file for interval randPoint_ method - translated from MATLAB

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 21-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.randPoint_ import randPoint_
from cora_python.contSet.interval.contains_ import contains_


def test_interval_randPoint():
    """Test interval randPoint_ method - translated from MATLAB"""
    
    # empty case
    n = 2
    I = Interval.empty(n)
    p = randPoint_(I)
    assert p.size == 0 and p.shape[0] == n
    
    # 2D
    lb = np.array([-2, -1])
    ub = np.array([1, 0])
    I = Interval(lb, ub)
    
    # Single point
    p = randPoint_(I)
    assert contains_(I, p)
    
    # Multiple points
    p = randPoint_(I, 10)
    assert np.all(contains_(I, p))
    
    # Extreme point
    p = randPoint_(I, 1, 'extreme')
    assert contains_(I, p)


if __name__ == '__main__':
    test_interval_randPoint()
    print("All tests passed!") 