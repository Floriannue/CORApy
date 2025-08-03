"""
test_zonotope_interval - unit test function of interval

Syntax:
    python -m pytest test_zonotope_interval.py

Inputs:
    -

Outputs:
    test results

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 26-July-2016 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeInterval:
    """Test class for zonotope interval method"""
    
    def test_basic_interval_conversion(self):
        """Test basic conversion to interval"""
        # Create zonotope
        Z1 = Zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        
        # Create interval
        I1 = Z1.interval()
        
        # Obtain results
        lb = I1.infimum()
        ub = I1.supremum()
        
        # True results
        true_lb = np.array([-10, -8])
        true_ub = np.array([2, 10])
        
        # Check result
        np.testing.assert_array_almost_equal(lb, true_lb)
        np.testing.assert_array_almost_equal(ub, true_ub)


if __name__ == "__main__":
    test_instance = TestZonotopeInterval()
    
    # Run all tests
    test_instance.test_basic_interval_conversion()
    
    print("All zonotope interval tests passed!") 