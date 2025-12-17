"""
testLong_zonotope_contains_SadraddiniTedrake - unit test function of contains_
using the method of SadraddiniTedrake.

We compare the results with the exact method to ensure correctness.

Authors: MATLAB: Matthias Althoff
         Python: AI Assistant
Written: 21-July-2022 (MATLAB)
Last update: ---
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestLongZonotopeContainsSadraddiniTedrake:
    """Tests for zonotope containment using SadraddiniTedrake method"""
    
    def test_random_cases_non_zero_center(self):
        """Test random cases in 3D (non-zero center)"""
        np.random.seed(42)  # For reproducibility
        
        for i_set in range(10):
            # Create zonotope which is supposed to be enclosed (s: small)
            Z_s = Zonotope.generateRandom('Dimension', 3, 'NrGenerators', 5)
            
            # Create zonotope which is supposed to enclose the smaller one (l: large)
            # Enlarge by 1.2 (using scalar multiplication)
            Z_l = 1.2 * Zonotope.generateRandom('Dimension', 3, 'NrGenerators', 5)
            
            # Compute result with approx:st
            res_original, cert_original, scaling_original = Z_l.contains_(Z_s, 'approx:st')
            
            # Compute alternative result with exact method
            # Currently, we compare the results of 'approx:st' with the exact method
            res_alternative, cert_alternative, scaling_alternative = Z_l.contains_(Z_s, 'exact')
            
            # If exact says it's contained, then approx:st should also say it's contained
            # (approx:st may be conservative, but shouldn't be wrong in the other direction)
            if res_alternative:
                assert res_original, f"Test {i_set+1}: If exact says contained, approx:st should also say contained"
    
    def test_random_cases_zero_center(self):
        """Test random cases in 3D (zero center)"""
        np.random.seed(43)  # For reproducibility
        
        for i_set in range(10):
            # Create zonotope which is supposed to be enclosed (s: small)
            Z_s = Zonotope.generateRandom('Center', np.zeros((3, 1)), 'Dimension', 3, 'NrGenerators', 5)
            
            # Create zonotope which is supposed to enclose the smaller one (l: large)
            # Enlarge by 1.2 (using scalar multiplication)
            Z_l = 1.2 * Zonotope.generateRandom('Center', np.zeros((3, 1)), 'Dimension', 3, 'NrGenerators', 5)
            
            # Compute result with approx:st
            res_original, cert_original, scaling_original = Z_l.contains_(Z_s, 'approx:st')
            
            # Compute alternative result with exact method
            # Currently, we compare the results of 'approx:st' with the exact method
            res_alternative, cert_alternative, scaling_alternative = Z_l.contains_(Z_s, 'exact')
            
            # If exact says it's contained, then approx:st should also say it's contained
            # (approx:st may be conservative, but shouldn't be wrong in the other direction)
            if res_alternative:
                assert res_original, f"Test {i_set+1}: If exact says contained, approx:st should also say contained"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

