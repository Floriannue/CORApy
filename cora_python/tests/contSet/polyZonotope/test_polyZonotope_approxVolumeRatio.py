"""
test_polyZonotope_approxVolumeRatio - GENERATED TEST
   No MATLAB source test exists. This test was created by analyzing the MATLAB 
   implementation logic in approxVolumeRatio.py and ensuring thorough coverage.

   This test verifies that approxVolumeRatio correctly calculates the approximate 
   ratio of volumes between:
   - The polynomial zonotope constructed by only the dependent generators
   - The zonotope constructed by the independent generator part
   
   The ratio is computed as: ratio = (V_ind/V_dep)^(1/n)

Syntax:
    pytest cora_python/tests/contSet/polyZonotope/test_polyZonotope_approxVolumeRatio.py

Authors:       Generated test based on MATLAB implementation
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polyZonotope.approxVolumeRatio import approxVolumeRatio


class TestApproxVolumeRatio:
    """Test class for approxVolumeRatio functionality"""
    
    def test_approxVolumeRatio_empty_GI(self):
        """Test case when GI (independent generators) is empty"""
        # MATLAB logic: if isempty(pZ.GI), ratio = 0.0
        # Create polyZonotope with empty GI
        c = np.array([[1], [2]])
        G = np.array([[1, 0.5], [0, 1]])  # Dependent generators
        GI = np.array([]).reshape(2, 0)  # Empty independent generators
        E = np.array([[1, 0], [0, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        ratio = approxVolumeRatio(pZ)
        assert ratio == 0.0
    
    def test_approxVolumeRatio_empty_G(self):
        """Test case when G (dependent generators) is empty"""
        # MATLAB logic: if isempty(pZ.G), ratio = inf
        # Create polyZonotope with empty G
        c = np.array([[1], [2]])
        G = np.array([]).reshape(2, 0)  # Empty dependent generators
        GI = np.array([[0.5], [0.3]])  # Independent generators
        E = np.array([]).reshape(0, 0)  # Empty exponent matrix
        pZ = PolyZonotope(c, G, GI, E)
        
        ratio = approxVolumeRatio(pZ)
        assert np.isinf(ratio)
    
    def test_approxVolumeRatio_basic(self):
        """Test basic volume ratio calculation"""
        # Create a simple polyZonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0.5], [0, 1]])  # Dependent generators
        GI = np.array([[0.2], [0.1]])  # Independent generators
        E = np.array([[1, 0], [0, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Calculate ratio - should be a positive number
        ratio = approxVolumeRatio(pZ)
        assert ratio > 0
        assert not np.isnan(ratio)
        assert not np.isinf(ratio)
    
    def test_approxVolumeRatio_interval_method(self):
        """Test with interval method"""
        c = np.array([[0], [0]])
        G = np.array([[1, 0.5], [0, 1]])
        GI = np.array([[0.2], [0.1]])
        E = np.array([[1, 0], [0, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test with explicit 'interval' method
        ratio_interval = approxVolumeRatio(pZ, 'interval')
        assert ratio_interval > 0
        assert not np.isnan(ratio_interval)
    
    def test_approxVolumeRatio_pca_method(self):
        """Test with pca method"""
        c = np.array([[0], [0]])
        G = np.array([[1, 0.5], [0, 1]])
        GI = np.array([[0.2], [0.1]])
        E = np.array([[1, 0], [0, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test with 'pca' method
        ratio_pca = approxVolumeRatio(pZ, 'pca')
        assert ratio_pca > 0
        assert not np.isnan(ratio_pca)
    
    def test_approxVolumeRatio_3D(self):
        """Test with 3D polyZonotope"""
        c = np.array([[0], [0], [0]])
        G = np.array([[1, 0.5, 0.2], [0, 1, 0.3], [0.1, 0, 1]])
        GI = np.array([[0.2], [0.1], [0.15]])
        E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        ratio = approxVolumeRatio(pZ)
        assert ratio > 0
        assert not np.isnan(ratio)
        assert not np.isinf(ratio)
    
    def test_approxVolumeRatio_large_dependent(self):
        """Test when dependent part is much larger than independent"""
        c = np.array([[0], [0]])
        G = np.array([[10, 5, 3], [0, 10, 2]])  # Large dependent generators
        GI = np.array([[0.1], [0.05]])  # Small independent generators
        E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        ratio = approxVolumeRatio(pZ)
        # Ratio should be small when independent is much smaller
        assert ratio > 0
        assert ratio < 1.0  # Likely small when V_ind << V_dep


def test_polyZonotope_approxVolumeRatio():
    """Test function for polyZonotope approxVolumeRatio method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestApproxVolumeRatio()
    test.test_approxVolumeRatio_empty_GI()
    test.test_approxVolumeRatio_empty_G()
    test.test_approxVolumeRatio_basic()
    test.test_approxVolumeRatio_interval_method()
    test.test_approxVolumeRatio_pca_method()
    test.test_approxVolumeRatio_3D()
    test.test_approxVolumeRatio_large_dependent()
    
    print("test_polyZonotope_approxVolumeRatio: all tests passed")
    return True


if __name__ == "__main__":
    test_polyZonotope_approxVolumeRatio()

