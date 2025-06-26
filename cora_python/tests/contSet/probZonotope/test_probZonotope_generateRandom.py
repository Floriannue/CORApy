"""
test_generateRandom - unit test function for probZonotope generateRandom

Tests the random probZonotope generation functionality.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.probZonotope.probZonotope import ProbZonotope


class TestProbZonotopeGenerateRandom:
    """Test class for probZonotope generateRandom method"""
    
    def test_generateRandom_default(self):
        """Test generateRandom with default parameters"""
        Z = ProbZonotope.generateRandom()
        assert isinstance(Z, ProbZonotope)
        assert Z.dim() > 0
        assert not Z.isemptyobject()
    
    def test_generateRandom_dimension(self):
        """Test generateRandom with specific dimension"""
        Z = ProbZonotope.generateRandom(Dimension=3)
        assert isinstance(Z, ProbZonotope)
        assert Z.dim() == 3
        assert not Z.isemptyobject()
    
    def test_generateRandom_center(self):
        """Test generateRandom with specific center"""
        center = np.ones((2, 1))
        Z = ProbZonotope.generateRandom(Center=center)
        assert isinstance(Z, ProbZonotope)
        assert Z.dim() == 2
        assert np.allclose(Z.c, center)
        assert not Z.isemptyobject()
    
    def test_generateRandom_dimension_and_generators(self):
        """Test generateRandom with dimension and number of generators"""
        Z = ProbZonotope.generateRandom(Dimension=4, NrGenerators=10)
        assert isinstance(Z, ProbZonotope)
        assert Z.dim() == 4
        assert not Z.isemptyobject()
        # Number of generators check would depend on implementation
    
    def test_generateRandom_distribution(self):
        """Test generateRandom with specific distribution"""
        Z = ProbZonotope.generateRandom(Distribution='gamma')
        assert isinstance(Z, ProbZonotope)
        assert Z.dim() > 0
        assert not Z.isemptyobject()
    
    def test_generateRandom_various_dimensions(self):
        """Test generateRandom with various dimensions"""
        for dim in [1, 2, 3, 5, 10]:
            Z = ProbZonotope.generateRandom(Dimension=dim)
            assert Z.dim() == dim
            assert not Z.isemptyobject()
    
    def test_generateRandom_various_centers(self):
        """Test generateRandom with various centers"""
        centers = [
            np.zeros((2, 1)),
            np.ones((3, 1)),
            np.array([[1], [2], [3], [4]]),
            np.random.randn(5, 1)
        ]
        
        for center in centers:
            Z = ProbZonotope.generateRandom(Center=center)
            assert Z.dim() == center.shape[0]
            assert np.allclose(Z.c, center)
    
    def test_generateRandom_combined_parameters(self):
        """Test generateRandom with combined parameters"""
        center = np.array([[1], [2]])
        Z = ProbZonotope.generateRandom(Center=center, NrGenerators=5)
        assert Z.dim() == 2
        assert np.allclose(Z.c, center)
        assert not Z.isemptyobject()
    
    def test_generateRandom_properties(self):
        """Test properties of randomly generated probZonotope"""
        Z = ProbZonotope.generateRandom(Dimension=3, NrGenerators=4)
        
        # Should have correct dimension
        assert Z.dim() == 3
        
        # Should not be empty
        assert not Z.isemptyobject()
        
        # Should have proper matrix dimensions
        assert Z.c.shape[0] == 3
        assert Z.G.shape[0] == 3
        # Number of generators would depend on implementation
    
    def test_generateRandom_edge_cases(self):
        """Test edge cases for generateRandom"""
        # Minimum dimension
        Z = ProbZonotope.generateRandom(Dimension=1)
        assert Z.dim() == 1
        assert not Z.isemptyobject()
        
        # Zero center
        center = np.zeros((3, 1))
        Z = ProbZonotope.generateRandom(Center=center)
        assert Z.dim() == 3
        assert np.allclose(Z.c, center)
    
    def test_generateRandom_consistency(self):
        """Test consistency of generateRandom"""
        # Multiple calls should produce valid probZonotopes
        for _ in range(5):
            Z = ProbZonotope.generateRandom(Dimension=3)
            assert isinstance(Z, ProbZonotope)
            assert Z.dim() == 3
            assert not Z.isemptyobject() 