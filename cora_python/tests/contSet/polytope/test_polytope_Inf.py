"""
Test file for polytope Inf method

This file contains unit tests for the polytope Inf method.
Tests the creation of n-dimensional polytopes equivalent to R^n
"""

import pytest
import numpy as np
from cora_python.contSet.polytope import Polytope


class TestPolytopeInf:
    """Test class for polytope Inf method"""
    
    def test_Inf_0d(self):
        """Test Inf for 0-dimensional polytope"""
        P = Polytope.Inf(0)
        
        # Check dimensions
        assert P.dim() == 0
        assert P.A.shape == (0, 0)
        assert P.b.shape == (0, 1)
        
        # Check properties - MATLAB behavior: Inf(0) is considered empty
        assert P.isemptyobject()  # MATLAB returns true for Inf(0)
        assert not P.isBounded()
        assert P.isFullDim()
        assert P.isHRep
    
    def test_Inf_1d(self):
        """Test Inf for 1-dimensional polytope"""
        P = Polytope.Inf(1)
        
        # Check dimensions
        assert P.dim() == 1
        assert P.A.shape == (0, 1)
        assert P.b.shape == (0, 1)
        
        # Check properties
        assert not P.isemptyobject()  # Should have vertices, so not empty
        assert not P.isBounded()
        assert P.isFullDim()
        assert P.isHRep
        
        # Check V-representation for low dimensions
        assert P.isVRep
        assert P.V.shape == (1, 2)  # 2^1 = 2 combinations
        # Should have [-inf] and [inf] as vertices
        assert np.allclose(P.V[:, 0], [-np.inf])
        assert np.allclose(P.V[:, 1], [np.inf])
    
    def test_Inf_2d(self):
        """Test Inf for 2-dimensional polytope"""
        P = Polytope.Inf(2)
        
        # Check dimensions
        assert P.dim() == 2
        assert P.A.shape == (0, 2)
        assert P.b.shape == (0, 1)
        
        # Check properties
        assert not P.isemptyobject()  # Should have vertices, so not empty
        assert not P.isBounded()
        assert P.isFullDim()
        assert P.isHRep
        
        # Check V-representation for low dimensions
        assert P.isVRep
        assert P.V.shape == (2, 4)  # 2^2 = 4 combinations
        # Should have all combinations of -inf/inf
        expected_vertices = [
            [-np.inf, -np.inf],
            [-np.inf, np.inf],
            [np.inf, -np.inf],
            [np.inf, np.inf]
        ]
        for i, expected in enumerate(expected_vertices):
            assert np.allclose(P.V[:, i], expected)
    
    def test_Inf_3d(self):
        """Test Inf for 3-dimensional polytope"""
        P = Polytope.Inf(3)
        
        # Check dimensions
        assert P.dim() == 3
        assert P.A.shape == (0, 3)
        assert P.b.shape == (0, 1)
        
        # Check properties
        assert not P.isemptyobject()  # Should have vertices, so not empty
        assert not P.isBounded()
        assert P.isFullDim()
        assert P.isHRep
        
        # Check V-representation for low dimensions
        assert P.isVRep
        assert P.V.shape == (3, 8)  # 2^3 = 8 combinations
    
    def test_Inf_8d(self):
        """Test Inf for 8-dimensional polytope (boundary case)"""
        P = Polytope.Inf(8)
        
        # Check dimensions
        assert P.dim() == 8
        assert P.A.shape == (0, 8)
        assert P.b.shape == (0, 1)
        
        # Check properties
        assert not P.isemptyobject()  # Should have vertices, so not empty
        assert not P.isBounded()
        assert P.isFullDim()
        assert P.isHRep
        
        # Check V-representation for low dimensions
        assert P.isVRep
        assert P.V.shape == (8, 256)  # 2^8 = 256 combinations
    
    def test_Inf_9d(self):
        """Test Inf for 9-dimensional polytope (high dimension case)"""
        P = Polytope.Inf(9)
        
        # Check dimensions
        assert P.dim() == 9
        assert P.A.shape == (0, 9)
        assert P.b.shape == (0, 1)
        
        # Check properties - for n > 8, MATLAB doesn't set vertices, so it's considered empty
        assert P.isemptyobject()  # MATLAB behavior: no vertices for high dimensions
        assert not P.isBounded()
        assert P.isFullDim()
        assert P.isHRep
        
        # Check V-representation for high dimensions (should not be set)
        assert not P.isVRep
        # MATLAB throws error when accessing P.V for high dimensions - test this behavior
        with pytest.raises(Exception):  # Should be CORA:notSupported
            _ = P.V
        
    def test_Inf_high_dimension(self):
        """Test Inf for high-dimensional polytope"""
        P = Polytope.Inf(100)
        
        # Check dimensions
        assert P.dim() == 100
        assert P.A.shape == (0, 100)
        assert P.b.shape == (0, 1)
        
        # Check properties - for n > 8, MATLAB doesn't set vertices, so it's considered empty
        assert P.isemptyobject()  # MATLAB behavior: no vertices for high dimensions
        assert not P.isBounded()
        assert P.isFullDim()
        assert P.isHRep
        
        # Check V-representation for high dimensions (should not be set)
        assert not P.isVRep
        # MATLAB throws error when accessing P.V for high dimensions - test this behavior
        with pytest.raises(Exception):  # Should be CORA:notSupported
            _ = P.V
    
    def test_Inf_contains_points(self):
        """Test that Inf polytope contains all points"""
        P = Polytope.Inf(2)
        
        # Test various points
        test_points = [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([-1, -1]),
            np.array([1000, -1000]),
            np.array([1e6, 1e-6])
        ]
        
        for point in test_points:
            assert P.contains_(point)
    
    def test_Inf_represents_fullspace(self):
        """Test that Inf polytope represents fullspace"""
        P = Polytope.Inf(3)
        
        # Should represent fullspace
        assert P.representsa_('fullspace')
        assert not P.representsa_('emptySet')
        # Note: 'bounded' is not a valid set type for representsa_
        # Instead, check the bounded property directly
        assert not P.isBounded()
    
    def test_Inf_invalid_input(self):
        """Test Inf with invalid inputs"""
        # Test negative dimension
        with pytest.raises(Exception):
            Polytope.Inf(-1)
        
        # Test non-integer dimension
        with pytest.raises(Exception):
            Polytope.Inf(1.5)
