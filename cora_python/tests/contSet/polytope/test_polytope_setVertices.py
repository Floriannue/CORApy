"""
Test file for polytope.setVertices method

This file contains unit tests for the setVertices method of the Polytope class.
"""

import pytest
import numpy as np
from cora_python.contSet.polytope import Polytope


class TestPolytopeSetVertices:
    """Test class for polytope.setVertices method"""
    
    def test_setVertices_basic(self):
        """Test basic vertex setting functionality"""
        # Create a polytope with constraints
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        # Set new vertices
        V = np.array([[-1, 0], [1, -2], [1, 2]]).T
        P.setVertices(V)
        
        # Check that vertices were set
        assert hasattr(P, 'V')
        assert P.isVRep
        assert np.array_equal(P.V, V)
    
    def test_setVertices_overwrite(self):
        """Test overwriting existing vertices"""
        # Create a polytope with vertices
        V1 = np.array([[0, 0], [1, 0], [0, 1]]).T
        P = Polytope(V1)
        
        # Set new vertices
        V2 = np.array([[-1, -1], [2, 0], [0, 2]]).T
        P.setVertices(V2)
        
        # Check that vertices were updated
        assert np.array_equal(P.V, V2)
        assert P.isVRep
    
    def test_setVertices_empty(self):
        """Test setting empty vertices"""
        P = Polytope()
        V = np.zeros((2, 0))
        P.setVertices(V)
        
        assert hasattr(P, 'V')
        assert P.isVRep
        assert P.V.size == 0
    
    def test_setVertices_1d(self):
        """Test setting vertices for 1D polytope"""
        P = Polytope()
        V = np.array([[0, 1, 2]]).T
        P.setVertices(V)
        
        assert P.V.shape == (1, 3)
        assert P.isVRep
    
    def test_setVertices_3d(self):
        """Test setting vertices for 3D polytope"""
        P = Polytope()
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        P.setVertices(V)
        
        assert P.V.shape == (3, 4)
        assert P.isVRep
    
    def test_setVertices_preserves_properties(self):
        """Test that setting vertices preserves other properties"""
        # Create polytope with constraints
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        # Set some cached properties
        P._bounded_val = True
        P._fullDim_val = True
        
        # Set vertices
        V = np.array([[-1, 0], [1, -2], [1, 2]]).T
        P.setVertices(V)
        
        # Check that vertices were set
        assert P.isVRep
        assert np.array_equal(P.V, V)
        
        # Note: In MATLAB, setting vertices doesn't clear H-rep,
        # but it does set isVRep flag
