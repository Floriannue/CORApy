"""
Test file for polytope.enclose method

This file contains unit tests for the enclose method of the Polytope class.
"""

import pytest
import numpy as np
from cora_python.contSet.polytope import Polytope


class TestPolytopeEnclose:
    """Test class for polytope.enclose method"""
    
    def test_enclose_two_polytopes(self):
        """Test enclosing two polytopes"""
        # Create two polytopes
        A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b1 = np.array([1, 1, 1, 1])
        P1 = Polytope(A1, b1)
        
        A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b2 = np.array([2, 2, 2, 2])
        P2 = Polytope(A2, b2)
        
        # Enclose them
        P_enclose = P1.enclose(P2)
        
        # Check that result is a polytope
        assert isinstance(P_enclose, Polytope)
        
        # Check that both original polytopes are contained
        assert P_enclose.contains_(P1)
        assert P_enclose.contains_(P2)
    
    def test_enclose_matrix_polytope(self):
        """Test enclosing a polytope and its transformation"""
        # Create base polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P1 = Polytope(A, b)
        
        # Transformation matrix and offset
        M = np.array([[2, 0], [0, 2]])  # Scale by 2
        P_plus = Polytope(np.array([[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]]), 
                          np.array([0.5, 0.5, 0.5, 0.5]))
        
        # Enclose
        P_enclose = P1.enclose(M, P_plus)
        
        # Check that result is a polytope
        assert isinstance(P_enclose, Polytope)
        
        # Check that original is contained
        assert P_enclose.contains_(P1)
    
    def test_enclose_error_wrong_input(self):
        """Test error handling for wrong input types"""
        P1 = Polytope()
        
        # Wrong number of arguments
        with pytest.raises(Exception):
            P1.enclose()
        
        with pytest.raises(Exception):
            P1.enclose(1, 2, 3)
    
    def test_enclose_error_non_polytope(self):
        """Test error handling for non-polytope first argument"""
        # First argument must be a polytope
        with pytest.raises(Exception):
            Polytope.enclose(1, Polytope())
    
    def test_enclose_single_polytope(self):
        """Test enclosing with single polytope argument"""
        P1 = Polytope()
        P2 = Polytope(np.array([[1, 0], [-1, 0]]), np.array([1, 1]))
        
        # This should work and return P2 (or equivalent)
        P_enclose = P1.enclose(P2)
        assert isinstance(P_enclose, Polytope)
    
    def test_enclose_identity_transformation(self):
        """Test enclosing with identity transformation"""
        P1 = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 
                      np.array([1, 1, 1, 1]))
        
        M = np.eye(2)  # Identity matrix
        P_plus = Polytope()  # Empty polytope
        
        P_enclose = P1.enclose(M, P_plus)
        assert isinstance(P_enclose, Polytope)
        
        # Result should contain P1
        assert P_enclose.contains_(P1)
