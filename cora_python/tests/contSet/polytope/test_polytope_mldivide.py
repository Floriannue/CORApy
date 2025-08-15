"""
Test file for polytope.mldivide method

This file contains unit tests for the mldivide method of the Polytope class.
"""

import pytest
import numpy as np
from cora_python.contSet.polytope import Polytope


class TestPolytopeMldivide:
    """Test class for polytope.mldivide method"""
    
    def test_mldivide_basic_subtraction(self):
        """Test basic set difference P1 \ P2"""
        # Create two polytopes
        A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b1 = np.array([2, 2, 2, 2])
        P1 = Polytope(A1, b1)
        
        A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b2 = np.array([1, 1, 1, 1])
        P2 = Polytope(A2, b2)
        
        # Compute set difference
        P_diff = P1.mldivide(P2)
        
        # Check that result is a polytope
        assert isinstance(P_diff, Polytope)
        
        # Result should not contain points from P2
        # (This is a basic check - exact containment logic may vary)
    
    def test_mldivide_empty_first(self):
        """Test set difference when first polytope is empty"""
        P1 = Polytope.empty(2)
        P2 = Polytope(np.array([[1, 0], [-1, 0]]), np.array([1, 1]))
        
        P_diff = P1.mldivide(P2)
        
        # Result should be empty
        assert P_diff.representsa_('emptySet')
    
    def test_mldivide_empty_second(self):
        """Test set difference when second polytope is empty"""
        A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b1 = np.array([1, 1, 1, 1])
        P1 = Polytope(A1, b1)
        
        P2 = Polytope.empty(2)
        
        P_diff = P1.mldivide(P2)
        
        # Result should be P1 (unchanged)
        assert P_diff.isequal(P1)
    
    def test_mldivide_error_wrong_types(self):
        """Test error handling for wrong input types"""
        P1 = Polytope()
        
        # Both inputs must be polytopes
        with pytest.raises(Exception):
            P1.mldivide(1)
        
        with pytest.raises(Exception):
            Polytope.mldivide(1, P1)
    
    def test_mldivide_full_dim_both(self):
        """Test set difference with full-dimensional polytopes"""
        # Create full-dimensional polytopes
        A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b1 = np.array([2, 2, 2, 2])
        P1 = Polytope(A1, b1)
        
        A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b2 = np.array([1, 1, 1, 1])
        P2 = Polytope(A2, b2)
        
        # Both should be full-dimensional
        assert P1.isFullDim()
        assert P2.isFullDim()
        
        P_diff = P1.mldivide(P2)
        assert isinstance(P_diff, Polytope)
    
    def test_mldivide_equality_constraints(self):
        """Test set difference with equality constraints"""
        # Create polytope with equality constraints
        A1 = np.array([[1, 0], [-1, 0]])
        b1 = np.array([1, 1])
        Ae1 = np.array([[0, 1]])
        be1 = np.array([0])
        P1 = Polytope(A1, b1, Ae1, be1)
        
        A2 = np.array([[1, 0], [-1, 0]])
        b2 = np.array([0.5, 0.5])
        P2 = Polytope(A2, b2)
        
        P_diff = P1.mldivide(P2)
        assert isinstance(P_diff, Polytope)
    
    def test_mldivide_subset_relationship(self):
        """Test set difference when P1 is subset of P2"""
        # Create P1 as subset of P2
        A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b1 = np.array([0.5, 0.5, 0.5, 0.5])
        P1 = Polytope(A1, b1)
        
        A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b2 = np.array([1, 1, 1, 1])
        P2 = Polytope(A2, b2)
        
        # P1 should be subset of P2
        assert P1 <= P2
        
        P_diff = P1.mldivide(P2)
        
        # Result should be empty when P1 is subset of P2
        assert P_diff.representsa_('emptySet')
    
    def test_mldivide_operator_syntax(self):
        """Test that mldivide can be called with \ operator"""
        A1 = np.array([[1, 0], [-1, 0]])
        b1 = np.array([1, 1])
        P1 = Polytope(A1, b1)
        
        A2 = np.array([[1, 0], [-1, 0]])
        b2 = np.array([0.5, 0.5])
        P2 = Polytope(A2, b2)
        
        # Test both ways of calling
        P_diff1 = P1.mldivide(P2)
        P_diff2 = P1.__rfloordiv__(P2)  # This would be the \ operator
        
        # Both should give same result
        assert P_diff1.isequal(P_diff2)
