"""
Test file for priv_minkdiff private helper function

This file contains unit tests for the priv_minkdiff function.
"""

import pytest
import numpy as np
from cora_python.contSet.polytope.private.priv_minkdiff import priv_minkdiff


class TestPrivMinkdiff:
    """Test class for priv_minkdiff function"""
    
    def test_priv_minkdiff_basic(self):
        """Test basic Minkowski difference computation"""
        # Create constraint matrices
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([2, 2, 2, 2])
        Ae = np.array([])
        be = np.array([])
        
        # Create a simple set (interval) for testing
        from cora_python.contSet.interval.interval import Interval
        S = Interval(np.array([[-0.5], [-0.5]]), np.array([[0.5], [0.5]]))
        
        # Compute Minkowski difference
        A_out, b_out, Ae_out, be_out, empty = priv_minkdiff(A, b.copy(), Ae, be, S)
        
        # Check that result is not empty
        assert not empty
        assert A_out.shape == A.shape
        assert b_out.shape == b.shape
        
        # Check that constraints were shifted
        assert not np.array_equal(b_out, b)
    
    def test_priv_minkdiff_empty_result(self):
        """Test Minkowski difference that results in empty set"""
        # Create constraint matrices for a degenerate polytope (line segment)
        # This creates a line segment at x = 3 (degenerate in y-direction)
        A = np.array([[0, 1], [0, -1]])  # 0*x + 1*y <= 2 and 0*x + 1*y <= 1
        b = np.array([2, 1])
        Ae = np.array([[1, 0]])  # 1*x + 0*y = 3 (equality constraint)
        be = np.array([3])
        
        # Create a bounded set for testing (box)
        from cora_python.contSet.interval.interval import Interval
        S = Interval(np.array([[-0.5], [-0.5]]), np.array([[0.5], [0.5]]))
        
        # Compute Minkowski difference
        A_out, b_out, Ae_out, be_out, empty = priv_minkdiff(A, b.copy(), Ae, be.copy(), S)
        
        # Check that result is empty due to dimension mismatch
        # The line segment doesn't have enough "thickness" in y-direction
        assert empty
        assert A_out.size == 0
        assert b_out.size == 0
        assert Ae_out.size == 0
        assert be_out.size == 0
    
    def test_priv_minkdiff_equality_constraints(self):
        """Test Minkowski difference with equality constraints"""
        # Create constraint matrices with equality constraints
        # This creates a line segment from (-1,0) to (1,0) with equality constraint y = 0
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        Ae = np.array([[0, 1]])
        be = np.array([0])
        
        # Create a vector (point) for testing, matching MATLAB test case
        # P2 = [-1; 1] - this is a single point, not an interval with extent
        from cora_python.contSet.interval.interval import Interval
        S = Interval(np.array([[-1], [1]]), np.array([[-1], [1]]))  # Single point at (-1, 1)
        
        # Compute Minkowski difference
        A_out, b_out, Ae_out, be_out, empty = priv_minkdiff(A, b.copy(), Ae, be.copy(), S)
        
        # Check that result is not empty
        assert not empty
        assert A_out.shape == A.shape
        assert Ae_out.shape == Ae.shape
        
        # Check that equality constraints were adjusted
        # The support function in y-direction will return a single value, and be should be adjusted
        assert not np.array_equal(be_out, be)
    
    def test_priv_minkdiff_equality_empty_result(self):
        """Test Minkowski difference with equality constraints that results in empty set"""
        # Create constraint matrices with equality constraints
        A = np.array([])
        b = np.array([])
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([0, 0])
        
        # Create a set that will cause equality constraints to be violated
        from cora_python.contSet.zonotope.zonotope import Zonotope
        S = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        
        # Compute Minkowski difference
        A_out, b_out, Ae_out, be_out, empty = priv_minkdiff(A, b, Ae, be.copy(), S)
        
        # Check that result is empty due to equality constraint violation
        assert empty
        assert A_out.size == 0
        assert b_out.size == 0
        assert Ae_out.size == 0
        assert be_out.size == 0
    
    def test_priv_minkdiff_no_constraints(self):
        """Test Minkowski difference with no constraints"""
        # Create empty constraint matrices
        A = np.array([])
        b = np.array([])
        Ae = np.array([])
        be = np.array([])
        
        # Create a set for testing
        from cora_python.contSet.interval.interval import Interval
        S = Interval(np.array([[-0.1], [-0.1]]), np.array([[0.1], [0.1]]))
        
        # Compute Minkowski difference
        A_out, b_out, Ae_out, be_out, empty = priv_minkdiff(A, b, Ae, be, S)
        
        # Check that result is not empty and unchanged
        assert not empty
        assert A_out.size == 0
        assert b_out.size == 0
        assert Ae_out.size == 0
        assert be_out.size == 0
