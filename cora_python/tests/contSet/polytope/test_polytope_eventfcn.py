"""
Test file for polytope eventFcn method

This file contains unit tests for the polytope eventFcn method.
Tests the event function for ODE event handling
"""

import pytest
import numpy as np
from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestPolytopeEventFcn:
    """Test class for polytope eventFcn method"""
    
    def test_eventFcn_1d_no_equalities(self):
        """Test eventFcn for 1D polytope without equality constraints"""
        # Create 1D polytope: -1 <= x <= 1
        A = np.array([[1], [-1]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        
        # Test with point inside
        x = np.array([0])
        val, isterminal, direction = P.eventFcn(x)
        
        # Check output shapes
        assert val.shape == (2,)  # 2 constraints
        assert isterminal.shape == (2,)
        assert direction.shape == (2,)
        
        # Check values
        # val = A*x - b = [1*0 - 1, -1*0 - 1] = [-1, -1]
        assert np.allclose(val, [-1, -1])
        assert np.all(isterminal == 1)
        assert np.all(direction == 1)
        
        # Test with point on boundary
        x = np.array([1])
        val, isterminal, direction = P.eventFcn(x)
        # val = [1*1 - 1, -1*1 - 1] = [0, -2]
        assert np.allclose(val, [0, -2])
        
        # Test with point outside
        x = np.array([2])
        val, isterminal, direction = P.eventFcn(x)
        # val = [1*2 - 1, -1*2 - 1] = [1, -3]
        assert np.allclose(val, [1, -3])
    
    def test_eventFcn_2d_no_equalities(self):
        """Test eventFcn for 2D polytope without equality constraints"""
        # Create 2D polytope: unit square
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        # Test with point inside
        x = np.array([0, 0])
        val, isterminal, direction = P.eventFcn(x)
        
        # Check output shapes
        assert val.shape == (4,)  # 4 constraints
        assert isterminal.shape == (4,)
        assert direction.shape == (4,)
        
        # Check values
        # val = A*x - b = [0-1, 0-1, 0-1, 0-1] = [-1, -1, -1, -1]
        assert np.allclose(val, [-1, -1, -1, -1])
        assert np.all(isterminal == 1)
        assert np.all(direction == 1)
        
        # Test with point on boundary
        x = np.array([1, 0])
        val, isterminal, direction = P.eventFcn(x)
        # val = [1-1, -1-1, 0-1, 0-1] = [0, -2, -1, -1]
        assert np.allclose(val, [0, -2, -1, -1])
    
    def test_eventFcn_with_equality_constraints(self):
        """Test eventFcn for polytope with equality constraints (conHyperplane)"""
        # Create polytope representing a line: x = 0 (degenerate in y-direction)
        A = np.array([])
        b = np.array([])
        Ae = np.array([[1, 0]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        
        # Test with point on the line
        x = np.array([0, 0])
        val, isterminal, direction = P.eventFcn(x)
        
        # Check output shapes
        assert val.shape == (1,)  # 1 equality constraint
        assert isterminal.shape == (1,)
        assert direction.shape == (1,)
        
        # Check values
        # val = Ae*x - be = [1*0 + 0*0 - 0] = [0]
        assert np.allclose(val, [0])
        assert np.all(isterminal == 1)
        assert np.all(direction == 1)
        
        # Test with point off the line
        x = np.array([1, 0])
        val, isterminal, direction = P.eventFcn(x)
        # val = [1*1 + 0*0 - 0] = [1]
        assert np.allclose(val, [1])
    
    def test_eventFcn_custom_direction(self):
        """Test eventFcn with custom direction value"""
        A = np.array([[1], [-1]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        
        x = np.array([0])
        val, isterminal, direction = P.eventFcn(x, direction_val=-1)
        
        # Check that direction is set to custom value
        assert np.all(direction == -1)
    
    def test_eventFcn_empty_polytope(self):
        """Test eventFcn for empty polytope"""
        # Create empty polytope
        A = np.array([[1, 1], [-1, -1]])
        b = np.array([-1, -1])  # Infeasible constraints
        P = Polytope(A, b)
        
        x = np.array([0, 0])
        val, isterminal, direction = P.eventFcn(x)
        
        # Should still work even if polytope is empty
        assert val.shape == (2,)
        assert isterminal.shape == (2,)
        assert direction.shape == (2,)
    
    def test_eventFcn_high_dimension(self):
        """Test eventFcn for high-dimensional polytope"""
        # Create 5D polytope
        n = 5
        A = np.eye(n)  # Identity matrix
        b = np.ones(n)
        P = Polytope(A, b)
        
        x = np.zeros(n)
        val, isterminal, direction = P.eventFcn(x)
        
        # Check output shapes
        assert val.shape == (n,)
        assert isterminal.shape == (n,)
        assert direction.shape == (n,)
        
        # Check values
        # val = A*x - b = [0-1, 0-1, 0-1, 0-1, 0-1] = [-1, -1, -1, -1, -1]
        assert np.allclose(val, [-1] * n)
    
    def test_eventFcn_invalid_input(self):
        """Test eventFcn with invalid inputs"""
        A = np.array([[1], [-1]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        
        # Test with wrong input type
        with pytest.raises(Exception):
            P.eventFcn("invalid")
        
        # Test with wrong dimension
        with pytest.raises(Exception):
            P.eventFcn(np.array([1, 2]))  # 2D point for 1D polytope
    
    def test_eventFcn_unsupported_polytope_type(self):
        """Test eventFcn with unsupported polytope type"""
        # Create a polytope that's not a conHyperplane but has equality constraints
        # This should raise an error
        A = np.array([[1, 0]])
        b = np.array([1])
        Ae = np.array([[0, 1]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        
        # This polytope has both inequality and equality constraints
        # but is not a conHyperplane, so it should raise an error
        x = np.array([0, 0])
        with pytest.raises(CORAerror, match="Given polytope type not supported"):
            P.eventFcn(x)
