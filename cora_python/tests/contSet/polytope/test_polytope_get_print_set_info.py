"""
Test file for polytope get_print_set_info method

This file contains unit tests for the polytope get_print_set_info method.
Tests the information returned for printing polytopes
"""

import pytest
import numpy as np
from cora_python.contSet.polytope import Polytope


class TestPolytopeGetPrintSetInfo:
    """Test class for polytope get_print_set_info method"""
    
    def test_get_print_set_info_h_rep(self):
        """Test get_print_set_info for H-representation polytope"""
        # Create polytope in H-representation
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Check property order for H-representation
        expected_order = ['A', 'b', 'Ae', 'be']
        assert property_order == expected_order
    
    def test_get_print_set_info_v_rep(self):
        """Test get_print_set_info for V-representation polytope"""
        # Create polytope in V-representation
        V = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]).T
        P = Polytope(V)
        
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Check property order for V-representation
        expected_order = ['V']
        assert property_order == expected_order
    
    def test_get_print_set_info_mixed_representation(self):
        """Test get_print_set_info for polytope with both H and V representation"""
        # Create polytope with both representations
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        
        # Force V-representation to be computed
        _ = P.V
        
        # Now it should have both representations
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Should return H-representation order since that's the primary representation
        expected_order = ['A', 'b', 'Ae', 'be']
        assert property_order == expected_order
    
    def test_get_print_set_info_empty_polytope(self):
        """Test get_print_set_info for empty polytope"""
        # Create empty polytope
        A = np.zeros((0, 2))
        b = np.zeros((0, 0))
        P = Polytope(A, b)
        
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Check property order for H-representation (even if empty)
        expected_order = ['A', 'b', 'Ae', 'be']
        assert property_order == expected_order
    
    def test_get_print_set_info_1d_polytope(self):
        """Test get_print_set_info for 1D polytope"""
        # Create 1D polytope
        A = np.array([[1], [-1]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Check property order for H-representation
        expected_order = ['A', 'b', 'Ae', 'be']
        assert property_order == expected_order
    
    def test_get_print_set_info_high_dimension(self):
        """Test get_print_set_info for high-dimensional polytope"""
        # Create high-dimensional polytope
        n = 10
        A = np.eye(n)
        b = np.ones(n)
        P = Polytope(A, b)
        
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Check property order for H-representation
        expected_order = ['A', 'b', 'Ae', 'be']
        assert property_order == expected_order
    
    def test_get_print_set_info_equality_constraints(self):
        """Test get_print_set_info for polytope with equality constraints"""
        # Create polytope with equality constraints
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        Ae = np.array([[0, 1]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Check property order for H-representation with equality constraints
        expected_order = ['A', 'b', 'Ae', 'be']
        assert property_order == expected_order
    
    def test_get_print_set_info_vertex_only(self):
        """Test get_print_set_info for polytope created only from vertices"""
        # Create polytope from vertices only
        V = np.array([[0, 0], [1, 0], [0, 1]]).T
        P = Polytope(V)
        
        abbrev, property_order = P.get_print_set_info()
        
        # Check abbreviation
        assert abbrev == 'P'
        
        # Check property order for V-representation
        expected_order = ['V']
        assert property_order == expected_order
    
    def test_get_print_set_info_consistency(self):
        """Test that get_print_set_info returns consistent results"""
        # Create polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        # Call multiple times
        abbrev1, property_order1 = P.get_print_set_info()
        abbrev2, property_order2 = P.get_print_set_info()
        
        # Results should be consistent
        assert abbrev1 == abbrev2
        assert property_order1 == property_order2
