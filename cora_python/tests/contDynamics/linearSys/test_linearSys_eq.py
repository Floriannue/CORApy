"""
test_linearSys_eq - unit test function for equality comparison

Tests the eq method for linearSys objects to check if two systems are equal.

Syntax:
    pytest cora_python/tests/contDynamics/linearSys/test_linearSys_eq.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contDynamics.linearSys.linearSys import LinearSys


class TestLinearSysEq:
    def test_eq_identical_systems(self):
        """Test equality of identical systems"""
        A = np.array([[0, 1], [-1, 0]])
        B = np.array([[0], [1]])
        
        sys1 = LinearSys(A, B)
        sys2 = LinearSys(A, B)
        
        assert sys1.eq(sys2)
        assert sys1 == sys2

    def test_eq_different_A_matrices(self):
        """Test inequality of systems with different A matrices"""
        A1 = np.array([[0, 1], [-1, 0]])
        A2 = np.array([[1, 1], [-1, 0]])
        B = np.array([[0], [1]])
        
        sys1 = LinearSys(A1, B)
        sys2 = LinearSys(A2, B)
        
        assert not sys1.eq(sys2)
        assert not (sys1 == sys2)

    def test_eq_different_B_matrices(self):
        """Test inequality of systems with different B matrices"""
        A = np.array([[0, 1], [-1, 0]])
        B1 = np.array([[0], [1]])
        B2 = np.array([[1], [1]])
        
        sys1 = LinearSys(A, B1)
        sys2 = LinearSys(A, B2)
        
        assert not sys1.eq(sys2)
        assert not (sys1 == sys2)

    def test_eq_with_offset(self):
        """Test equality with constant offset"""
        A = np.array([[0, 1], [-1, 0]])
        B = np.array([[0], [1]])
        c = np.array([[1], [0]])
        
        sys1 = LinearSys(A, B, c)
        sys2 = LinearSys(A, B, c)
        sys3 = LinearSys(A, B, np.array([[2], [0]]))
        
        assert sys1.eq(sys2)
        assert not sys1.eq(sys3)

    def test_eq_with_output_matrices(self):
        """Test equality with output matrices"""
        A = np.array([[0, 1], [-1, 0]])
        B = np.array([[0], [1]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
        
        sys1 = LinearSys(A, B, None, C, D)
        sys2 = LinearSys(A, B, None, C, D)
        sys3 = LinearSys(A, B, None, np.array([[0, 1]]), D)
        
        assert sys1.eq(sys2)
        assert not sys1.eq(sys3)

    def test_eq_full_system(self):
        """Test equality with full system specification"""
        A = np.array([[0, 1, -1], [1, 0, 0], [0, 1, 0]])
        B = np.array([[0], [-1], [0]])
        c = np.array([[0], [0], [0]])
        C = np.array([[0, 0, 0.05], [0.05, 0.05, 0]])
        D = np.array([[0], [0]])
        k = np.array([[0], [0]])
        
        sys1 = LinearSys(A, B, c, C, D, k)
        sys2 = LinearSys(A, B, c, C, D, k)
        
        assert sys1.eq(sys2)

    def test_eq_with_tolerance(self):
        """Test equality with tolerance"""
        A = np.array([[0, 1], [-1, 0]])
        B = np.array([[0], [1]])
        
        # Small perturbation
        eps = 1e-15
        A_pert = A + eps * np.ones_like(A)
        
        sys1 = LinearSys(A, B)
        sys2 = LinearSys(A_pert, B)
        
        # Should be equal with appropriate tolerance
        assert sys1.eq(sys2, 1e-14)
        # Should not be equal with strict tolerance
        assert not sys1.eq(sys2, 1e-16)

    def test_eq_empty_systems(self):
        """Test equality of empty systems"""
        sys1 = LinearSys()
        sys2 = LinearSys()
        
        assert sys1.eq(sys2)

    def test_eq_different_dimensions(self):
        """Test inequality of systems with different dimensions"""
        A1 = np.array([[0, 1], [-1, 0]])
        B1 = np.array([[0], [1]])
        
        A2 = np.array([[0]])
        B2 = np.array([[1]])
        
        sys1 = LinearSys(A1, B1)
        sys2 = LinearSys(A2, B2)
        
        assert not sys1.eq(sys2)

    def test_eq_scalar_B_expansion(self):
        """Test equality when B is scalar and gets expanded"""
        A = np.array([[1, 2], [3, 4]])
        B_scalar = 1
        B_matrix = np.eye(2)
        
        sys1 = LinearSys(A, B_scalar)
        sys2 = LinearSys(A, B_matrix)
        
        # These should be equal since scalar B gets expanded to identity
        assert sys1.eq(sys2)

    def test_eq_with_disturbance_matrices(self):
        """Test equality with disturbance matrices E and F"""
        A = np.array([[0, 1], [-1, 0]])
        B = np.array([[0], [1]])
        E = np.array([[1], [0]])
        F = np.array([[0]])
        
        sys1 = LinearSys(A, B, None, None, None, None, E, F)
        sys2 = LinearSys(A, B, None, None, None, None, E, F)
        sys3 = LinearSys(A, B, None, None, None, None, np.array([[0], [1]]), F)
        
        assert sys1.eq(sys2)
        assert not sys1.eq(sys3)

    def test_ne_operator(self):
        """Test not-equal operator"""
        A1 = np.array([[0, 1], [-1, 0]])
        A2 = np.array([[1, 1], [-1, 0]])
        B = np.array([[0], [1]])
        
        sys1 = LinearSys(A1, B)
        sys2 = LinearSys(A2, B)
        
        assert sys1.ne(sys2)
        assert sys1 != sys2


if __name__ == "__main__":
    pytest.main([__file__]) 