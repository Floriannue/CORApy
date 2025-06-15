"""
test_taylorLinSys - unit test function for taylorLinSys class

Tests the taylorLinSys class for Taylor series computations in linear systems.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.classes.taylorLinSys import TaylorLinSys


class TestTaylorLinSys:
    def test_taylorLinSys_constructor(self):
        """Test constructor"""
        A = np.array([[0, 1], [-1, -2]])
        taylor = TaylorLinSys(A)
        
        assert np.allclose(taylor.A, A)
        assert taylor.timeStep is None
        assert taylor.eAt is None
        assert taylor.eAdt is None
    
    def test_taylorLinSys_constructor_invertible(self):
        """Test constructor with invertible matrix"""
        A = np.array([[1, 2], [3, 4]])
        taylor = TaylorLinSys(A)
        
        assert taylor.Ainv is not None
        # Check that A * Ainv = I
        assert np.allclose(A @ taylor.Ainv, np.eye(2), atol=1e-10)
    
    def test_taylorLinSys_constructor_singular(self):
        """Test constructor with singular matrix"""
        A = np.array([[1, 2], [2, 4]])  # Singular matrix
        taylor = TaylorLinSys(A)
        
        assert taylor.Ainv is None
    
    def test_computeField_eAt(self):
        """Test computeField for matrix exponential"""
        A = np.array([[0, 1], [-1, -2]])
        taylor = TaylorLinSys(A)
        
        timeStep = 0.1
        eAt = taylor.computeField('eAt', timeStep=timeStep)
        
        # Check that it's stored
        assert taylor.timeStep == timeStep
        assert taylor.eAt is not None
        assert taylor.eAdt is not None
        assert np.allclose(taylor.eAt, eAt)
        assert np.allclose(taylor.eAdt, eAt)  # Alias
        
        # Check against scipy.linalg.expm
        from scipy.linalg import expm
        expected = expm(A * timeStep)
        assert np.allclose(eAt, expected, atol=1e-10)
    
    def test_computeField_eAt_cached(self):
        """Test that eAt is cached for same time step"""
        A = np.array([[0, 1], [-1, -2]])
        taylor = TaylorLinSys(A)
        
        timeStep = 0.1
        eAt1 = taylor.computeField('eAt', timeStep=timeStep)
        eAt2 = taylor.computeField('eAt', timeStep=timeStep)
        
        # Should be the same object (cached)
        assert eAt1 is eAt2
    
    def test_computeField_eAt_different_timestep(self):
        """Test eAt computation with different time steps"""
        A = np.array([[0, 1], [-1, -2]])
        taylor = TaylorLinSys(A)
        
        eAt1 = taylor.computeField('eAt', timeStep=0.1)
        eAt2 = taylor.computeField('eAt', timeStep=0.2)
        
        # Should be different
        assert not np.allclose(eAt1, eAt2)
        assert taylor.timeStep == 0.2  # Last one used
    
    def test_computeField_Ainv(self):
        """Test computeField for matrix inverse"""
        A = np.array([[1, 2], [3, 4]])
        taylor = TaylorLinSys(A)
        
        Ainv = taylor.computeField('Ainv')
        
        assert np.allclose(A @ Ainv, np.eye(2), atol=1e-10)
        assert taylor.Ainv is Ainv  # Should be stored
    
    def test_computeField_Apower(self):
        """Test computeField for matrix powers"""
        A = np.array([[1, 1], [0, 1]])
        taylor = TaylorLinSys(A)
        
        # Test A^0 (should be identity)
        A0 = taylor.computeField('Apower', ithpower=0)
        assert np.allclose(A0, np.eye(2))
        
        # Test A^1 (should be A)
        A1 = taylor.computeField('Apower', ithpower=1)
        assert np.allclose(A1, A)
        
        # Test A^2
        A2 = taylor.computeField('Apower', ithpower=2)
        expected_A2 = A @ A
        assert np.allclose(A2, expected_A2)
        
        # Test A^3
        A3 = taylor.computeField('Apower', ithpower=3)
        expected_A3 = A @ A @ A
        assert np.allclose(A3, expected_A3)
    
    def test_computeField_unknown(self):
        """Test computeField with unknown field name"""
        A = np.array([[1, 0], [0, 1]])
        taylor = TaylorLinSys(A)
        
        with pytest.raises(Exception):  # CORAError
            taylor.computeField('unknown_field')
    
    def test_getTaylor(self):
        """Test getTaylor method"""
        A = np.array([[0, 1], [-1, -2]])
        taylor = TaylorLinSys(A)
        
        # Should delegate to computeField
        eAt = taylor.getTaylor('eAt', timeStep=0.1)
        
        from scipy.linalg import expm
        expected = expm(A * 0.1)
        assert np.allclose(eAt, expected, atol=1e-10)
    
    def test_matrix_exponential_properties(self):
        """Test mathematical properties of matrix exponential"""
        A = np.array([[0, 1], [-1, -2]])
        taylor = TaylorLinSys(A)
        
        dt1 = 0.1
        dt2 = 0.2
        
        eAt1 = taylor.computeField('eAt', timeStep=dt1)
        eAt2 = taylor.computeField('eAt', timeStep=dt2)
        eAt_sum = taylor.computeField('eAt', timeStep=dt1 + dt2)
        
        # Property: e^(A*(t1+t2)) = e^(A*t1) * e^(A*t2)
        product = eAt1 @ eAt2
        assert np.allclose(eAt_sum, product, atol=1e-10)
    
    def test_matrix_exponential_identity(self):
        """Test matrix exponential of zero matrix"""
        A = np.zeros((2, 2))
        taylor = TaylorLinSys(A)
        
        eAt = taylor.computeField('eAt', timeStep=0.5)
        
        # e^(0*t) should be identity matrix
        assert np.allclose(eAt, np.eye(2))
    
    def test_matrix_exponential_diagonal(self):
        """Test matrix exponential of diagonal matrix"""
        A = np.array([[-1, 0], [0, -2]])
        taylor = TaylorLinSys(A)
        
        timeStep = 0.5
        eAt = taylor.computeField('eAt', timeStep=timeStep)
        
        # For diagonal matrix, e^(A*t) should have e^(a_ii*t) on diagonal
        expected = np.array([[np.exp(-0.5), 0], [0, np.exp(-1.0)]])
        assert np.allclose(eAt, expected, atol=1e-10)
    
    def test_large_matrix(self):
        """Test with larger matrix"""
        n = 5
        A = np.random.randn(n, n)
        taylor = TaylorLinSys(A)
        
        eAt = taylor.computeField('eAt', timeStep=0.1)
        
        # Check dimensions
        assert eAt.shape == (n, n)
        
        # Check that it's computed correctly
        from scipy.linalg import expm
        expected = expm(A * 0.1)
        assert np.allclose(eAt, expected, atol=1e-10)
    
    def test_multiple_power_computations(self):
        """Test computing multiple powers efficiently"""
        A = np.array([[1, 1], [0, 1]])
        taylor = TaylorLinSys(A)
        
        # Compute powers in order
        powers = []
        for i in range(5):
            Ai = taylor.computeField('Apower', ithpower=i)
            powers.append(Ai)
        
        # Check that A^0 = I
        assert np.allclose(powers[0], np.eye(2))
        
        # Check that A^i = A^(i-1) * A
        for i in range(1, 5):
            expected = powers[i-1] @ A
            assert np.allclose(powers[i], expected)
    
    def test_error_handling_no_timestep(self):
        """Test error when timeStep not specified"""
        A = np.array([[1, 0], [0, 1]])
        taylor = TaylorLinSys(A)
        
        with pytest.raises(Exception):  # CORAError
            taylor._computeEAt()  # No timeStep provided


if __name__ == "__main__":
    pytest.main([__file__]) 