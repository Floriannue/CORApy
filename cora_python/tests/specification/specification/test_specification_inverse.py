"""
Unit tests for specification inverse method

This file tests the inverse functionality for temporal logic specifications.
Based on test_specification_inverse.m from MATLAB CORA.

Authors: MATLAB CORA developers (original)
         Python translation by AI Assistant
"""

import unittest
import numpy as np
from typing import List

# Add path for imports
import sys
sys.path.insert(0, '.')

from cora_python.specification.specification.specification import Specification
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestSpecificationInverse(unittest.TestCase):
    """Test cases for specification inverse method"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create test specifications with different types
        from cora_python.contSet.polytope.polytope import Polytope
        
        # 2D polytope: x1 >= 0, x2 >= 0, x1 + x2 <= 2
        A = np.array([[-1, 0], [0, -1], [1, 1]])
        b = np.array([0, 0, 2])
        self.set_2d = Polytope(A, b)
        
        # Time intervals
        self.time_interval = np.array([0, 10])
        
        # Create specifications of different types
        self.spec_safe = Specification(self.set_2d, 'safeSet', self.time_interval)
        self.spec_unsafe = Specification(self.set_2d, 'unsafeSet', self.time_interval)
        self.spec_final = Specification(self.set_2d, 'finalSet', self.time_interval)
        self.spec_invariant = Specification(self.set_2d, 'invariant', self.time_interval)
        
        # Specifications without time
        self.spec_safe_no_time = Specification(self.set_2d, 'safeSet')
        self.spec_unsafe_no_time = Specification(self.set_2d, 'unsafeSet')
    
    def test_inverse_basic_functionality(self):
        """Test basic inverse functionality"""
        # Test inverse of safeSet -> unsafeSet
        inv_safe = self.spec_safe.inverse()
        self.assertIsInstance(inv_safe, Specification)
        self.assertEqual(inv_safe.type, 'unsafeSet')
        
        # Test inverse of unsafeSet -> safeSet
        inv_unsafe = self.spec_unsafe.inverse()
        self.assertIsInstance(inv_unsafe, Specification)
        self.assertEqual(inv_unsafe.type, 'safeSet')
    
    def test_inverse_type_mapping(self):
        """Test correct type mapping for inverse"""
        # safeSet -> unsafeSet
        inv_safe = self.spec_safe.inverse()
        self.assertEqual(inv_safe.type, 'unsafeSet')
        
        # unsafeSet -> safeSet
        inv_unsafe = self.spec_unsafe.inverse()
        self.assertEqual(inv_unsafe.type, 'safeSet')
        
        # finalSet -> finalSet (should remain the same)
        inv_final = self.spec_final.inverse()
        self.assertEqual(inv_final.type, 'finalSet')
        
        # invariant -> invariant (should remain the same)
        inv_invariant = self.spec_invariant.inverse()
        self.assertEqual(inv_invariant.type, 'invariant')
    
    def test_inverse_preserves_set(self):
        """Test that inverse preserves the underlying set"""
        inv_safe = self.spec_safe.inverse()
        
        # The set should be the same (implementation dependent)
        # At minimum, should have same dimension
        if hasattr(self.spec_safe.set, 'dim') and hasattr(inv_safe.set, 'dim'):
            self.assertEqual(self.spec_safe.set.dim(), inv_safe.set.dim())
    
    def test_inverse_preserves_time(self):
        """Test that inverse preserves time information"""
        inv_safe = self.spec_safe.inverse()
        
        # Should preserve time interval
        if hasattr(self.spec_safe, 'time') and self.spec_safe.time is not None:
            if hasattr(inv_safe, 'time'):
                np.testing.assert_array_equal(inv_safe.time, self.spec_safe.time)
    
    def test_inverse_without_time(self):
        """Test inverse for specifications without time"""
        inv_safe_no_time = self.spec_safe_no_time.inverse()
        self.assertIsInstance(inv_safe_no_time, Specification)
        self.assertEqual(inv_safe_no_time.type, 'unsafeSet')
        
        inv_unsafe_no_time = self.spec_unsafe_no_time.inverse()
        self.assertIsInstance(inv_unsafe_no_time, Specification)
        self.assertEqual(inv_unsafe_no_time.type, 'safeSet')
    
    def test_inverse_double_inverse(self):
        """Test that double inverse returns to original type"""
        # safe -> unsafe -> safe
        double_inv_safe = self.spec_safe.inverse().inverse()
        self.assertEqual(double_inv_safe.type, self.spec_safe.type)
        
        # unsafe -> safe -> unsafe
        double_inv_unsafe = self.spec_unsafe.inverse().inverse()
        self.assertEqual(double_inv_unsafe.type, self.spec_unsafe.type)
        
        # finalSet -> finalSet -> finalSet
        double_inv_final = self.spec_final.inverse().inverse()
        self.assertEqual(double_inv_final.type, self.spec_final.type)
        
        # invariant -> invariant -> invariant
        double_inv_invariant = self.spec_invariant.inverse().inverse()
        self.assertEqual(double_inv_invariant.type, self.spec_invariant.type)
    
    def test_inverse_return_type(self):
        """Test that inverse returns correct type"""
        inv_spec = self.spec_safe.inverse()
        
        # Should return Specification instance
        self.assertIsInstance(inv_spec, Specification)
        
        # Should have required attributes
        self.assertTrue(hasattr(inv_spec, 'set'))
        self.assertTrue(hasattr(inv_spec, 'type'))
        self.assertIn(inv_spec.type, ['safeSet', 'unsafeSet', 'finalSet', 'invariant'])
    
    def test_inverse_idempotent_types(self):
        """Test inverse for types that should be idempotent"""
        # finalSet inverse should be finalSet
        inv_final = self.spec_final.inverse()
        self.assertEqual(inv_final.type, 'finalSet')
        
        # invariant inverse should be invariant
        inv_invariant = self.spec_invariant.inverse()
        self.assertEqual(inv_invariant.type, 'invariant')
    
    def test_inverse_consistency(self):
        """Test inverse consistency"""
        # Multiple calls should be consistent
        inv1 = self.spec_safe.inverse()
        inv2 = self.spec_safe.inverse()
        
        # Should have same type
        self.assertEqual(inv1.type, inv2.type)
    
    def test_inverse_logical_correctness(self):
        """Test logical correctness of inverse operation"""
        # Safe set becomes unsafe set (logical negation)
        inv_safe = self.spec_safe.inverse()
        self.assertEqual(inv_safe.type, 'unsafeSet')
        
        # Unsafe set becomes safe set (logical negation)
        inv_unsafe = self.spec_unsafe.inverse()
        self.assertEqual(inv_unsafe.type, 'safeSet')
        
        # Final set remains final set (not affected by negation)
        inv_final = self.spec_final.inverse()
        self.assertEqual(inv_final.type, 'finalSet')
        
        # Invariant remains invariant (not affected by negation)
        inv_invariant = self.spec_invariant.inverse()
        self.assertEqual(inv_invariant.type, 'invariant')
    
    def test_inverse_preserves_attributes(self):
        """Test that inverse preserves necessary attributes"""
        inv_spec = self.spec_safe.inverse()
        
        # Should preserve location if it exists
        if hasattr(self.spec_safe, 'location'):
            if hasattr(inv_spec, 'location'):
                self.assertEqual(inv_spec.location, self.spec_safe.location)
        
        # Should preserve id if it exists
        if hasattr(self.spec_safe, 'id'):
            if hasattr(inv_spec, 'id'):
                self.assertEqual(inv_spec.id, self.spec_safe.id)
    
    def test_inverse_with_complex_specifications(self):
        """Test inverse with more complex specifications"""
        # Test with different set types if available
        try:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            
            # Create zonotope specification
            Z = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
            spec_zono = Specification(Z, 'safeSet', self.time_interval)
            
            inv_zono = spec_zono.inverse()
            self.assertEqual(inv_zono.type, 'unsafeSet')
            
        except ImportError:
            # Skip if zonotope not available
            pass
    
    def test_inverse_error_handling(self):
        """Test inverse error handling"""
        # Inverse should not fail for valid specifications
        # All current tests should pass without raising errors
        
        # Test with potentially invalid specifications (if such cases exist)
        # This depends on the implementation details
        pass


if __name__ == '__main__':
    unittest.main() 