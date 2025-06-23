"""
Unit tests for specification project method

This file tests the project functionality for temporal logic specifications.
Based on test_specification_project.m from MATLAB CORA.

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


class TestSpecificationProject(unittest.TestCase):
    """Test cases for specification project method"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create test specifications with polytopes of different dimensions
        from cora_python.contSet.polytope.polytope import Polytope
        
        # 3D polytope: x1 >= 0, x2 >= 0, x3 >= 0, x1 + x2 + x3 <= 3
        A_3d = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]])
        b_3d = np.array([0, 0, 0, 3])
        self.set_3d = Polytope(A_3d, b_3d)
        
        # 4D polytope for more complex projection tests
        A_4d = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1], [1, 1, 1, 1]])
        b_4d = np.array([0, 0, 0, 0, 4])
        self.set_4d = Polytope(A_4d, b_4d)
        
        # Time intervals
        self.time_interval = np.array([0, 10])
        
        # Create specifications
        self.spec_3d_safe = Specification(self.set_3d, 'safeSet', self.time_interval)
        self.spec_3d_unsafe = Specification(self.set_3d, 'unsafeSet', self.time_interval)
        self.spec_4d_safe = Specification(self.set_4d, 'safeSet', self.time_interval)
        
        # Specification without time
        self.spec_3d_no_time = Specification(self.set_3d, 'safeSet')
    
    def test_project_basic_functionality(self):
        """Test basic projection functionality"""
        # Project 3D specification to 2D
        projected_12 = self.spec_3d_safe.project([1, 2])
        self.assertIsInstance(projected_12, Specification)
        self.assertEqual(projected_12.type, 'safeSet')
        
        # Project to different dimensions
        projected_13 = self.spec_3d_safe.project([1, 3])
        projected_23 = self.spec_3d_safe.project([2, 3])
        
        self.assertIsInstance(projected_13, Specification)
        self.assertIsInstance(projected_23, Specification)
    
    def test_project_single_dimension(self):
        """Test projection to single dimension"""
        # Project to 1D
        projected_1 = self.spec_3d_safe.project([1])
        projected_2 = self.spec_3d_safe.project([2])
        projected_3 = self.spec_3d_safe.project([3])
        
        self.assertIsInstance(projected_1, Specification)
        self.assertIsInstance(projected_2, Specification)
        self.assertIsInstance(projected_3, Specification)
    
    def test_project_preserves_properties(self):
        """Test that projection preserves specification properties"""
        projected = self.spec_3d_safe.project([1, 2])
        
        # Should preserve type
        self.assertEqual(projected.type, self.spec_3d_safe.type)
        
        # Should preserve time if it exists
        if hasattr(self.spec_3d_safe, 'time') and self.spec_3d_safe.time is not None:
            if hasattr(projected, 'time'):
                np.testing.assert_array_equal(projected.time, self.spec_3d_safe.time)
    
    def test_project_different_types(self):
        """Test projection for different specification types"""
        # Test safeSet
        projected_safe = self.spec_3d_safe.project([1, 2])
        self.assertEqual(projected_safe.type, 'safeSet')
        
        # Test unsafeSet
        projected_unsafe = self.spec_3d_unsafe.project([1, 2])
        self.assertEqual(projected_unsafe.type, 'unsafeSet')
    
    def test_project_without_time(self):
        """Test projection for specification without time"""
        projected = self.spec_3d_no_time.project([1, 2])
        self.assertIsInstance(projected, Specification)
        self.assertEqual(projected.type, 'safeSet')
    
    def test_project_high_dimension(self):
        """Test projection from higher dimensions"""
        # Project 4D to 2D
        projected_4d_2d = self.spec_4d_safe.project([1, 3])
        self.assertIsInstance(projected_4d_2d, Specification)
        
        # Project 4D to 3D
        projected_4d_3d = self.spec_4d_safe.project([1, 2, 4])
        self.assertIsInstance(projected_4d_3d, Specification)
    
    def test_project_identity(self):
        """Test projection to same dimensions (identity)"""
        # Project 3D to all 3 dimensions - should be equivalent to original
        projected_identity = self.spec_3d_safe.project([1, 2, 3])
        self.assertIsInstance(projected_identity, Specification)
        self.assertEqual(projected_identity.type, self.spec_3d_safe.type)
    
    def test_project_reordered_dimensions(self):
        """Test projection with reordered dimensions"""
        # Project with different ordering
        projected_21 = self.spec_3d_safe.project([2, 1])  # Reversed order
        projected_321 = self.spec_3d_safe.project([3, 2, 1])  # Full reverse
        
        self.assertIsInstance(projected_21, Specification)
        self.assertIsInstance(projected_321, Specification)
    
    def test_project_error_cases(self):
        """Test project error handling"""
        # Test invalid dimension indices
        with self.assertRaises((ValueError, IndexError, CORAerror)):
            self.spec_3d_safe.project([0])  # Invalid dimension (0-based indexing issue)
        
        with self.assertRaises((ValueError, IndexError, CORAerror)):
            self.spec_3d_safe.project([4])  # Dimension out of range for 3D
        
        with self.assertRaises((ValueError, IndexError, CORAerror)):
            self.spec_3d_safe.project([-1])  # Negative dimension
        
        # Test empty dimension list
        with self.assertRaises((ValueError, CORAerror)):
            self.spec_3d_safe.project([])
        
        # Test invalid input types
        with self.assertRaises((TypeError, ValueError)):
            self.spec_3d_safe.project("invalid")
        
        with self.assertRaises((TypeError, ValueError)):
            self.spec_3d_safe.project(None)
    
    def test_project_duplicate_dimensions(self):
        """Test projection with duplicate dimensions"""
        # Should handle or reject duplicate dimensions appropriately
        try:
            projected = self.spec_3d_safe.project([1, 1, 2])
            # If it succeeds, should be valid specification
            self.assertIsInstance(projected, Specification)
        except (ValueError, CORAerror):
            # It's also valid to reject duplicate dimensions
            pass
    
    def test_project_return_type(self):
        """Test that project returns correct type"""
        projected = self.spec_3d_safe.project([1, 2])
        
        # Should return Specification instance
        self.assertIsInstance(projected, Specification)
        
        # Should have a set attribute
        self.assertTrue(hasattr(projected, 'set'))
        
        # Should have a type attribute
        self.assertTrue(hasattr(projected, 'type'))
        self.assertIn(projected.type, ['safeSet', 'unsafeSet', 'finalSet', 'invariant'])
    
    def test_project_consistency(self):
        """Test projection consistency"""
        # Multiple projections should be consistent
        proj1 = self.spec_3d_safe.project([1, 2])
        proj2 = self.spec_3d_safe.project([1, 2])
        
        # Should have same type
        self.assertEqual(proj1.type, proj2.type)
        
        # Sets should be equivalent (if implemented)
        # Note: This depends on the set comparison implementation
    
    def test_project_chaining(self):
        """Test chaining of projections"""
        # Project 3D -> 2D -> 1D
        proj_2d = self.spec_3d_safe.project([1, 2])
        proj_1d = proj_2d.project([1])
        
        self.assertIsInstance(proj_1d, Specification)
        
        # Should be equivalent to direct projection
        direct_1d = self.spec_3d_safe.project([1])
        self.assertEqual(proj_1d.type, direct_1d.type)


if __name__ == '__main__':
    unittest.main() 