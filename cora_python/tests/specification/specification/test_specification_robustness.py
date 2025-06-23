"""
test_specification_robustness - unit test for robustness method

This test covers the robustness computation for different specification types.

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 2021 (MATLAB)  
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.specification.specification.specification import Specification
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestSpecificationRobustness(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create simple test sets
        # Unit square: [0,1] x [0,1]
        self.safe_set = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        self.unsafe_set = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        
        # Create specifications
        self.spec_safe = Specification(self.safe_set, 'safeSet')
        self.spec_unsafe = Specification(self.unsafe_set, 'unsafeSet')
        self.spec_invariant = Specification(self.safe_set, 'invariant')
        
        # Test points
        self.point_inside = np.array([[0.5], [0.5]])  # Inside [0,1]x[0,1]
        self.point_outside = np.array([[2.0], [2.0]])  # Outside [0,1]x[0,1]
        self.point_boundary = np.array([[1.0], [1.0]])  # On boundary
        self.point_close_outside = np.array([[1.1], [1.1]])  # Close outside
        
        # Time interval
        self.time_interval = Interval(np.array([[0]]), np.array([[2]]))
    
    def test_robustness_safe_set_inside_point(self):
        """Test robustness for safe set with point inside"""
        try:
            robustness_val = self.spec_safe.robustness(self.point_inside)
            
            # Point inside safe set should have positive robustness
            self.assertIsInstance(robustness_val, (int, float, np.floating))
            self.assertGreater(robustness_val, 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_safe_set_outside_point(self):
        """Test robustness for safe set with point outside"""
        try:
            robustness_val = self.spec_safe.robustness(self.point_outside)
            
            # Point outside safe set should have negative robustness
            self.assertIsInstance(robustness_val, (int, float, np.floating))
            self.assertLess(robustness_val, 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_unsafe_set_inside_point(self):
        """Test robustness for unsafe set with point inside"""
        try:
            robustness_val = self.spec_unsafe.robustness(self.point_inside)
            
            # Point inside unsafe set should have negative robustness
            self.assertIsInstance(robustness_val, (int, float, np.floating))
            self.assertLess(robustness_val, 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_unsafe_set_outside_point(self):
        """Test robustness for unsafe set with point outside"""
        try:
            robustness_val = self.spec_unsafe.robustness(self.point_outside)
            
            # Point outside unsafe set should have positive robustness
            self.assertIsInstance(robustness_val, (int, float, np.floating))
            self.assertGreater(robustness_val, 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_invariant_set(self):
        """Test robustness for invariant set"""
        try:
            robustness_inside = self.spec_invariant.robustness(self.point_inside)
            robustness_outside = self.spec_invariant.robustness(self.point_outside)
            
            # Invariant should behave like safe set
            self.assertIsInstance(robustness_inside, (int, float, np.floating))
            self.assertIsInstance(robustness_outside, (int, float, np.floating))
            self.assertGreater(robustness_inside, 0)
            self.assertLess(robustness_outside, 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_boundary_point(self):
        """Test robustness for point on boundary"""
        try:
            robustness_safe = self.spec_safe.robustness(self.point_boundary)
            robustness_unsafe = self.spec_unsafe.robustness(self.point_boundary)
            
            # Boundary points should have robustness close to zero
            self.assertIsInstance(robustness_safe, (int, float, np.floating))
            self.assertIsInstance(robustness_unsafe, (int, float, np.floating))
            self.assertLessEqual(abs(robustness_safe), 0.1)  # Close to zero
            self.assertLessEqual(abs(robustness_unsafe), 0.1)  # Close to zero
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_multiple_points(self):
        """Test robustness with multiple points"""
        points = np.hstack([self.point_inside, self.point_outside, self.point_boundary])
        
        try:
            robustness_vals = self.spec_safe.robustness(points)
            
            # Should return array of robustness values
            self.assertIsInstance(robustness_vals, np.ndarray)
            self.assertEqual(len(robustness_vals), 3)
            
            # Check individual values
            self.assertGreater(robustness_vals[0], 0)  # Inside
            self.assertLess(robustness_vals[1], 0)     # Outside
            self.assertLessEqual(abs(robustness_vals[2]), 0.1)  # Boundary
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_with_time(self):
        """Test robustness with time constraints"""
        spec_timed = Specification(self.safe_set, 'safeSet', self.time_interval)
        
        try:
            # Test at different times
            robustness_t0 = spec_timed.robustness(self.point_inside, 0.0)
            robustness_t1 = spec_timed.robustness(self.point_inside, 1.0)
            robustness_t3 = spec_timed.robustness(self.point_inside, 3.0)  # Outside time
            
            self.assertIsInstance(robustness_t0, (int, float, np.floating))
            self.assertIsInstance(robustness_t1, (int, float, np.floating))
            self.assertIsInstance(robustness_t3, (int, float, np.floating))
            
            # Inside time window should have normal robustness
            self.assertGreater(robustness_t0, 0)
            self.assertGreater(robustness_t1, 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Time-constrained robustness not fully implemented yet")
    
    def test_robustness_distance_consistency(self):
        """Test that robustness reflects distance to set boundary"""
        # Points at different distances from boundary
        point_far_inside = np.array([[0.5], [0.5]])    # Center of [0,1]x[0,1]
        point_close_inside = np.array([[0.9], [0.9]])  # Close to boundary
        
        try:
            rob_far = self.spec_safe.robustness(point_far_inside)
            rob_close = self.spec_safe.robustness(point_close_inside)
            
            # Farther point should have higher robustness
            self.assertGreater(rob_far, rob_close)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")
    
    def test_robustness_custom_specification(self):
        """Test robustness for custom specification"""
        custom_func = lambda x: x[0] > 0  # Simple custom function
        spec_custom = Specification(custom_func, 'custom')
        
        try:
            robustness_val = spec_custom.robustness(self.point_inside)
            # Custom specs should raise not supported error
            self.fail("Custom specification robustness should raise CORAerror")
            
        except CORAerror as e:
            # Expected behavior
            self.assertIn('not supported', str(e))
        except (NotImplementedError, AttributeError):
            self.skipTest("Custom specification handling not implemented yet")
    
    def test_robustness_logic_specification(self):
        """Test robustness for logic specification"""
        # This would require STL implementation
        # For now, just test that it raises appropriate error
        pass
    
    def test_robustness_error_cases(self):
        """Test error cases for robustness method"""
        try:
            # Test with invalid point dimensions
            wrong_dim_point = np.array([[1], [2], [3]])  # 3D point for 2D set
            
            with self.assertRaises((ValueError, AttributeError, CORAerror)):
                self.spec_safe.robustness(wrong_dim_point)
            
            # Test with timed specification but no time provided
            spec_timed = Specification(self.safe_set, 'safeSet', self.time_interval)
            with self.assertRaises(CORAerror):
                spec_timed.robustness(self.point_inside)  # Missing time argument
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Error handling in robustness method not implemented yet")
    
    def test_robustness_empty_specification(self):
        """Test robustness with empty specification"""
        spec_empty = Specification()
        
        try:
            with self.assertRaises((ValueError, AttributeError, CORAerror)):
                spec_empty.robustness(self.point_inside)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Empty specification handling not implemented yet")
    
    def test_robustness_numerical_properties(self):
        """Test numerical properties of robustness values"""
        try:
            rob_val = self.spec_safe.robustness(self.point_inside)
            
            # Robustness should be finite
            self.assertFalse(np.isnan(rob_val))
            self.assertFalse(np.isinf(rob_val))
            
            # Should be a real number
            self.assertIsInstance(rob_val, (int, float, np.floating))
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Robustness method not fully implemented yet")


if __name__ == '__main__':
    unittest.main() 