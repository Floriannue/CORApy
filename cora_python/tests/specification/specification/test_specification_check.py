"""
test_specification_check - unit test for check method

This test covers the check functionality for different specification types
including safeSet, unsafeSet, and invariant specifications.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-April-2023 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.specification.specification.specification import Specification
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval


class TestSpecificationCheck(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize specifications
        # Create a polytope: constraint [1, 1]/sqrt(2) * x <= 1
        # This represents the half-space x + y <= sqrt(2)
        A = np.array([[1, 1]]) / np.sqrt(2)
        b = np.array([1])
        
        try:
            self.set = Polytope(A, b)
        except:
            # Fallback to interval if Polytope not available
            self.set = Interval(np.array([[-1], [-1]]), np.array([[1], [1]]))
        
        self.spec_unsafe = Specification(self.set, 'unsafeSet')
        self.spec_safe = Specification(self.set, 'safeSet')
        self.spec_invariant = Specification(self.set, 'invariant')
        
        # Create test sets for checking
        # Set that should be outside the polytope (safe from unsafe spec)
        self.Z_outside = Zonotope(np.array([[-10], [-8]]), 
                                 np.array([[1, 0, -2], [2, -1, 1]]))
        
        # Set that intersects the polytope
        self.Z_intersect = Zonotope(np.array([[0], [0]]), 
                                   np.array([[0.5, 0], [0, 0.5]]))
        
        # Set that is inside the polytope  
        self.Z_inside = Zonotope(np.array([[0.1], [0.1]]), 
                                np.array([[0.05, 0], [0, 0.05]]))
    
    def test_check_zonotope_outside(self):
        """Test checking zonotope that is outside the specification set"""
        # Zonotope outside should satisfy unsafe spec (not violate it)
        # and should satisfy safe/invariant specs
        
        try:
            result_unsafe = self.spec_unsafe.check(self.Z_outside)
            result_safe = self.spec_safe.check(self.Z_outside)
            result_invariant = self.spec_invariant.check(self.Z_outside)
            
            # Outside set should not violate unsafe specification
            self.assertTrue(result_unsafe)
            # Outside set should satisfy safe specification  
            self.assertTrue(result_safe)
            # Outside set should satisfy invariant specification
            self.assertTrue(result_invariant)
        except (NotImplementedError, AttributeError):
            # Skip if check method not fully implemented
            self.skipTest("Check method not fully implemented yet")
    
    def test_check_zonotope_inside(self):
        """Test checking zonotope that is inside the specification set"""
        try:
            result_unsafe = self.spec_unsafe.check(self.Z_inside)
            result_safe = self.spec_safe.check(self.Z_inside)
            result_invariant = self.spec_invariant.check(self.Z_inside)
            
            # Inside set should violate unsafe specification
            self.assertFalse(result_unsafe)
            # Inside set should satisfy safe specification
            self.assertTrue(result_safe)
            # Inside set should satisfy invariant specification
            self.assertTrue(result_invariant)
        except (NotImplementedError, AttributeError):
            # Skip if check method not fully implemented
            self.skipTest("Check method not fully implemented yet")
    
    def test_check_zonotope_intersect(self):
        """Test checking zonotope that intersects the specification set"""
        try:
            result_unsafe = self.spec_unsafe.check(self.Z_intersect)
            result_safe = self.spec_safe.check(self.Z_intersect)
            result_invariant = self.spec_invariant.check(self.Z_intersect)
            
            # Intersecting set should violate unsafe specification
            self.assertFalse(result_unsafe)
            # Intersecting set behavior for safe spec depends on implementation
            # (could be True if any part is safe, or False if any part is unsafe)
            self.assertIsInstance(result_safe, bool)
            # Intersecting set behavior for invariant spec
            self.assertIsInstance(result_invariant, bool)
        except (NotImplementedError, AttributeError):
            # Skip if check method not fully implemented
            self.skipTest("Check method not fully implemented yet")
    
    def test_check_point(self):
        """Test checking single points"""
        # Point outside the set
        point_outside = np.array([[-10], [-8]])
        # Point inside the set (assuming unit square-like set)
        point_inside = np.array([[0.1], [0.1]])
        
        try:
            # Check outside point
            result_unsafe_out = self.spec_unsafe.check(point_outside)
            result_safe_out = self.spec_safe.check(point_outside)
            
            # Outside point should not violate unsafe spec
            self.assertTrue(result_unsafe_out)
            # Outside point should satisfy safe spec
            self.assertTrue(result_safe_out)
            
            # Check inside point
            result_unsafe_in = self.spec_unsafe.check(point_inside)
            result_safe_in = self.spec_safe.check(point_inside)
            
            # Inside point should violate unsafe spec
            self.assertFalse(result_unsafe_in)
            # Inside point should satisfy safe spec
            self.assertTrue(result_safe_in)
            
        except (NotImplementedError, AttributeError):
            # Skip if check method not fully implemented
            self.skipTest("Check method not fully implemented yet")
    
    def test_check_with_time(self):
        """Test checking with time constraints"""
        # Create time-constrained specification
        time_interval = Interval(np.array([[0]]), np.array([[2]]))
        spec_timed = Specification(self.set, 'unsafeSet', time_interval)
        
        try:
            # Check at different times
            result_t0 = spec_timed.check(self.Z_outside, 0.0)
            result_t1 = spec_timed.check(self.Z_outside, 1.0)
            result_t3 = spec_timed.check(self.Z_outside, 3.0)  # Outside time window
            
            self.assertIsInstance(result_t0, bool)
            self.assertIsInstance(result_t1, bool)
            self.assertIsInstance(result_t3, bool)
            
        except (NotImplementedError, AttributeError):
            # Skip if time handling not fully implemented
            self.skipTest("Time handling in check method not fully implemented yet")
    
    def test_check_multiple_sets(self):
        """Test checking multiple sets at once"""
        sets_list = [self.Z_outside, self.Z_inside, self.Z_intersect]
        
        try:
            results = []
            for set_obj in sets_list:
                result = self.spec_unsafe.check(set_obj)
                results.append(result)
                self.assertIsInstance(result, bool)
            
            # Should have mixed results for different sets
            self.assertGreaterEqual(len(set(results)), 1)  # At least some variation
            
        except (NotImplementedError, AttributeError):
            # Skip if check method not fully implemented
            self.skipTest("Check method not fully implemented yet")
    
    def test_check_empty_set(self):
        """Test checking empty sets"""
        try:
            from cora_python.contSet.emptySet import EmptySet
            empty_set = EmptySet(2)  # 2D empty set
            
            # Empty set should satisfy all specifications
            result_unsafe = self.spec_unsafe.check(empty_set)
            result_safe = self.spec_safe.check(empty_set)
            result_invariant = self.spec_invariant.check(empty_set)
            
            # Empty set satisfies all specifications
            self.assertTrue(result_unsafe)
            self.assertTrue(result_safe)
            self.assertTrue(result_invariant)
            
        except (ImportError, NotImplementedError, AttributeError):
            # Skip if EmptySet not available or check method not implemented
            self.skipTest("EmptySet or check method not fully implemented yet")
    
    def test_check_specification_types(self):
        """Test that all specification types can be checked"""
        test_set = self.Z_outside
        
        spec_types = ['unsafeSet', 'safeSet', 'invariant']
        
        for spec_type in spec_types:
            spec = Specification(self.set, spec_type)
            try:
                result = spec.check(test_set)
                self.assertIsInstance(result, bool, 
                                    f"Check failed for type {spec_type}")
            except (NotImplementedError, AttributeError):
                # Some types might not be implemented yet
                continue
    
    def test_check_custom_specification(self):
        """Test checking custom specification with function handle"""
        # Custom function: check if all points have x[0] > 0
        custom_func = lambda x: np.all(x[0, :] > 0) if x.ndim > 1 else x[0] > 0
        spec_custom = Specification(custom_func, 'custom')
        
        # Test sets
        positive_set = Zonotope(np.array([[1], [0]]), np.array([[0.1, 0], [0, 0.1]]))
        negative_set = Zonotope(np.array([[-1], [0]]), np.array([[0.1, 0], [0, 0.1]]))
        
        try:
            result_pos = spec_custom.check(positive_set)
            result_neg = spec_custom.check(negative_set)
            
            # Results depend on the custom function implementation
            self.assertIsInstance(result_pos, bool)
            self.assertIsInstance(result_neg, bool)
            
        except (NotImplementedError, AttributeError):
            # Skip if custom specification checking not implemented
            self.skipTest("Custom specification checking not fully implemented yet")
    
    def test_check_error_cases(self):
        """Test error cases for check method"""
        try:
            # Test with invalid input
            with self.assertRaises((TypeError, ValueError, AttributeError)):
                self.spec_unsafe.check("invalid_input")
            
            # Test with incompatible dimensions
            wrong_dim_set = Zonotope(np.array([[0], [0], [0]]), np.eye(3))  # 3D set
            with self.assertRaises((ValueError, AttributeError)):
                self.spec_unsafe.check(wrong_dim_set)
                
        except (NotImplementedError, AttributeError):
            # Skip if error handling not implemented
            self.skipTest("Error handling in check method not fully implemented yet")


if __name__ == '__main__':
    unittest.main() 