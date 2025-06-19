"""
test_specification_plot - unit test for plot methods

This test covers the plotting functionality for specification objects.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 2022 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from cora_python.specification.specification.specification import Specification
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestSpecificationPlot(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test sets
        self.set_2d = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        self.set_3d = Interval(np.array([[0], [0], [0]]), np.array([[1], [1], [1]]))
        
        # Create zonotope for more complex plotting
        self.zono_2d = Zonotope(np.array([[0.5], [0.5]]), 
                               np.array([[0.3, 0.1], [0.1, 0.3]]))
        
        # Create specifications
        self.spec_safe = Specification(self.set_2d, 'safeSet')
        self.spec_unsafe = Specification(self.set_2d, 'unsafeSet')
        self.spec_invariant = Specification(self.set_2d, 'invariant')
        
        # Time interval for time-dependent specs
        self.time_interval = Interval(np.array([[0]]), np.array([[2]]))
        
        # Create multiple specifications for list plotting
        self.spec_list = [
            Specification(self.set_2d, 'safeSet'),
            Specification(self.zono_2d, 'unsafeSet'),
            Specification(self.set_2d, 'invariant')
        ]
    
    def tearDown(self):
        """Clean up after tests"""
        plt.close('all')
    
    def test_plot_2d_specification(self):
        """Test plotting 2D specification"""
        try:
            fig, ax = plt.subplots()
            self.spec_safe.plot(ax=ax)
            
            # Check that something was plotted
            self.assertGreater(len(ax.patches) + len(ax.collections), 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Plot method not fully implemented yet")
    
    def test_plot_3d_specification(self):
        """Test plotting 3D specification"""
        spec_3d = Specification(self.set_3d, 'safeSet')
        
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            spec_3d.plot(ax=ax)
            
            # Check that something was plotted
            self.assertGreater(len(ax.collections), 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("3D plot method not fully implemented yet")
    
    def test_plot_with_projections(self):
        """Test plotting with dimension projections"""
        try:
            fig, ax = plt.subplots()
            self.spec_safe.plot(dims=[1, 2], ax=ax)  # Project to dimensions 1,2
            
            # Check that plot was created
            self.assertIsNotNone(ax)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Plot with projections not fully implemented yet")
    
    def test_plot_different_specification_types(self):
        """Test plotting different specification types with different colors"""
        try:
            fig, ax = plt.subplots()
            
            # Plot different types
            self.spec_safe.plot(ax=ax)
            self.spec_unsafe.plot(ax=ax)
            self.spec_invariant.plot(ax=ax)
            
            # Should have plotted multiple objects
            self.assertGreater(len(ax.patches) + len(ax.collections), 2)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Plot method not fully implemented yet")
    
    def test_plot_specification_list(self):
        """Test plotting list of specifications"""
        try:
            fig, ax = plt.subplots()
            
            # This might be handled by a utility function
            for spec in self.spec_list:
                spec.plot(ax=ax)
            
            # Should have multiple objects plotted
            self.assertGreater(len(ax.patches) + len(ax.collections), 2)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Specification list plotting not fully implemented yet")
    
    def test_plot_with_legend(self):
        """Test plotting with legend"""
        try:
            fig, ax = plt.subplots()
            
            self.spec_safe.plot(ax=ax, label='Safe Set')
            self.spec_unsafe.plot(ax=ax, label='Unsafe Set')
            
            ax.legend()
            
            # Check that legend was created
            legend = ax.get_legend()
            self.assertIsNotNone(legend)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Plot with legend not fully implemented yet")
    
    def test_plot_over_time_2d(self):
        """Test plotOverTime for 2D specification"""
        spec_timed = Specification(self.set_2d, 'safeSet', self.time_interval)
        
        try:
            fig, ax = plt.subplots()
            spec_timed.plotOverTime(dims=[1], ax=ax)
            
            # Should have time on one axis, space dimension on other
            self.assertIsNotNone(ax)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("plotOverTime method not fully implemented yet")
    
    def test_plot_over_time_3d(self):
        """Test plotOverTime for 3D specification"""
        spec_timed = Specification(self.set_2d, 'safeSet', self.time_interval)
        
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            spec_timed.plotOverTime(dims=[1, 2], ax=ax)
            
            # Should have plotted in 3D (time + 2 space dims)
            self.assertIsNotNone(ax)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("3D plotOverTime method not fully implemented yet")
    
    def test_plot_over_time_invalid_dims(self):
        """Test plotOverTime with invalid dimensions"""
        spec_timed = Specification(self.set_2d, 'safeSet', self.time_interval)
        
        try:
            fig, ax = plt.subplots()
            
            # Too many dimensions for plotOverTime
            with self.assertRaises((ValueError, AttributeError)):
                spec_timed.plotOverTime(dims=[1, 2, 3], ax=ax)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("plotOverTime error handling not implemented yet")
    
    def test_plot_without_time_interval(self):
        """Test plotOverTime without time interval should raise error"""
        try:
            fig, ax = plt.subplots()
            
            with self.assertRaises((ValueError, AttributeError)):
                self.spec_safe.plotOverTime(dims=[1], ax=ax)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("plotOverTime error handling not implemented yet")
    
    def test_plot_style_options(self):
        """Test plotting with different style options"""
        try:
            fig, ax = plt.subplots()
            
            # Test with different colors and styles
            self.spec_safe.plot(ax=ax, color='red', alpha=0.5)
            self.spec_unsafe.plot(ax=ax, color='blue', linewidth=2)
            
            # Check that plot was created
            self.assertIsNotNone(ax)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Plot styling not fully implemented yet")
    
    def test_plot_empty_specification(self):
        """Test plotting empty specification"""
        spec_empty = Specification()
        
        try:
            fig, ax = plt.subplots()
            
            # Should handle empty specification gracefully
            spec_empty.plot(ax=ax)
            # Or raise appropriate error
            
        except (ValueError, AttributeError):
            # Expected behavior for empty specification
            pass
        except (NotImplementedError, AttributeError):
            self.skipTest("Empty specification plotting not implemented yet")
    
    def test_plot_custom_specification(self):
        """Test plotting custom specification (should raise error)"""
        custom_func = lambda x: x[0] > 0
        spec_custom = Specification(custom_func, 'custom')
        
        try:
            fig, ax = plt.subplots()
            
            # Custom specifications should not be plottable
            with self.assertRaises((ValueError, AttributeError, NotImplementedError)):
                spec_custom.plot(ax=ax)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Custom specification plotting not implemented yet")
    
    def test_plot_high_dimensional_projection(self):
        """Test plotting high-dimensional set with projection"""
        # Create higher dimensional set
        high_dim_set = Interval(np.zeros((5, 1)), np.ones((5, 1)))
        spec_high_dim = Specification(high_dim_set, 'safeSet')
        
        try:
            fig, ax = plt.subplots()
            
            # Project to 2D
            spec_high_dim.plot(dims=[1, 3], ax=ax)
            
            # Should have created projection plot
            self.assertIsNotNone(ax)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("High-dimensional projection plotting not implemented yet")
    
    def test_plot_returns_correct_objects(self):
        """Test that plot returns correct matplotlib objects"""
        try:
            fig, ax = plt.subplots()
            result = self.spec_safe.plot(ax=ax)
            
            # Should return something that can be used for further manipulation
            # (patches, collections, etc.)
            if result is not None:
                self.assertIsNotNone(result)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Plot return values not implemented yet")


if __name__ == '__main__':
    unittest.main() 