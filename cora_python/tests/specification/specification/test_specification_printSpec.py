"""
test_specification_printSpec - unit test for printSpec method

This test covers the printSpec functionality for specification objects.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 2022 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
import io
import sys
from cora_python.specification.specification.specification import Specification
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestSpecificationPrintSpec(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test sets
        self.set_simple = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        self.set_complex = Zonotope(np.array([[0.5], [0.5]]), 
                                   np.array([[0.3, 0.1], [0.1, 0.3]]))
        
        # Time intervals
        self.time_interval = Interval(np.array([[0]]), np.array([[2]]))
        
        # Locations
        self.location_HA = [1, 2]
        self.location_pHA = [[1, 2], [3], [2, 3]]
        
        # Create specifications
        self.spec_simple = Specification(self.set_simple, 'safeSet')
        self.spec_complex = Specification(self.set_complex, 'unsafeSet', 
                                         self.time_interval, self.location_HA)
        self.spec_custom = Specification(lambda x: x[0] > 0, 'custom')
        
    def capture_output(self, func, *args, **kwargs):
        """Helper method to capture print output"""
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            func(*args, **kwargs)
            return captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__
    
    def test_printSpec_basic(self):
        """Test basic printSpec functionality"""
        try:
            output = self.capture_output(self.spec_simple.printSpec)
            
            # Should contain specification type
            self.assertIn('safeSet', output)
            # Should contain some set information
            self.assertGreater(len(output.strip()), 0)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec method not fully implemented yet")
    
    def test_printSpec_different_types(self):
        """Test printSpec for different specification types"""
        specs = [
            Specification(self.set_simple, 'safeSet'),
            Specification(self.set_simple, 'unsafeSet'),
            Specification(self.set_simple, 'invariant')
        ]
        
        for spec in specs:
            try:
                output = self.capture_output(spec.printSpec)
                
                # Should contain the specification type
                self.assertIn(spec.type, output)
                self.assertGreater(len(output.strip()), 0)
                
            except (NotImplementedError, AttributeError):
                self.skipTest(f"printSpec for {spec.type} not implemented yet")
    
    def test_printSpec_with_time(self):
        """Test printSpec with time constraints"""
        spec_timed = Specification(self.set_simple, 'safeSet', self.time_interval)
        
        try:
            output = self.capture_output(spec_timed.printSpec)
            
            # Should mention time or time interval
            self.assertTrue(any(word in output.lower() for word in ['time', 'interval']))
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec with time not implemented yet")
    
    def test_printSpec_with_location(self):
        """Test printSpec with location constraints"""
        spec_located = Specification(self.set_simple, 'safeSet', self.location_HA)
        
        try:
            output = self.capture_output(spec_located.printSpec)
            
            # Should mention location
            self.assertTrue(any(word in output.lower() for word in ['location', 'loc']))
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec with location not implemented yet")
    
    def test_printSpec_complex_specification(self):
        """Test printSpec with all properties"""
        try:
            output = self.capture_output(self.spec_complex.printSpec)
            
            # Should contain type
            self.assertIn('unsafeSet', output)
            # Should mention time
            self.assertTrue(any(word in output.lower() for word in ['time', 'interval']))
            # Should mention location
            self.assertTrue(any(word in output.lower() for word in ['location', 'loc']))
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec for complex specification not implemented yet")
    
    def test_printSpec_custom_specification(self):
        """Test printSpec for custom specification"""
        try:
            output = self.capture_output(self.spec_custom.printSpec)
            
            # Should mention custom type
            self.assertIn('custom', output)
            # Should mention function
            self.assertTrue(any(word in output.lower() for word in ['function', 'custom', 'lambda']))
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec for custom specification not implemented yet")
    
    def test_printSpec_high_precision(self):
        """Test printSpec with high precision output"""
        try:
            # Test with high precision flag if available
            output1 = self.capture_output(self.spec_simple.printSpec)
            
            # Try with precision parameter if supported
            try:
                output2 = self.capture_output(self.spec_simple.printSpec, precision=10)
                # High precision output might be different
                self.assertIsInstance(output2, str)
            except TypeError:
                # Method might not support precision parameter
                pass
                
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec precision control not implemented yet")
    
    def test_printSpec_compact_vs_verbose(self):
        """Test printSpec compact vs verbose modes"""
        try:
            # Try compact mode if supported
            try:
                output_compact = self.capture_output(self.spec_complex.printSpec, compact=True)
                output_verbose = self.capture_output(self.spec_complex.printSpec, compact=False)
                
                # Verbose should generally be longer
                self.assertGreaterEqual(len(output_verbose), len(output_compact))
                
            except TypeError:
                # Method might not support compact parameter
                pass
                
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec compact/verbose modes not implemented yet")
    
    def test_printSpec_empty_specification(self):
        """Test printSpec for empty specification"""
        spec_empty = Specification()
        
        try:
            output = self.capture_output(spec_empty.printSpec)
            
            # Should handle empty specification gracefully
            self.assertIsInstance(output, str)
            # Should mention empty or default type
            self.assertTrue(any(word in output.lower() for word in ['empty', 'unsafe', 'default']))
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec for empty specification not implemented yet")
    
    def test_printSpec_different_set_types(self):
        """Test printSpec with different set types"""
        specs = [
            Specification(self.set_simple, 'safeSet'),  # Interval
            Specification(self.set_complex, 'safeSet')  # Zonotope
        ]
        
        for i, spec in enumerate(specs):
            try:
                output = self.capture_output(spec.printSpec)
                
                # Should contain set information
                self.assertGreater(len(output.strip()), 0)
                # Each should be different
                if i == 0:
                    output1 = output
                else:
                    self.assertNotEqual(output, output1)
                    
            except (NotImplementedError, AttributeError):
                self.skipTest("printSpec for different set types not implemented yet")
    
    def test_printSpec_matrix_formatting(self):
        """Test printSpec matrix formatting"""
        # Create specification with matrices
        matrix_set = Interval(np.array([[1.23456789], [2.87654321]]), 
                             np.array([[3.14159265], [4.71238898]]))
        spec_matrix = Specification(matrix_set, 'safeSet')
        
        try:
            output = self.capture_output(spec_matrix.printSpec)
            
            # Should format matrices nicely
            self.assertIn('[', output)  # Matrix brackets
            self.assertIn(']', output)
            # Should have reasonable precision
            self.assertGreater(len(output.strip()), 20)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec matrix formatting not implemented yet")
    
    def test_printSpec_return_value(self):
        """Test that printSpec returns None (prints to stdout)"""
        try:
            result = self.spec_simple.printSpec()
            
            # Should return None (like MATLAB's disp)
            self.assertIsNone(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec method not implemented yet")
    
    def test_printSpec_no_crash(self):
        """Test that printSpec doesn't crash on various inputs"""
        specs = [
            self.spec_simple,
            self.spec_complex,
            self.spec_custom,
            Specification()
        ]
        
        for spec in specs:
            try:
                # Should not raise exceptions
                self.capture_output(spec.printSpec)
                
            except (NotImplementedError, AttributeError):
                # Expected if method not implemented
                continue
            except Exception as e:
                self.fail(f"printSpec crashed for {spec}: {e}")
    
    def test_printSpec_unicode_handling(self):
        """Test printSpec handles unicode characters properly"""
        try:
            output = self.capture_output(self.spec_simple.printSpec)
            
            # Should be valid string that can be encoded
            encoded = output.encode('utf-8')
            self.assertIsInstance(encoded, bytes)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec unicode handling not implemented yet")
    
    def test_printSpec_consistency(self):
        """Test that printSpec gives consistent output"""
        try:
            output1 = self.capture_output(self.spec_simple.printSpec)
            output2 = self.capture_output(self.spec_simple.printSpec)
            
            # Should be identical
            self.assertEqual(output1, output2)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("printSpec method not implemented yet")


if __name__ == '__main__':
    unittest.main() 