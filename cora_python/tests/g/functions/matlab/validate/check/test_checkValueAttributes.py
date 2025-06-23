"""
test_checkValueAttributes - unit tests for checkValueAttributes validation function

Syntax:
    python -m pytest cora_python/tests/g/functions/matlab/validate/check/test_checkValueAttributes.py

Authors: MATLAB original tests, Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.validate.check.checkValueAttributes import checkValueAttributes
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestCheckValueAttributes:
    """Test class for checkValueAttributes function."""

    def test_numeric_class_validation(self):
        """Test validation of numeric class - this is the fix we implemented"""
        # Test scalar numeric values
        assert checkValueAttributes(5, 'numeric', ['finite'])
        assert checkValueAttributes(3.14, 'numeric', ['finite'])
        assert checkValueAttributes(np.float64(2.5), 'numeric', ['finite'])
        
        # Test numpy arrays (this was the main fix)
        arr_int = np.array([1, 2, 3])
        assert checkValueAttributes(arr_int, 'numeric', ['finite'])
        
        arr_float = np.array([1.5, 2.7, 3.1])
        assert checkValueAttributes(arr_float, 'numeric', ['finite'])
        
        arr_2d = np.array([[1, 2], [3, 4]])
        assert checkValueAttributes(arr_2d, 'numeric', ['finite'])
        
        # Test complex arrays
        arr_complex = np.array([1+2j, 3+4j])
        assert checkValueAttributes(arr_complex, 'numeric', ['finite'])
        
        # Non-numeric should fail
        assert not checkValueAttributes("string", 'numeric', ['finite'])
        assert not checkValueAttributes([1, 2, 3], 'numeric', ['finite'])

    def test_finite_attribute(self):
        """Test finite attribute validation"""
        # Finite values should pass
        assert checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['finite'])
        assert checkValueAttributes(5.0, 'numeric', ['finite'])
        
        # Infinite values should fail
        assert not checkValueAttributes(np.array([1, np.inf, 3]), 'numeric', ['finite'])
        assert not checkValueAttributes(np.inf, 'numeric', ['finite'])
        assert not checkValueAttributes(np.array([1, -np.inf, 3]), 'numeric', ['finite'])
        
        # NaN values should fail finite test
        assert not checkValueAttributes(np.array([1, np.nan, 3]), 'numeric', ['finite'])
        assert not checkValueAttributes(np.nan, 'numeric', ['finite'])

    def test_nonnan_attribute(self):
        """Test nonnan attribute validation"""
        # Non-NaN values should pass
        assert checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['nonnan'])
        assert checkValueAttributes(5.0, 'numeric', ['nonnan'])
        assert checkValueAttributes(np.inf, 'numeric', ['nonnan'])  # inf is not nan
        
        # NaN values should fail
        assert not checkValueAttributes(np.array([1, np.nan, 3]), 'numeric', ['nonnan'])
        assert not checkValueAttributes(np.nan, 'numeric', ['nonnan'])

    def test_matrix_attribute(self):
        """Test matrix attribute validation"""
        # 2D arrays should pass
        assert checkValueAttributes(np.array([[1, 2], [3, 4]]), 'numeric', ['matrix'])
        assert checkValueAttributes(np.zeros((3, 2)), 'numeric', ['matrix'])
        
        # 1D arrays should fail
        assert not checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['matrix'])
        
        # 3D arrays should fail
        assert not checkValueAttributes(np.zeros((2, 2, 2)), 'numeric', ['matrix'])
        
        # Scalars should fail
        assert not checkValueAttributes(5, 'numeric', ['matrix'])

    def test_vector_attribute(self):
        """Test vector attribute validation"""
        # 1D arrays should pass
        assert checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['vector'])
        
        # Row vectors (2D with one row) should pass
        assert checkValueAttributes(np.array([[1, 2, 3]]), 'numeric', ['vector'])
        
        # Column vectors (2D with one column) should pass
        assert checkValueAttributes(np.array([[1], [2], [3]]), 'numeric', ['vector'])
        
        # 2D matrices should fail
        assert not checkValueAttributes(np.array([[1, 2], [3, 4]]), 'numeric', ['vector'])
        
        # 3D arrays should fail
        assert not checkValueAttributes(np.zeros((2, 2, 2)), 'numeric', ['vector'])

    def test_positive_attribute(self):
        """Test positive attribute validation"""
        # All positive values should pass
        assert checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['positive'])
        assert checkValueAttributes(5.0, 'numeric', ['positive'])
        
        # Arrays with zero or negative should fail
        assert not checkValueAttributes(np.array([1, 0, 3]), 'numeric', ['positive'])
        assert not checkValueAttributes(np.array([1, -1, 3]), 'numeric', ['positive'])
        assert not checkValueAttributes(0, 'numeric', ['positive'])
        assert not checkValueAttributes(-1, 'numeric', ['positive'])

    def test_nonnegative_attribute(self):
        """Test nonnegative attribute validation"""
        # Non-negative values should pass
        assert checkValueAttributes(np.array([0, 1, 2]), 'numeric', ['nonnegative'])
        assert checkValueAttributes(0.0, 'numeric', ['nonnegative'])
        assert checkValueAttributes(5.0, 'numeric', ['nonnegative'])
        
        # Negative values should fail
        assert not checkValueAttributes(np.array([1, -1, 3]), 'numeric', ['nonnegative'])
        assert not checkValueAttributes(-1, 'numeric', ['nonnegative'])

    def test_integer_attribute(self):
        """Test integer attribute validation"""
        # Integer values should pass
        assert checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['integer'])
        assert checkValueAttributes(5, 'numeric', ['integer'])
        assert checkValueAttributes(np.array([1.0, 2.0, 3.0]), 'numeric', ['integer'])  # Exact floats
        
        # Non-integer values should fail
        assert not checkValueAttributes(np.array([1.5, 2, 3]), 'numeric', ['integer'])
        assert not checkValueAttributes(3.14, 'numeric', ['integer'])

    def test_nonempty_attribute(self):
        """Test nonempty attribute validation"""
        # Non-empty arrays should pass
        assert checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['nonempty'])
        assert checkValueAttributes(5, 'numeric', ['nonempty'])
        
        # Empty arrays should fail
        assert not checkValueAttributes(np.array([]), 'numeric', ['nonempty'])

    def test_square_attribute(self):
        """Test square attribute validation"""
        # Square matrices should pass
        assert checkValueAttributes(np.array([[1, 2], [3, 4]]), 'numeric', ['square'])
        assert checkValueAttributes(np.eye(3), 'numeric', ['square'])
        
        # Non-square matrices should fail
        assert not checkValueAttributes(np.array([[1, 2, 3], [4, 5, 6]]), 'numeric', ['square'])
        assert not checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['square'])

    def test_2d_attribute(self):
        """Test 2d attribute validation"""
        # 2D arrays should pass
        assert checkValueAttributes(np.array([[1, 2], [3, 4]]), 'numeric', ['2d'])
        assert checkValueAttributes(np.zeros((2, 3)), 'numeric', ['2d'])
        
        # 1D arrays should fail
        assert not checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['2d'])
        
        # 3D arrays should fail
        assert not checkValueAttributes(np.zeros((2, 2, 2)), 'numeric', ['2d'])

    def test_multiple_attributes(self):
        """Test validation with multiple attributes"""
        # Array that satisfies all attributes
        arr = np.array([[1, 2], [3, 4]])
        assert checkValueAttributes(arr, 'numeric', ['finite', 'positive', 'integer', 'matrix'])
        
        # Array that fails one attribute
        arr_mixed = np.array([[1.5, 2], [3, 4]])
        assert not checkValueAttributes(arr_mixed, 'numeric', ['finite', 'positive', 'integer', 'matrix'])

    def test_negation_attributes(self):
        """Test negated attributes (non-prefix)"""
        # nonfinite should be opposite of finite
        assert not checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['nonfinite'])
        assert checkValueAttributes(np.array([1, np.inf, 3]), 'numeric', ['nonfinite'])
        
        # nonpositive should be opposite of positive
        assert not checkValueAttributes(np.array([1, 2, 3]), 'numeric', ['nonpositive'])
        assert checkValueAttributes(np.array([-1, 0, -3]), 'numeric', ['nonpositive'])

    def test_string_class_validation(self):
        """Test string/char class validation"""
        # String values should pass
        assert checkValueAttributes("test", 'char', [])
        assert checkValueAttributes("test", 'string', [])
        
        # Non-string values should fail
        assert not checkValueAttributes(123, 'char', [])
        assert not checkValueAttributes(np.array([1, 2, 3]), 'string', [])

    def test_logical_class_validation(self):
        """Test logical class validation"""
        # Boolean values should pass
        assert checkValueAttributes(True, 'logical', [])
        assert checkValueAttributes(False, 'logical', [])
        assert checkValueAttributes(np.array([True, False, True]), 'logical', [])
        
        # Non-boolean values should fail
        assert not checkValueAttributes(1, 'logical', [])
        assert not checkValueAttributes(np.array([1, 0, 1]), 'logical', [])

    def test_unknown_attribute_error(self):
        """Test error handling for unknown attributes"""
        with pytest.raises(CORAerror):
            checkValueAttributes(5, 'numeric', ['unknownAttribute'])

    def test_class_mismatch(self):
        """Test class mismatch scenarios"""
        # Numeric value with wrong class
        assert not checkValueAttributes(5, 'char', ['finite'])
        
        # String value with numeric class
        assert not checkValueAttributes("test", 'numeric', ['nonempty'])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty string
        assert not checkValueAttributes("", 'char', ['nonempty'])
        assert checkValueAttributes("", 'char', [])
        
        # Single element arrays
        assert checkValueAttributes(np.array([5]), 'numeric', ['positive'])
        
        # Very small/large numbers
        assert checkValueAttributes(1e-100, 'numeric', ['positive', 'finite'])
        assert checkValueAttributes(1e100, 'numeric', ['positive', 'finite'])

    def test_combined_validation_integration(self):
        """Test integration with inputArgsCheck style validation"""
        # Test the specific case that was failing in ConZonotope
        c = np.array([[0.0], [0.0]], dtype=np.float64)
        G = np.array([[1.5, -1.5, 0.5], [1.0, 0.5, -1.0]], dtype=np.float64)
        
        # Test combined [c, G] matrix validation
        combined = np.column_stack([c, G])
        assert checkValueAttributes(combined, 'numeric', ['finite'])
        
        # Test individual validation
        assert checkValueAttributes(c, 'numeric', ['finite'])
        assert checkValueAttributes(G, 'numeric', ['finite'])


def test_checkValueAttributes():
    """Main test function for checkValueAttributes."""
    test = TestCheckValueAttributes()
    
    # Run all tests
    test.test_numeric_class_validation()
    test.test_finite_attribute()
    test.test_nonnan_attribute()
    test.test_matrix_attribute()
    test.test_vector_attribute()
    test.test_positive_attribute()
    test.test_nonnegative_attribute()
    test.test_integer_attribute()
    test.test_nonempty_attribute()
    test.test_square_attribute()
    test.test_2d_attribute()
    test.test_multiple_attributes()
    test.test_negation_attributes()
    test.test_string_class_validation()
    test.test_logical_class_validation()
    test.test_unknown_attribute_error()
    test.test_class_mismatch()
    test.test_edge_cases()
    test.test_combined_validation_integration()
    
    print("test_checkValueAttributes: all tests passed")


if __name__ == "__main__":
    test_checkValueAttributes() 