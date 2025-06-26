"""
test_dispEmptySet - unit test function for dispEmptySet

Syntax:
    pytest test_dispEmptySet.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: AI Assistant
Written: 2025
Last update: ---
Last revision: ---
"""

import pytest
import io
from contextlib import redirect_stdout
from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet


class TestDispEmptySet:
    """Test class for dispEmptySet function"""
    
    def capture_output(self, func, *args, **kwargs):
        """Helper to capture stdout output"""
        f = io.StringIO()
        with redirect_stdout(f):
            func(*args, **kwargs)
        return f.getvalue()
    
    def test_dispEmptySet_basic(self):
        """Test basic dispEmptySet functionality"""
        
        # Test with class name
        output = self.capture_output(dispEmptySet, "zonotope")
        assert "zonotope" in output
        assert "empty" in output.lower()
        
        # Test with different class names
        output = self.capture_output(dispEmptySet, "interval")
        assert "interval" in output
        assert "empty" in output.lower()
        
        output = self.capture_output(dispEmptySet, "polytope")
        assert "polytope" in output
        assert "empty" in output.lower()
    
    def test_dispEmptySet_with_dimensions(self):
        """Test dispEmptySet with dimension information"""
        
        # Test with dimensions
        output = self.capture_output(dispEmptySet, "zonotope", 3)
        assert "zonotope" in output
        assert "3" in output
        assert "empty" in output.lower()
        
        # Test with zero dimensions
        output = self.capture_output(dispEmptySet, "interval", 0)
        assert "interval" in output
        assert "0" in output
        
        # Test with large dimensions
        output = self.capture_output(dispEmptySet, "polytope", 100)
        assert "polytope" in output
        assert "100" in output
    
    def test_dispEmptySet_edge_cases(self):
        """Test edge cases for dispEmptySet"""
        
        # Test with empty string
        output = self.capture_output(dispEmptySet, "")
        assert "empty" in output.lower()
        
        # Test with long class name
        output = self.capture_output(dispEmptySet, "veryLongClassNameForTesting")
        assert "veryLongClassNameForTesting" in output
        
        # Test with special characters in class name
        output = self.capture_output(dispEmptySet, "class_with_underscore")
        assert "class_with_underscore" in output
    
    def test_dispEmptySet_formatting(self):
        """Test output formatting of dispEmptySet"""
        
        # Test that output is properly formatted
        output = self.capture_output(dispEmptySet, "zonotope", 5)
        lines = output.strip().split('\n')
        
        # Should have at least one line
        assert len(lines) >= 1
        
        # Should contain relevant information
        full_output = ' '.join(lines)
        assert "zonotope" in full_output
        assert "5" in full_output or "empty" in full_output.lower()


if __name__ == "__main__":
    pytest.main([__file__]) 