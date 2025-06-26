"""
test_verboseLog - unit test function for verboseLog

Syntax:
    pytest test_verboseLog.py

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
import sys
from contextlib import redirect_stdout, redirect_stderr
from cora_python.g.functions.verbose.verboseLog import (
    verboseLog, verboseLogReach, verboseLogAdaptive, 
    verboseLogHeader, verboseLogFooter
)


class TestVerboseLog:
    """Test class for verboseLog functions"""
    
    def capture_output(self, func, *args, **kwargs):
        """Helper to capture stdout output"""
        f = io.StringIO()
        with redirect_stdout(f):
            func(*args, **kwargs)
        return f.getvalue()
    
    def test_verboseLog_basic(self):
        """Test basic verboseLog functionality"""
        
        # Test verbose = 0 (no output)
        output = self.capture_output(verboseLog, 0, 5, 0.5, 0.0, 2.0)
        assert output == ""
        
        # Test verbose = 1 with all parameters
        output = self.capture_output(verboseLog, 1, 5, 0.5, 0.0, 2.0)
        assert "Step    5" in output
        assert "t =   0.5000" in output
        assert "25.0%" in output
        
        # Test verbose = 1 without tStart/tFinal
        output = self.capture_output(verboseLog, 1, 5, 0.5)
        assert "Step    5" in output
        assert "t =   0.5000" in output
        assert "%" not in output
        
        # Test verbose = 2 with detailed output
        output = self.capture_output(verboseLog, 2, 5, 0.5, 0.0, 2.0)
        assert "Step    5" in output
        assert "t =   0.5000" in output
        assert "25.0%" in output
        assert "remaining" in output
        
        # Test verbose = 3 with debug info
        output = self.capture_output(verboseLog, 3, 5, 0.5, 0.0, 2.0)
        assert "Step    5" in output
        assert "tStart" in output
        assert "tFinal" in output
    
    def test_verboseLog_partial_inputs(self):
        """Test verboseLog with partial inputs"""
        
        # Test with only step counter
        output = self.capture_output(verboseLog, 1, 5)
        assert "Step 5" in output
        
        # Test with only time
        output = self.capture_output(verboseLog, 1, None, 0.5)
        assert "t =   0.5000" in output
        
        # Test with neither k nor t
        output = self.capture_output(verboseLog, 1)
        assert output == ""
    
    def test_verboseLogReach(self):
        """Test verboseLogReach functionality"""
        
        # Test verbose = 0 (no output)
        output = self.capture_output(verboseLogReach, 0, 5, 0.5, 0.0, 2.0)
        assert output == ""
        
        # Test verbose = 1
        output = self.capture_output(verboseLogReach, 1, 5, 0.5, 0.0, 2.0)
        assert "Step    5" in output
        assert "t =   0.5000" in output
        assert "25.0%" in output
        
        # Test verbose = 2 with timeStep
        output = self.capture_output(verboseLogReach, 2, 5, 0.5, 0.0, 2.0, 0.1)
        assert "Step    5" in output
        assert "dt =   0.1000" in output
        assert "remaining" in output
        
        # Test verbose = 3 with error
        output = self.capture_output(verboseLogReach, 3, 5, 0.5, 0.0, 2.0, 0.1, 1e-6)
        assert "Step    5" in output
        assert "error" in output
        assert "1.000000e-06" in output
    
    def test_verboseLogAdaptive(self):
        """Test verboseLogAdaptive functionality"""
        
        # Test verbose = 0 (no output)
        output = self.capture_output(verboseLogAdaptive, 0, 5, 2, 0.5, 0.0, 2.0, 0.1)
        assert output == ""
        
        # Test verbose = 1
        output = self.capture_output(verboseLogAdaptive, 1, 5, 2, 0.5, 0.0, 2.0, 0.1)
        assert "Step    5.2" in output
        assert "t =   0.5000" in output
        assert "dt =   0.1000" in output
        
        # Test verbose = 2
        output = self.capture_output(verboseLogAdaptive, 2, 5, 2, 0.5, 0.0, 2.0, 0.1)
        assert "Step    5.2" in output
        assert "remaining" in output
        
        # Test verbose = 3 with error and bound
        output = self.capture_output(verboseLogAdaptive, 3, 5, 2, 0.5, 0.0, 2.0, 0.1, 1e-6, 1e-5)
        assert "Step    5.2" in output
        assert "error" in output
        assert "bound" in output
        assert "ratio" in output
        
        # Test verbose = 3 with only error
        output = self.capture_output(verboseLogAdaptive, 3, 5, 2, 0.5, 0.0, 2.0, 0.1, 1e-6)
        assert "Step    5.2" in output
        assert "error" in output
        assert "bound" not in output
    
    def test_verboseLogHeader(self):
        """Test verboseLogHeader functionality"""
        
        # Test verbose = 0 (no output)
        output = self.capture_output(verboseLogHeader, 0, "Test Algorithm", "Test System")
        assert output == ""
        
        # Test with algorithm and system name
        output = self.capture_output(verboseLogHeader, 1, "Test Algorithm", "Test System")
        assert "=" * 60 in output
        assert "CORA - Test Algorithm for system 'Test System'" in output
        
        # Test with only algorithm
        output = self.capture_output(verboseLogHeader, 1, "Test Algorithm")
        assert "CORA - Test Algorithm" in output
        
        # Test with no algorithm
        output = self.capture_output(verboseLogHeader, 1)
        assert "CORA - Reachability Analysis" in output
    
    def test_verboseLogFooter(self):
        """Test verboseLogFooter functionality"""
        
        # Test verbose = 0 (no output)
        output = self.capture_output(verboseLogFooter, 0, 1.234)
        assert output == ""
        
        # Test with computation time
        output = self.capture_output(verboseLogFooter, 1, 1.234)
        assert "=" * 60 in output
        assert "1.234 seconds" in output
        
        # Test without computation time
        output = self.capture_output(verboseLogFooter, 1)
        assert "=" * 60 in output
        assert "seconds" not in output
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        
        # Test with zero time span
        output = self.capture_output(verboseLog, 1, 5, 0.5, 0.5, 0.5)
        # Should not crash, but progress calculation might be undefined
        assert "Step    5" in output
        
        # Test with negative time span (edge case)
        output = self.capture_output(verboseLog, 1, 5, 0.5, 1.0, 0.0)
        assert "Step    5" in output
        
        # Test with very large numbers
        output = self.capture_output(verboseLog, 1, 999999, 1e10, 0.0, 2e10)
        assert "Step 999999" in output
        
        # Test with very small numbers
        output = self.capture_output(verboseLog, 1, 1, 1e-10, 0.0, 2e-10)
        assert "Step    1" in output


if __name__ == "__main__":
    pytest.main([__file__]) 