"""
test_capsule_display - unit test function of display

Tests the display method for capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule_display.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule
import io
import sys


class TestCapsuleDisplay:
    def test_display_empty_capsule(self):
        """Test display of empty capsule"""
        n = 2
        C = Capsule.empty(n)
        
        # Get display output
        output = C.display()
        
        # Check that output contains relevant information
        assert "empty" in output.lower() or "Empty" in output
        assert "capsule" in output.lower() or "Capsule" in output

    def test_display_center_only(self):
        """Test display of capsule with center only"""
        c = np.array([2, 0])
        C = Capsule(c)
        
        # Get display output
        output = C.display()
        
        # Check that output contains center values
        assert "2" in output
        assert "0" in output
        assert "capsule" in output.lower() or "Capsule" in output

    def test_display_center_generator(self):
        """Test display of capsule with center and generator"""
        c = np.array([2, 0])
        g = np.array([1, -1])
        C = Capsule(c, g)
        
        # Get display output
        output = C.display()
        
        # Check that output contains center and generator values
        assert "2" in output
        assert "1" in output
        assert "-1" in output

    def test_display_full_capsule(self):
        """Test display of full capsule with center, generator, and radius"""
        c = np.array([2, 0])
        g = np.array([1, -1])
        r = 0.5
        C = Capsule(c, g, r)
        
        # Get display output
        output = C.display()
        
        # Check that output contains all values
        assert "2" in output
        assert "1" in output
        assert "-1" in output
        assert "0.5" in output or "0.50" in output

    def test_display_no_exception(self):
        """Test that display doesn't raise exceptions"""
        # Test various capsule configurations
        capsules = [
            Capsule.empty(2),
            Capsule.origin(3),
            Capsule(np.array([1, 2])),
            Capsule(np.array([1, 2]), np.array([0, 1])),
            Capsule(np.array([1, 2]), np.array([0, 1]), 1.5)
        ]
        
        for C in capsules:
            # Should not raise any exceptions
            try:
                output = C.display()
                # Test passed if no exception is raised
                assert isinstance(output, str)
            except Exception as e:
                pytest.fail(f"Display raised unexpected exception: {e}")

    def test_display_different_dimensions(self):
        """Test display for different dimensions"""
        # 1D capsule
        C_1d = Capsule(np.array([5]), np.array([2]), 1.0)
        
        # 4D capsule
        C_4d = Capsule(np.array([1, 2, 3, 4]), np.array([0, 1, 0, 1]), 0.5)
        
        capsules = [C_1d, C_4d]
        
        for C in capsules:
            output = C.display()
            
            # Should produce some output without errors
            assert len(output) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 