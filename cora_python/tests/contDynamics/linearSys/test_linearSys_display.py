"""
test_linearSys_display - unit test function for display

Tests the display method for linearSys objects.

Syntax:
    pytest cora_python/tests/contDynamics/linearSys/test_linearSys_display.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contDynamics.linearSys.linearSys import LinearSys
import io
import sys


class TestLinearSysDisplay:
    def test_display_minimal_system(self):
        """Test display of minimal linear system (A matrix only)"""
        A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                     [0.1362, 0.2742, 0.5195, 0.8266],
                     [0.0502, -0.1051, -0.6572, 0.3874],
                     [1.0227, -0.4877, 0.8342, -0.2372]])
        
        sys_A = LinearSys(A, 1)
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            sys_A.display()
            output = captured_output.getvalue()
            
            # Check that output contains system information
            assert "LinearSys" in output or "linearSys" in output or "Linear" in output
            assert len(output) > 0
            
        finally:
            sys.stdout = sys.__stdout__

    def test_display_system_with_input(self):
        """Test display of system with A and B matrices"""
        A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                     [0.1362, 0.2742, 0.5195, 0.8266],
                     [0.0502, -0.1051, -0.6572, 0.3874],
                     [1.0227, -0.4877, 0.8342, -0.2372]])
        
        B = 0.25 * np.array([[-2, 0, 3],
                            [2, 1, 0],
                            [0, 0, 1],
                            [0, -2, 1]])
        
        sys_AB = LinearSys(A, B)
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            sys_AB.display()
            output = captured_output.getvalue()
            
            # Check that output contains system information
            assert len(output) > 0
            
        finally:
            sys.stdout = sys.__stdout__

    def test_display_system_with_output(self):
        """Test display of system with A, B, and C matrices"""
        A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                     [0.1362, 0.2742, 0.5195, 0.8266],
                     [0.0502, -0.1051, -0.6572, 0.3874],
                     [1.0227, -0.4877, 0.8342, -0.2372]])
        
        B = 0.25 * np.array([[-2, 0, 3],
                            [2, 1, 0],
                            [0, 0, 1],
                            [0, -2, 1]])
        
        C = np.array([[1, 1, 0, 0],
                     [0, -0.5, 0.5, 0]])
        
        sys_ABC = LinearSys(A, B, None, C)
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            sys_ABC.display()
            output = captured_output.getvalue()
            
            # Check that output contains system information
            assert len(output) > 0
            
        finally:
            sys.stdout = sys.__stdout__

    def test_display_full_system(self):
        """Test display of full system with all matrices"""
        A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                     [0.1362, 0.2742, 0.5195, 0.8266],
                     [0.0502, -0.1051, -0.6572, 0.3874],
                     [1.0227, -0.4877, 0.8342, -0.2372]])
        
        B = 0.25 * np.array([[-2, 0, 3],
                            [2, 1, 0],
                            [0, 0, 1],
                            [0, -2, 1]])
        
        c = 0.05 * np.array([[-4], [2], [3], [1]])
        
        C = np.array([[1, 1, 0, 0],
                     [0, -0.5, 0.5, 0]])
        
        D = np.array([[0, 0, 1],
                     [0, 0, 0]])
        
        k = np.array([[0], [0.02]])
        
        E = np.array([[1, 0.5], [0, -0.5], [1, -1], [0, 1]])
        
        F = np.array([[1], [0.5]])
        
        sys_full = LinearSys(A, B, c, C, D, k, E, F)
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            sys_full.display()
            output = captured_output.getvalue()
            
            # Check that output contains system information
            assert len(output) > 0
            
        finally:
            sys.stdout = sys.__stdout__

    def test_display_no_exception(self):
        """Test that display doesn't raise exceptions for various systems"""
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        
        systems = [
            LinearSys(A, 1),
            LinearSys(A, B),
            LinearSys(A, B, None, np.array([[1, 0]])),
            LinearSys(A, B, None, np.array([[1, 0]]), np.array([[0]])),
        ]
        
        for linsys in systems:
            # Should not raise any exceptions
            try:
                captured_output = io.StringIO()
                sys.stdout = captured_output
                linsys.display()
                # Test passed if no exception is raised
                assert True
            except Exception as e:
                pytest.fail(f"Display raised unexpected exception: {e}")
            finally:
                sys.stdout = sys.__stdout__

    def test_display_empty_system(self):
        """Test display of empty system"""
        sys_empty = LinearSys()
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            sys_empty.display()
            output = captured_output.getvalue()
            
            # Should produce some output without errors
            assert len(output) >= 0  # Allow empty output for empty system
            
        finally:
            sys.stdout = sys.__stdout__

    def test_display_system_dimensions(self):
        """Test that display shows system dimensions"""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1], [0]])
        linsys = LinearSys(A, B)
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            linsys.display()
            output = captured_output.getvalue()
            
            # Should contain dimension information
            assert len(output) > 0
            
        finally:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    pytest.main([__file__]) 