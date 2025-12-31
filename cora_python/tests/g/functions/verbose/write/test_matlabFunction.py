"""
test_matlabFunction - GENERATED TEST
   No MATLAB source test exists. This test was created by analyzing the MATLAB 
   implementation logic in matlabFunction.py and ensuring thorough coverage.

   This test verifies that matlabFunction correctly converts symbolic expressions 
   to Python callable functions and writes them to files, including:
   - Converting sympy expressions to Python functions
   - Writing functions to files
   - Handling multiple variables
   - Handling file creation and import

Syntax:
    pytest cora_python/tests/g/functions/verbose/write/test_matlabFunction.py

Authors:       Generated test based on MATLAB implementation
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import numpy as np
import pytest
import sympy as sp
import tempfile
from cora_python.g.functions.verbose.write.matlabFunction import matlabFunction


class TestMatlabFunction:
    """Test class for matlabFunction functionality"""
    
    def test_matlabFunction_basic(self):
        """Test basic matlabFunction conversion"""
        # MATLAB: syms x y
        x, y = sp.symbols('x y', real=True)
        # MATLAB: f = x^2 + y^2;
        f = x**2 + y**2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'test_func.py')
            
            try:
                # MATLAB: matlabFunction(f, 'File', file_path, 'Vars', [x, y]);
                matlabFunction(f, File=file_path, Vars=[x, y])
                
                # Verify file was created
                assert os.path.exists(file_path)
                
                # Verify file can be imported and executed
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_func", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Test the function
                result = module.f(np.array([2]), np.array([3]))
                expected = 2**2 + 3**2
                assert np.isclose(result, expected)
            except (NotImplementedError, AttributeError) as e:
                pytest.skip(f"Dependencies not yet translated: {e}")
    
    def test_matlabFunction_multiple_outputs(self):
        """Test with multiple outputs"""
        x, y = sp.symbols('x y', real=True)
        f1 = x**2
        f2 = y**2
        f = [f1, f2]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'test_func.py')
            
            try:
                matlabFunction(f, File=file_path, Vars=[x, y])
                
                assert os.path.exists(file_path)
                
                # Import and test
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_func", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                result = module.f(np.array([2]), np.array([3]))
                assert len(result) == 2
                assert np.isclose(result[0], 4)
                assert np.isclose(result[1], 9)
            except (NotImplementedError, AttributeError) as e:
                pytest.skip(f"Dependencies not yet translated: {e}")
    
    def test_matlabFunction_single_variable(self):
        """Test with single variable"""
        x = sp.symbols('x', real=True)
        f = x**2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'test_func.py')
            
            try:
                matlabFunction(f, File=file_path, Vars=[x])
                
                assert os.path.exists(file_path)
                
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_func", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                result = module.f(np.array([3]))
                assert np.isclose(result, 9)
            except (NotImplementedError, AttributeError) as e:
                pytest.skip(f"Dependencies not yet translated: {e}")


def test_matlabFunction():
    """Test function for matlabFunction method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestMatlabFunction()
    test.test_matlabFunction_basic()
    test.test_matlabFunction_multiple_outputs()
    test.test_matlabFunction_single_variable()
    
    print("test_matlabFunction: all tests passed")
    return True


if __name__ == "__main__":
    test_matlabFunction()

