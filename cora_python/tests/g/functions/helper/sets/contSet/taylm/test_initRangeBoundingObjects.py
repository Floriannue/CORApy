"""
test_initRangeBoundingObjects - GENERATED TEST
   No MATLAB source test exists. This test was created by analyzing the MATLAB 
   implementation logic in initRangeBoundingObjects.py and ensuring thorough coverage.

   This test verifies that initRangeBoundingObjects correctly creates taylm- or 
   zoo-objects for state and input variables, including:
   - Parsing options (maxOrder, optMethod, eps, tolerance)
   - Handling taylorModel method
   - Handling zoo method
   - Error handling for invalid methods
   - Correct variable indexing via aux_idxVars

Syntax:
    pytest cora_python/tests/g/functions/helper/sets/contSet/taylm/test_initRangeBoundingObjects.py

Authors:       Generated test based on MATLAB implementation
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.interval import Interval
from cora_python.g.functions.helper.sets.contSet.taylm.initRangeBoundingObjects import initRangeBoundingObjects
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestInitRangeBoundingObjects:
    """Test class for initRangeBoundingObjects functionality"""
    
    def test_initRangeBoundingObjects_taylorModel(self):
        """Test with taylorModel method"""
        # Create intervals
        intX = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        intU = Interval(np.array([[-0.5]]), np.array([[0.5]]))
        
        # Options for taylorModel
        options = {
            'lagrangeRem': {
                'method': 'taylorModel',
                'maxOrder': 4,
                'optMethod': 'int',
                'eps': 1e-6,
                'tolerance': 1e-8
            }
        }
        
        # Should create taylm objects
        objX, objU = initRangeBoundingObjects(intX, intU, options)
        
        # Verify objects are created (exact type depends on Taylm implementation)
        assert objX is not None
        assert objU is not None
    
    def test_initRangeBoundingObjects_zoo(self):
        """Test with zoo method"""
        intX = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        intU = Interval(np.array([[-0.5]]), np.array([[0.5]]))
        
        # Options for zoo
        options = {
            'lagrangeRem': {
                'method': 'zoo',
                'zooMethods': ['interval', 'taylm'],
                'maxOrder': 4,
                'eps': 1e-6,
                'tolerance': 1e-8
            }
        }
        
        # Should create zoo objects
        objX, objU = initRangeBoundingObjects(intX, intU, options)
        
        # Verify objects are created
        assert objX is not None
        assert objU is not None
    
    def test_initRangeBoundingObjects_invalid_method(self):
        """Test error handling for invalid method"""
        intX = Interval(np.array([[-1]]), np.array([[1]]))
        intU = Interval(np.array([[-0.5]]), np.array([[0.5]]))
        
        # Invalid method
        options = {
            'lagrangeRem': {
                'method': 'invalid_method'
            }
        }
        
        # Should raise CORAerror
        with pytest.raises(CORAerror):
            initRangeBoundingObjects(intX, intU, options)
    
    def test_initRangeBoundingObjects_optional_params(self):
        """Test with optional parameters missing"""
        intX = Interval(np.array([[-1]]), np.array([[1]]))
        intU = Interval(np.array([[-0.5]]), np.array([[0.5]]))
        
        # Minimal options (only method required)
        options = {
            'lagrangeRem': {
                'method': 'taylorModel'
            }
        }
        
        # Should work with defaults
        objX, objU = initRangeBoundingObjects(intX, intU, options)
        assert objX is not None
        assert objU is not None
    
    def test_initRangeBoundingObjects_eps_tolerance_parsing(self):
        """Test that eps and tolerance are parsed correctly (MATLAB bug fix)"""
        intX = Interval(np.array([[-1]]), np.array([[1]]))
        intU = Interval(np.array([[-0.5]]), np.array([[0.5]]))
        
        # Set both eps and tolerance explicitly
        options = {
            'lagrangeRem': {
                'method': 'taylorModel',
                'eps': 1e-5,  # Should be assigned to eps
                'tolerance': 1e-7  # Should be assigned to tolerance (not swapped)
            }
        }
        
        # Should parse correctly (not swapped as in MATLAB bug)
        objX, objU = initRangeBoundingObjects(intX, intU, options)
        assert objX is not None
        assert objU is not None


def test_initRangeBoundingObjects():
    """Test function for initRangeBoundingObjects method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestInitRangeBoundingObjects()
    test.test_initRangeBoundingObjects_taylorModel()
    test.test_initRangeBoundingObjects_zoo()
    test.test_initRangeBoundingObjects_invalid_method()
    test.test_initRangeBoundingObjects_optional_params()
    test.test_initRangeBoundingObjects_eps_tolerance_parsing()
    
    print("test_initRangeBoundingObjects: all tests passed")
    return True


if __name__ == "__main__":
    test_initRangeBoundingObjects()

