"""
test_unitvector - unit test function for instantiation of the standard
   unit vector

Syntax:
    pytest cora_python/tests/g/functions/matlab/init/test_unitvector.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner
Written:       27-June-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.g.functions.matlab.init.unitvector import unitvector


class TestUnitvector:
    """Test class for unitvector functionality"""
    
    def test_unitvector_empty(self):
        """Test empty cases"""
        # MATLAB: assert(isempty(unitvector(0,0)));
        result = unitvector(0, 0)
        assert result.size == 0 or len(result) == 0
        
        # MATLAB: assert(isempty(unitvector(1,0)));% index is ignored for n=0
        result = unitvector(1, 0)
        assert result.size == 0 or len(result) == 0
    
    def test_unitvector_simple_cases(self):
        """Test simple cases"""
        # MATLAB: assert(isequal(unitvector(1,1),1));
        result = unitvector(1, 1)
        assert np.array_equal(result, np.array([[1]]))
        
        # MATLAB: assert(isequal(unitvector(1,2),[1;0]));
        result = unitvector(1, 2)
        assert np.array_equal(result, np.array([[1], [0]]))
        
        # MATLAB: assert(isequal(unitvector(2,2),[0;1]));
        result = unitvector(2, 2)
        assert np.array_equal(result, np.array([[0], [1]]))
        
        # MATLAB: assert(isequal(unitvector(3,4),[0;0;1;0]));
        result = unitvector(3, 4)
        assert np.array_equal(result, np.array([[0], [0], [1], [0]]))
        
        # MATLAB: assert(isequal(unitvector(2,5),[0;1;0;0;0]));
        result = unitvector(2, 5)
        assert np.array_equal(result, np.array([[0], [1], [0], [0], [0]]))
    
    def test_unitvector_identity(self):
        """Test that unit vectors form identity matrix"""
        # MATLAB: assert(isequal(eye(3),[unitvector(1,3),unitvector(2,3),unitvector(3,3)]));
        eye3 = np.eye(3)
        u1 = unitvector(1, 3)
        u2 = unitvector(2, 3)
        u3 = unitvector(3, 3)
        # Concatenate column vectors horizontally
        result = np.hstack([u1, u2, u3])
        assert np.array_equal(result, eye3)


def test_unitvector():
    """Test function for unitvector method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestUnitvector()
    test.test_unitvector_empty()
    test.test_unitvector_simple_cases()
    test.test_unitvector_identity()
    
    print("test_unitvector: all tests passed")
    return True


if __name__ == "__main__":
    test_unitvector()
