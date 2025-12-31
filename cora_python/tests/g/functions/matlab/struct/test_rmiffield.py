"""
test_rmiffield - unit test function for rmiffield

Syntax:
    pytest cora_python/tests/g/functions/matlab/struct/test_rmiffield.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner
Written:       17-October-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import pytest
from cora_python.g.functions.matlab.struct.rmiffield import rmiffield


class TestRmiffield:
    """Test class for rmiffield functionality"""
    
    def test_rmiffield_empty(self):
        """Test empty case"""
        # MATLAB: S = struct();
        S = {}
        # MATLAB: label = 'test';
        label = 'test'
        # MATLAB: S = rmiffield(S,label);
        S = rmiffield(S, label)
        # MATLAB: assert(~isfield(S,label));
        assert label not in S
    
    def test_rmiffield_normal(self):
        """Test normal case"""
        # MATLAB: S = struct();
        S = {}
        # MATLAB: label1 = 'label1';
        label1 = 'label1'
        # MATLAB: label2 = 'label2';
        label2 = 'label2'
        # MATLAB: S.(label1) = 'value1';
        S[label1] = 'value1'
        # MATLAB: S.(label2) = 'value2';
        S[label2] = 'value2'
        # MATLAB: assert(isfield(S,label1));
        assert label1 in S
        # MATLAB: assert(isfield(S,label2));
        assert label2 in S
        # MATLAB: S = rmiffield(S,label1);
        S = rmiffield(S, label1)
        # MATLAB: assert(~isfield(S,label1));
        assert label1 not in S
        # MATLAB: assert(isfield(S,label2));
        assert label2 in S
    
    def test_rmiffield_repeated_call(self):
        """Test that another call should not change the result"""
        S = {}
        label1 = 'label1'
        label2 = 'label2'
        S[label1] = 'value1'
        S[label2] = 'value2'
        S = rmiffield(S, label1)
        # MATLAB: % another call should not change the result
        # MATLAB: S = rmiffield(S,label1);
        S = rmiffield(S, label1)
        # MATLAB: assert(~isfield(S,label1));
        assert label1 not in S
        # MATLAB: assert(isfield(S,label2));
        assert label2 in S


def test_rmiffield():
    """Test function for rmiffield method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestRmiffield()
    test.test_rmiffield_empty()
    test.test_rmiffield_normal()
    test.test_rmiffield_repeated_call()
    
    print("test_rmiffield: all tests passed")
    return True


if __name__ == "__main__":
    test_rmiffield()

