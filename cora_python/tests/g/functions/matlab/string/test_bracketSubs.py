"""
test_bracketSubs - unit test function for substitution of brackets with
   letters 'L' and 'R'

Syntax:
    pytest cora_python/tests/g/functions/matlab/string/test_bracketSubs.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       28-April-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import pytest
from cora_python.g.functions.matlab.string.bracketSubs import bracketSubs


class TestBracketSubs:
    """Test class for bracketSubs functionality"""
    
    def test_bracketSubs_init_state(self):
        """Test init state string"""
        # MATLAB: str = 'xL1R';
        str_val = 'xL1R'
        # MATLAB: str_ = bracketSubs(str);
        str_result = bracketSubs(str_val)
        # MATLAB: str_true = 'x(1)';
        str_true = 'x(1)'
        # MATLAB: assert(strcmp(str_,str_true));
        assert str_result == str_true
    
    def test_bracketSubs_input_string(self):
        """Test input string"""
        # MATLAB: str = 'uL16R';
        str_val = 'uL16R'
        # MATLAB: str_ = bracketSubs(str);
        str_result = bracketSubs(str_val)
        # MATLAB: str_true = 'u(16)';
        str_true = 'u(16)'
        # MATLAB: assert(strcmp(str_,str_true));
        assert str_result == str_true
    
    def test_bracketSubs_longer_name(self):
        """Test longer name"""
        # MATLAB: str = 'abcdeL99R';
        str_val = 'abcdeL99R'
        # MATLAB: str_ = bracketSubs(str);
        str_result = bracketSubs(str_val)
        # MATLAB: str_true = 'abcde(99)';
        str_true = 'abcde(99)'
        # MATLAB: assert(strcmp(str_,str_true));
        assert str_result == str_true


def test_bracketSubs():
    """Test function for bracketSubs method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestBracketSubs()
    test.test_bracketSubs_init_state()
    test.test_bracketSubs_input_string()
    test.test_bracketSubs_longer_name()
    
    print("test_bracketSubs: all tests passed")


if __name__ == "__main__":
    test_bracketSubs()

