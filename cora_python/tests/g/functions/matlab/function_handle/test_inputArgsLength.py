"""
test_inputArgsLength - unit test function for automated read out of
   number of input arguments to a function handle

Syntax:
    pytest cora_python/tests/g/functions/matlab/function_handle/test_inputArgsLength.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       20-November-2022
Last update:   19-January-2024 (MW, test reported bug)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.g.functions.matlab.function_handle.inputArgsLength import inputArgsLength


class TestInputArgsLength:
    """Test class for inputArgsLength functionality"""
    
    def test_inputArgsLength_empty(self):
        """Test empty function handle"""
        # MATLAB: f = @(x,u) [];
        f = lambda x, u: np.array([])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); m = inputArgs(2);
        n = inputArgs[0]
        m = inputArgs[1] if len(inputArgs) > 1 else 0
        # MATLAB: assert(n == 0);
        assert n == 0
        # MATLAB: assert(m == 0);
        assert m == 0
        # MATLAB: assert(r == 0);
        assert r == 0
    
    def test_inputArgsLength_one_input(self):
        """Test one input variable: x"""
        # n = 1, r = 1
        # MATLAB: f = @(x) x(1)^2;
        f = lambda x: np.array([x[0]**2])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: assert(length(inputArgs) == 1 && n == 1 && r == 1);
        assert len(inputArgs) == 1 and n == 1 and r == 1
        
        # n = 2, r = 1
        # MATLAB: f = @(x) x(1)^2 + x(2) - 4;
        f = lambda x: np.array([x[0]**2 + x[1] - 4])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: assert(length(inputArgs) == 1 && n == 2 && r == 1);
        assert len(inputArgs) == 1 and n == 2 and r == 1
        
        # n = 5, r = 3
        # MATLAB: f = @(x) [x(1)^2 + x(5)/5; 2*x(2) + x(3); x(4)^3];
        f = lambda x: np.array([x[0]**2 + x[4]/5, 2*x[1] + x[2], x[3]**3])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: assert(length(inputArgs) == 1 && n == 5 && r == 3);
        assert len(inputArgs) == 1 and n == 5 and r == 3
        
        # n = 5, r = 3 (but not all x are used)
        # MATLAB: f = @(x) [x(1)^2 + x(5)/3; x(2)*2; 1-x(4)^3];
        f = lambda x: np.array([x[0]**2 + x[4]/3, x[1]*2, 1 - x[3]**3])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: assert(length(inputArgs) == 1 && n == 5 && r == 3);
        assert len(inputArgs) == 1 and n == 5 and r == 3
        
        # n = 4, r = 2 (matrix multiplication with indexing)
        # MATLAB: f = @(x) randn(2,4) * x(1:4);
        # Note: We use a fixed matrix for reproducibility
        M = np.random.RandomState(42).randn(2, 4)
        f = lambda x: M @ x[:4]
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: assert(length(inputArgs) == 1 && n == 4 && r == 2);
        assert len(inputArgs) == 1 and n == 4 and r == 2
        
        # n = 5, r = 2 (matrix multiplication with zeros)
        # MATLAB: M = [0 1 0 0 0; 0 0 1 0 0];
        M = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        # MATLAB: f = @(x) M * x(1:5);
        f = lambda x: M @ x[:5]
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: assert(length(inputArgs) == 1 && n == 5 && r == 2);
        assert len(inputArgs) == 1 and n == 5 and r == 2
        
        # n = 4, r = 2 (matrix multiplication without indexing)
        # MATLAB: f = @(x) randn(2,4) * x;
        M = np.random.RandomState(42).randn(2, 4)
        f = lambda x: M @ x
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: assert(length(inputArgs) == 1 && n == 4 && r == 2);
        assert len(inputArgs) == 1 and n == 4 and r == 2
        
        # n = 4, r = 2 (matrix addition without indexing)
        # MATLAB: f = @(x) randn(2,1) + x;
        # Note: MATLAB comment says this will most likely fail because ambiguous
        # MATLAB: assert(length(inputArgs) == 1 && ~(n == 2 && r == 2));
        M = np.random.RandomState(42).randn(2, 1)
        f = lambda x: M + x
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1);
        n = inputArgs[0]
        # MATLAB: % This will most likely fail, because the function definition is ambiguous;
        # MATLAB: % should x be a vector? A scalar added component-wise?
        # MATLAB: % For now, we negate this test; if you see that it gives you 'false' here,
        # MATLAB: % then either you did something wrong, or you found a better way to deal
        # MATLAB: % with this problem
        # MATLAB: assert(length(inputArgs) == 1 && ~(n == 2 && r == 2));
        assert len(inputArgs) == 1 and not (n == 2 and r == 2)
    
    def test_inputArgsLength_two_inputs(self):
        """Test two input variables: x, u"""
        # n = 1, m = 1, r = 1
        # MATLAB: f = @(x,u) x(1)^2 + u(1);
        f = lambda x, u: np.array([x[0]**2 + u[0]])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); m = inputArgs(2);
        n = inputArgs[0]
        m = inputArgs[1]
        # MATLAB: assert(length(inputArgs) == 2 && n == 1 && m == 1 && r == 1);
        assert len(inputArgs) == 2 and n == 1 and m == 1 and r == 1
        
        # n = 6, m = 4, r = 2
        # MATLAB: f = @(x,u) [x(1)^2 + u(1); u(4)^2 - x(6)];
        f = lambda x, u: np.array([x[0]**2 + u[0], u[3]**2 - x[5]])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); m = inputArgs(2);
        n = inputArgs[0]
        m = inputArgs[1]
        # MATLAB: assert(length(inputArgs) == 2 && n == 6 && m == 4 && r == 2);
        assert len(inputArgs) == 2 and n == 6 and m == 4 and r == 2
        
        # n = 4, m = 2, r = 2 (matrix multiplication with indexing)
        # MATLAB: f = @(x,u) randn(2,4) * x(1:4) + randn(2,2) * u(1:2);
        M1 = np.random.RandomState(42).randn(2, 4)
        M2 = np.random.RandomState(43).randn(2, 2)
        f = lambda x, u: M1 @ x[:4] + M2 @ u[:2]
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); m = inputArgs(2);
        n = inputArgs[0]
        m = inputArgs[1]
        # MATLAB: assert(length(inputArgs) == 2 && n == 4 && m == 2 && r == 2);
        assert len(inputArgs) == 2 and n == 4 and m == 2 and r == 2
        
        # n = 4, r = 2 (matrix multiplication without indexing)
        # MATLAB: f = @(x,u) randn(2,4) * x + randn(2,2) * u;
        M1 = np.random.RandomState(42).randn(2, 4)
        M2 = np.random.RandomState(43).randn(2, 2)
        f = lambda x, u: M1 @ x + M2 @ u
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); m = inputArgs(2);
        n = inputArgs[0]
        m = inputArgs[1]
        # MATLAB: assert(length(inputArgs) == 2 && n == 4 && m == 2 && r == 2);
        assert len(inputArgs) == 2 and n == 4 and m == 2 and r == 2
        
        # n = 5, m = 2, r = 2 (matrix multiplication with zeros)
        # MATLAB: M = [0 1 0 0 0; 0 0 1 0 0];
        M = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        # MATLAB: N = [1 0; 0 0];
        N = np.array([[1, 0], [0, 0]])
        # MATLAB: f = @(x,u) M * x(1:5) + N * u(1:2);
        f = lambda x, u: M @ x[:5] + N @ u[:2]
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); m = inputArgs(2);
        n = inputArgs[0]
        m = inputArgs[1]
        # MATLAB: assert(length(inputArgs) == 2 && n == 5 && m == 2 && r == 2);
        assert len(inputArgs) == 2 and n == 5 and m == 2 and r == 2
        
        # n = 2, m = 0, r = 2
        # MATLAB: f = @(x,u) [x(1)^2 - x(2), x(2)];
        f = lambda x, u: np.array([x[0]**2 - x[1], x[1]])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); m = inputArgs(2);
        n = inputArgs[0]
        m = inputArgs[1]
        # MATLAB: assert(length(inputArgs) == 2 && n == 2 && m == 0 && r == 2);
        assert len(inputArgs) == 2 and n == 2 and m == 0 and r == 2
    
    def test_inputArgsLength_three_inputs(self):
        """Test three input variables: x, y, u"""
        # n = 3, m1 = 2, m2 = 1, r = 2
        # MATLAB: f = @(x,y,u) [x(1)^2 + y(1) - x(2), x(2)*x(3) + u(1) - y(2)^2];
        f = lambda x, y, u: np.array([x[0]**2 + y[0] - x[1], x[1]*x[2] + u[0] - y[1]**2])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); y = inputArgs(2); m = inputArgs(3);
        n = inputArgs[0]
        y_dim = inputArgs[1]
        m = inputArgs[2]
        # MATLAB: assert(length(inputArgs) == 3 && n == 3 && y == 2 && m == 1 && r == 2);
        assert len(inputArgs) == 3 and n == 3 and y_dim == 2 and m == 1 and r == 2
        
        # n = 2, m1 = 0, m2 = 1, r = 2
        # MATLAB: f = @(x,y,u) [x(1)^2 - x(2), x(2) + u(1)];
        f = lambda x, y, u: np.array([x[0]**2 - x[1], x[1] + u[0]])
        # MATLAB: [inputArgs,r] = inputArgsLength(f);
        inputArgs, r = inputArgsLength(f)
        # MATLAB: n = inputArgs(1); y = inputArgs(2); m = inputArgs(3);
        n = inputArgs[0]
        y_dim = inputArgs[1]
        m = inputArgs[2]
        # MATLAB: assert(length(inputArgs) == 3 && n == 2 && y == 0 && m == 1 && r == 2);
        assert len(inputArgs) == 3 and n == 2 and y_dim == 0 and m == 1 and r == 2
        
        # n = 6, m = 2, o = 1
        # MATLAB: f = @(x,u,w) [x(4)*cos(x(3)) - x(5)*sin(x(3));
        #              x(4)*sin(x(3)) + x(5)*cos(x(3));
        #              x(6);
        #              x(5)*x(6) - sin(x(3)) + cos(x(3))*w(1);
        #              -x(5)*x(6) - cos(x(3)) + u(1) + u(2) - sin(x(3))*w(1);
        #              u(1) - u(2)];
        f = lambda x, u, w: np.array([
            x[3]*np.cos(x[2]) - x[4]*np.sin(x[2]),
            x[3]*np.sin(x[2]) + x[4]*np.cos(x[2]),
            x[5],
            x[4]*x[5] - np.sin(x[2]) + np.cos(x[2])*w[0],
            -x[4]*x[5] - np.cos(x[2]) + u[0] + u[1] - np.sin(x[2])*w[0],
            u[0] - u[1]
        ])
        # MATLAB: inputArgs = inputArgsLength(f,3);
        inputArgs = inputArgsLength(f, 3)
        # MATLAB: n = inputArgs(1); m = inputArgs(2); o = inputArgs(3);
        n = inputArgs[0]
        m = inputArgs[1]
        o = inputArgs[2]
        # MATLAB: assert(length(inputArgs) == 3 && n == 6 && m == 2 && o == 1);
        assert len(inputArgs) == 3 and n == 6 and m == 2 and o == 1


def test_inputArgsLength():
    """Test function for inputArgsLength method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestInputArgsLength()
    test.test_inputArgsLength_empty()
    test.test_inputArgsLength_one_input()
    test.test_inputArgsLength_two_inputs()
    test.test_inputArgsLength_three_inputs()
    
    print("test_inputArgsLength: all tests passed")
    return True


if __name__ == "__main__":
    test_inputArgsLength()

