"""
test_nonlinearSys_evalNthTensor - unit-test for the tensor generation and evaluation

The solution from the evaluation of and generation with the toolbox
functions "generateNthTensor" and "evalNthTensor" is compared to the
evaluation using the corresponding closed-expression equation

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_evalNthTensor.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Niklas Kochdumper
Written:       08-February-2018
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import sympy as sp
from cora_python.g.functions.helper.dynamics.contDynamics.contDynamics.generateNthTensor import generateNthTensor
from cora_python.g.functions.helper.dynamics.contDynamics.contDynamics.evalNthTensor import evalNthTensor
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestNonlinearSysEvalNthTensor:
    """Test class for evalNthTensor functionality"""
    
    def test_nonlinearSys_evalNthTensor(self):
        """Test tensor generation and evaluation for 2D example"""
        # assume true
        # MATLAB: res = true;
        res = True
        
        # 2D-example --------------------------------------------------------------
        # MATLAB: syms x y
        x, y = sp.symbols('x y', real=True)
        # MATLAB: f = sin(x)*cos(y+x)*exp(x*y);
        f = sp.sin(x) * sp.cos(y + x) * sp.exp(x * y)
        
        # MATLAB: N = 1;
        N = 1
        # MATLAB: val = rand(2,10);
        np.random.seed(42)  # For reproducibility
        val = np.random.rand(2, 10)
        
        # compute derivatives
        # MATLAB: df_x = diff(f,x);
        df_x = sp.diff(f, x)
        # MATLAB: df_y = diff(f,y);
        df_y = sp.diff(f, y)
        
        # MATLAB: df_xx = diff(df_x,x);
        df_xx = sp.diff(df_x, x)
        # MATLAB: df_xy = diff(df_x,y);
        df_xy = sp.diff(df_x, y)
        # MATLAB: df_yy = diff(df_y,y);
        df_yy = sp.diff(df_y, y)
        
        # MATLAB: df_xxx = diff(df_xx,x);
        df_xxx = sp.diff(df_xx, x)
        # MATLAB: df_xxy = diff(df_xx,y);
        df_xxy = sp.diff(df_xx, y)
        # MATLAB: df_xyy = diff(df_xy,y);
        df_xyy = sp.diff(df_xy, y)
        # MATLAB: df_yyy = diff(df_yy,y);
        df_yyy = sp.diff(df_yy, y)
        
        # MATLAB: df_xxxx = diff(df_xxx,x);
        df_xxxx = sp.diff(df_xxx, x)
        # MATLAB: df_xxxy = diff(df_xxx,y);
        df_xxxy = sp.diff(df_xxx, y)
        # MATLAB: df_xxyy = diff(df_xxy,y);
        df_xxyy = sp.diff(df_xxy, y)
        # MATLAB: df_xyyy = diff(df_xyy,y);
        df_xyyy = sp.diff(df_xyy, y)
        # MATLAB: df_yyyy = diff(df_yyy,y);
        df_yyyy = sp.diff(df_yyy, y)
        
        # evaluate function with formula
        # MATLAB: res_real = zeros(N,1);
        res_real = np.zeros(N)
        
        # MATLAB: for i = 1:N
        for i in range(N):
            # MATLAB: p = val(:,i);
            p = val[:, i]
            
            # MATLAB: first = eval(subs(df_x,[x;y],p)) * p(1) + eval(subs(df_y,[x;y],p)) * p(2);
            first = float(df_x.subs([(x, p[0]), (y, p[1])])) * p[0] + \
                    float(df_y.subs([(x, p[0]), (y, p[1])])) * p[1]
            
            # MATLAB: second = 0.5 * eval(subs(df_xx,[x;y],p)) * p(1)^2 + ...
            #            eval(subs(df_xy,[x;y],p)) * p(1) * p(2) + ...
            #            0.5 * eval(subs(df_yy,[x;y],p)) * p(2)^2;
            second = 0.5 * float(df_xx.subs([(x, p[0]), (y, p[1])])) * p[0]**2 + \
                     float(df_xy.subs([(x, p[0]), (y, p[1])])) * p[0] * p[1] + \
                     0.5 * float(df_yy.subs([(x, p[0]), (y, p[1])])) * p[1]**2
            
            # MATLAB: third = 1/6 * eval(subs(df_xxx,[x;y],p)) * p(1)^3 + ...
            #           0.5 * eval(subs(df_xxy,[x;y],p)) * p(1)^2 * p(2) + ...
            #           0.5 * eval(subs(df_xyy,[x;y],p)) * p(1) * p(2)^2 + ...
            #           1/6 * eval(subs(df_yyy,[x;y],p)) * p(2)^3;
            third = (1/6) * float(df_xxx.subs([(x, p[0]), (y, p[1])])) * p[0]**3 + \
                   0.5 * float(df_xxy.subs([(x, p[0]), (y, p[1])])) * p[0]**2 * p[1] + \
                   0.5 * float(df_xyy.subs([(x, p[0]), (y, p[1])])) * p[0] * p[1]**2 + \
                   (1/6) * float(df_yyy.subs([(x, p[0]), (y, p[1])])) * p[1]**3
            
            # MATLAB: fourth = 1/24 * eval(subs(df_xxxx,[x;y],p)) * p(1)^4 + ...
            #            1/6 * eval(subs(df_xxxy,[x;y],p)) * p(1)^3 * p(2) + ...
            #            1/4 * eval(subs(df_xxyy,[x;y],p)) * p(1)^2 * p(2)^2 + ...
            #            1/6 * eval(subs(df_xyyy,[x;y],p)) * p(1) * p(2)^3 + ...
            #            1/24 * eval(subs(df_yyyy,[x;y],p)) * p(2)^4;
            fourth = (1/24) * float(df_xxxx.subs([(x, p[0]), (y, p[1])])) * p[0]**4 + \
                    (1/6) * float(df_xxxy.subs([(x, p[0]), (y, p[1])])) * p[0]**3 * p[1] + \
                    (1/4) * float(df_xxyy.subs([(x, p[0]), (y, p[1])])) * p[0]**2 * p[1]**2 + \
                    (1/6) * float(df_xyyy.subs([(x, p[0]), (y, p[1])])) * p[0] * p[1]**3 + \
                    (1/24) * float(df_yyyy.subs([(x, p[0]), (y, p[1])])) * p[1]**4
            
            # MATLAB: res_real(i) = first + second + third + fourth;
            res_real[i] = first + second + third + fourth
        
        # evaluate function with method "evalNthTensor"
        # MATLAB: T = cell(3,1);
        # MATLAB: T{1} = generateNthTensor(f,[x;y],1);
        # MATLAB: T{2} = generateNthTensor(f,[x;y],2);
        # MATLAB: T{3} = generateNthTensor(f,[x;y],3);
        # MATLAB: T{4} = generateNthTensor(f,[x;y],4);
        vars_list = [x, y]
        T = [
            generateNthTensor(f, vars_list, 1),
            generateNthTensor(f, vars_list, 2),
            generateNthTensor(f, vars_list, 3),
            generateNthTensor(f, vars_list, 4)
        ]
        
        # MATLAB: first = evalNthTensor(T{1},[x;y],1);
        first_sym = evalNthTensor(T[0], vars_list, 1)
        # MATLAB: second = evalNthTensor(T{2},[x;y],2);
        second_sym = evalNthTensor(T[1], vars_list, 2)
        # MATLAB: third = evalNthTensor(T{3},[x;y],3);
        third_sym = evalNthTensor(T[2], vars_list, 3)
        # MATLAB: fourth = evalNthTensor(T{4},[x;y],4);
        fourth_sym = evalNthTensor(T[3], vars_list, 4)
        
        # MATLAB: res_test = zeros(N,1);
        res_test = np.zeros(N)
        
        # MATLAB: for i = 1:N
        for i in range(N):
            # MATLAB: p = val(:,i);
            p = val[:, i]
            
            # MATLAB: res_test(i) = eval(subs(first,[x;y],p)) + eval(subs(second,[x;y],p)) + ...
            #                  eval(subs(third,[x;y],p)) + eval(subs(fourth,[x;y],p));
            # Convert symbolic expressions to numeric values
            def _eval_expr(expr):
                expr_val = expr
                if isinstance(expr_val, np.ndarray):
                    expr_val = expr_val.reshape(-1)[0]
                if hasattr(expr_val, 'subs'):
                    expr_val = expr_val.subs([(x, p[0]), (y, p[1])])
                return float(sp.N(expr_val))

            first_val = _eval_expr(first_sym)
            second_val = _eval_expr(second_sym)
            third_val = _eval_expr(third_sym)
            fourth_val = _eval_expr(fourth_sym)
            
            res_test[i] = first_val + second_val + third_val + fourth_val
        
        # compare the results
        # MATLAB: assert(1);
        # MATLAB: for i = 1:N
        # MATLAB:     assertLoop(withinTol(res_test(i),res_real(i),1e-12),i)
        # MATLAB: end
        for i in range(N):
            assert withinTol(res_test[i], res_real[i], 1e-12), \
                f"Tensor evaluation mismatch at index {i}: res_test={res_test[i]}, res_real={res_real[i]}"


def test_nonlinearSys_evalNthTensor():
    """Test function for evalNthTensor method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysEvalNthTensor()
    test.test_nonlinearSys_evalNthTensor()
    
    print("test_nonlinearSys_evalNthTensor: all tests passed")
    return True


if __name__ == "__main__":
    test_nonlinearSys_evalNthTensor()

