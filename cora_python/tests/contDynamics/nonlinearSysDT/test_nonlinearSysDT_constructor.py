import pytest
import numpy as np
import sympy
from cora_python.contDynamics.nonlinearSysDT import NonlinearSysDT

# Helper function for tests
def dyn_fun(x, u):
    # Make it robust to the dimension of u
    u_len = u.shape[0] if isinstance(u, sympy.Matrix) else u.size
    u_comp = u[0] if u_len > 0 else 0
    return np.array([x[0] + u_comp, x[1] - u_comp])

def out_fun_linear(x, u):
    # C = [1, 1]
    # u is ignored
    return np.array([x[0] + x[1]])

def out_fun_linear_sympy(x, u):
    # This version is needed for the symbolic engine to work properly
    return sympy.Matrix([x[0] + x[1]])

def out_fun_nonlinear(x, u):
    return np.array([x[0]*x[1]])

def test_constructor_autodetection():
    # Test syntax: nonlinearSysDT(fun, dt)
    # This should now work without explicitly providing dimensions
    sys = NonlinearSysDT(dyn_fun, 0.1)
    assert sys.name == 'dyn_fun'
    assert sys.dt == 0.1
    assert sys.nr_of_dims == 2  # inferred from dyn_fun
    assert sys.nr_of_inputs == 1 # inferred from dyn_fun

def test_constructor_with_name():
    # Test syntax: nonlinearSysDT(name, fun, dt)
    sys = NonlinearSysDT("mySys", dyn_fun, 0.1)
    assert sys.name == 'mySys'
    assert sys.nr_of_dims == 2
    assert sys.nr_of_inputs == 1

def test_constructor_with_output_fun_auto():
    # Test syntax: nonlinearSysDT(fun, dt, out_fun)
    # Linearity and output dims should be auto-detected
    sys_nl = NonlinearSysDT(dyn_fun, 0.1, out_fun=out_fun_nonlinear)
    assert sys_nl.nr_of_outputs == 1
    assert sys_nl.isLinear == False # dyn_fun is linear, but out_fun makes it non-linear
    
    sys_l = NonlinearSysDT(dyn_fun, 0.1, out_fun=out_fun_linear)
    assert sys_l.nr_of_outputs == 1
    assert sys_l.isLinear == True
    assert sys_l.out_isLinear is True

def test_constructor_explicit_dims():
    # Verify that providing explicit dimensions still works
    sys = NonlinearSysDT(dyn_fun, 0.1, states=2, inputs=1, out_fun=out_fun_nonlinear, outputs=1)
    assert sys.nr_of_dims == 2
    assert sys.nr_of_inputs == 1
    assert sys.nr_of_outputs == 1
    assert sys.isLinear == False

def test_constructor_kwargs():
    def jac(x,u): return "jacobian"
    sys = NonlinearSysDT("mySys", dyn_fun, 0.1, jacobian=jac)
    assert sys.jacobian == jac
    assert sys.nr_of_dims == 2 # still auto-detected
    assert sys.nr_of_inputs == 1

def test_parser():
    # test a few of the many syntaxes
    # (fun, dt, states, inputs)
    sys = NonlinearSysDT(dyn_fun, 0.1, states=2, inputs=1)
    assert sys.nr_of_dims == 2
    assert sys.nr_of_inputs == 1

    # (name, fun, dt, out_fun)
    sys = NonlinearSysDT("anotherSys", dyn_fun, 0.2, out_fun=out_fun_linear)
    assert sys.name == "anotherSys"
    assert sys.nr_of_dims == 2
    assert sys.nr_of_outputs == 1
    assert sys.out_isLinear == True

    
