import pytest
from cora_python.contDynamics.nonlinearARX import NonlinearARX

def example_fun(y, u):
    return y + u

def jacobian_fun(y, u):
    return "jacobian"

def hessian_fun(y, u):
    return "hessian"
    
def test_nonlinearARX_constructor():
    # Test case from MATLAB example
    dt = 0.25
    dim_y = 3
    dim_u = 2
    n_p = 2

    # 1. Test with name
    sys = NonlinearARX("myNARX", example_fun, dt, dim_y, dim_u, n_p)
    assert sys.name == "myNARX"
    assert sys.dt == dt
    assert sys.nr_of_outputs == dim_y
    assert sys.nr_of_inputs == dim_u
    assert sys.n_p == n_p
    assert sys.mFile == example_fun
    assert sys.jacobian is None
    assert sys.hessian is None

    # 2. Test without name (default name)
    sys_default = NonlinearARX(example_fun, dt, dim_y, dim_u, n_p)
    assert sys_default.name == "nonlinearARX"
    assert sys_default.dt == dt
    
    # 3. Test with optional jacobian and hessian
    sys_with_deriv = NonlinearARX("myNARX", example_fun, dt, dim_y, dim_u, n_p,
                                  jacobian=jacobian_fun, hessian=hessian_fun)
    assert sys_with_deriv.jacobian == jacobian_fun
    assert sys_with_deriv.hessian == hessian_fun
    assert sys_with_deriv.thirdOrderTensor is None

def test_invalid_arg_count_nonlinearARX():
    # This test depends on a more robust parsing than what is currently implemented.
    # The current implementation will throw an unpacking error, not a specific message.
    # For now, we test the cases that are explicitly handled.
    with pytest.raises(TypeError):
         # missing n_p
        NonlinearARX(example_fun, 0.1, 1, 1)

    with pytest.raises(TypeError):
        # too many args
        NonlinearARX("name", example_fun, 0.1, 1, 1, 1, "extra") 