import sympy
import numpy as np
from cora_python.g.functions.matlab.function_handle.input_args_length import input_args_length
import inspect

def is_func_linear(f, input_args_sizes=None):
    """
    Checks if a function handle is linear in its arguments.
    """
    if input_args_sizes is None:
        try:
            input_args_sizes, _ = input_args_length(f)
        except Exception as e:
            print(f"Warning: Could not determine input arg sizes to check linearity: {e}. Assuming non-linear.")
            return False

    # Prepare symbolic variables only for inputs that have dimensions
    f_args = []
    sym_vars_for_diff = []
    
    # Create a list of symbolic variables for differentiation
    symbol_idx = 0
    for size in input_args_sizes:
        arg_symbols = []
        if size > 0:
            for _ in range(size):
                s = sympy.Symbol(f'z_{symbol_idx}')
                arg_symbols.append(s)
                sym_vars_for_diff.append(s)
                symbol_idx += 1
        f_args.append(np.array(arg_symbols))

    if not sym_vars_for_diff:
        # No variables means it's a constant function, which is linear
        return True

    # Evaluate function symbolically
    f_sym = np.array(f(*f_args))
        
    # Check if Hessian is zero for each component of the output
    for expr in np.ravel(f_sym):
        for var1 in sym_vars_for_diff:
            for var2 in sym_vars_for_diff:
                if sympy.diff(expr, var1, var2) != 0:
                    return False
    return True 