import inspect
import sympy
import numpy as np

def input_args_length(f):
    """
    Computes the number and required vector length of inputs for a function handle
    using symbolic evaluation.

    Args:
        f (callable): The function handle.

    Returns:
        tuple: A tuple containing:
            - count (list): A list with the required length of each input argument.
            - out_dim (tuple): The dimensions of the function's output.
    """
    try:
        sig = inspect.signature(f)
        nargin_f = len(sig.parameters)
    except (TypeError, ValueError):
        # Fallback for built-ins, assume 2 args for (x, u)
        nargin_f = 2

    if nargin_f == 0:
        return [], (0,)

    # Create symbolic variables for each input argument
    max_vars = 100
    sym_args = [sympy.symbols(f'x_{i}_1:{max_vars+1}') for i in range(nargin_f)]
    
    # Create numpy arrays of sympy symbols. This allows symbols to flow
    # through standard numpy operations within the function f.
    f_args = [np.array(arg) for arg in sym_args]

    # Evaluate the function symbolically
    try:
        fsym = f(*f_args)
    except Exception as e:
        raise RuntimeError(f"Could not symbolically evaluate the function '{f.__name__}': {e}")

    # Get all unique symbols used in the output expression
    if isinstance(fsym, (np.ndarray, list, tuple)):
        vars_used = set().union(*(expr.free_symbols for expr in np.ravel(fsym) if hasattr(expr, 'free_symbols')))
    elif hasattr(fsym, 'free_symbols'):
        vars_used = fsym.free_symbols
    else:
        vars_used = set()

    # Determine the required length for each input argument's vector
    counts = []
    for i in range(nargin_f):
        max_idx = 0
        for var in vars_used:
            # check if a symbol from the output belongs to the i-th input arg tuple
            if var in sym_args[i]:
                # symbols are named 'x_i_j', so j is the 1-based index
                idx = int(str(var).split('_')[-1])
                if idx > max_idx:
                    max_idx = idx
        counts.append(max_idx)
    
    out_dim = np.array(fsym).shape

    return counts, out_dim 