"""
matlabFunction - Python equivalent of MATLAB's matlabFunction
Generates Python code from symbolic expressions

This is a simplified version that generates Python function files
from symbolic expressions, similar to MATLAB's matlabFunction.
"""

import sympy as sp
import os
from typing import List, Any, Optional
from cora_python.g.functions.verbose.write.writeMatrix import _sympy_expr_to_python


def matlabFunction(expr: Any, File: Optional[str] = None, Vars: Optional[List[Any]] = None) -> str:
    """
    Generate Python code from symbolic expression (equivalent to MATLAB's matlabFunction)
    
    Args:
        expr: symbolic expression or list of expressions
        File: optional file path to write the function to
        Vars: list of symbolic variables (in order of function arguments)
        
    Returns:
        Python code string
    """
    
    # Convert expr to list if single expression
    if not isinstance(expr, (list, tuple)):
        expr = [expr]
    
    # Get variable names
    if Vars is None:
        # Extract variables from expression
        all_vars = set()
        for e in expr:
            if hasattr(e, 'free_symbols'):
                all_vars.update(e.free_symbols)
        Vars = sorted(all_vars, key=str)
    
    # Generate function signature
    var_names = [str(v) for v in Vars]
    func_name = 'fun'  # Default function name
    
    # Generate function code
    code_lines = []
    code_lines.append('import numpy as np\n')
    code_lines.append('import sympy as sp\n')
    code_lines.append('\n')
    
    # Function signature
    if len(expr) == 1:
        code_lines.append(f'def {func_name}({", ".join(var_names)}):\n')
        # Convert expression to Python code
        expr_code = _sympy_expr_to_python(expr[0])
        code_lines.append(f'    return {expr_code}\n')
    else:
        code_lines.append(f'def {func_name}({", ".join(var_names)}):\n')
        # Multiple outputs
        for i, e in enumerate(expr):
            expr_code = _sympy_expr_to_python(e)
            code_lines.append(f'    out{i+1} = {expr_code}\n')
        outputs = ', '.join([f'out{i+1}' for i in range(len(expr))])
        code_lines.append(f'    return {outputs}\n')
    
    code = '\n'.join(code_lines)
    
    # Write to file if specified
    if File is not None:
        # Remove .m extension if present, add .py
        if File.endswith('.m'):
            File = File[:-2] + '.py'
        elif not File.endswith('.py'):
            File = File + '.py'
        
        with open(File, 'w') as f:
            f.write(code)
    
    return code

