"""
display - Displays a transition object

Syntax:
    display(trans)

Inputs:
    trans - transition object

Outputs:
    ---

Example:
    guard = polytope([],[],[1 0],0);
    reset = linearReset.eye(2);
    target = 1;
    transition(guard,reset,target)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff
Written:       20-September-2007
Last update:   18-June-2022 (MW, improved formatting, empty cases)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any
import numpy as np


def display_(trans: Any, var_name: str = None) -> str:
    """
    Display a transition object (internal function that returns string)
    
    Args:
        trans: Transition object
        var_name: Optional variable name (for display)
    
    Returns:
        String representation for display
    """
    output_lines = []
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    # MATLAB: disp([inputname(1), ' =']);
    if var_name is None:
        var_name = 'trans'
    output_lines.append(f"{var_name} =")
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    # MATLAB: if length(trans) > 1
    # For now, handle single objects (not arrays)
    # TODO: Handle arrays if needed
    
    # MATLAB: display guard set
    # if (isnumeric(trans.guard) && isempty(trans.guard)) ...
    #         || (~isnumeric(trans.guard) && ~representsa_(trans.guard,'emptySet',eps))
    guard_is_numeric_empty = isinstance(trans.guard, (int, float, np.number)) or \
                             (isinstance(trans.guard, np.ndarray) and trans.guard.size == 0)
    
    if guard_is_numeric_empty:
        output_lines.append("Guard set: (none)")
    else:
        # Check if guard represents empty set
        guard_is_empty_set = False
        if hasattr(trans.guard, 'representsa_'):
            try:
                guard_is_empty_set = trans.guard.representsa_('emptySet', np.finfo(float).eps)
            except:
                pass
        
        if guard_is_empty_set:
            output_lines.append("Guard set: (none)")
        else:
            output_lines.append("Guard set:")
            if hasattr(trans.guard, 'display'):
                guard_display = trans.guard.display()
                if isinstance(guard_display, str):
                    # Split into lines and indent
                    for line in guard_display.split('\n'):
                        if line.strip():
                            output_lines.append(f"  {line}")
                elif isinstance(guard_display, list):
                    for line in guard_display:
                        if line.strip():
                            output_lines.append(f"  {line}")
            else:
                output_lines.append(f"  {trans.guard}")
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    # MATLAB: display reset function
    # if isnumeric(trans.reset) && isempty(trans.reset)
    reset_is_numeric_empty = isinstance(trans.reset, (int, float, np.number)) or \
                             (isinstance(trans.reset, np.ndarray) and trans.reset.size == 0)
    
    if reset_is_numeric_empty:
        # MATLAB: no reset function (empty transition)
        output_lines.append("Reset function: (none)")
    else:
        # Check if reset is LinearReset
        from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
        if isinstance(trans.reset, LinearReset):
            # MATLAB: linear reset function
            output_lines.append("Reset function: Ax + Bu + c")
            
            # MATLAB: display state matrix
            output_lines.extend(_display_matrix_vector(trans.reset.A, "A"))
            
            # MATLAB: display input matrix
            output_lines.extend(_display_matrix_vector(trans.reset.B, "B"))
            
            # MATLAB: display constant offset
            output_lines.extend(_display_matrix_vector(trans.reset.c, "c"))
        else:
            # Check if reset is NonlinearReset (has nonlinearReset field or is NonlinearReset)
            from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
            if isinstance(trans.reset, NonlinearReset) or \
               (hasattr(trans.reset, 'nonlinearReset') and trans.reset.nonlinearReset):
                # MATLAB: nonlinear reset function
                output_lines.append("Reset function: nonlinear")
                
                # MATLAB: use try-block to avoid annoying error messages
                try:
                    # Create symbolic variables
                    from cora_python.contDynamics.contDynamics.symVariables import symVariables
                    from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
                    
                    sys_obj = NonlinearSys(trans.reset.f,
                                          trans.reset.preStateDim,
                                          trans.reset.inputDim)
                    vars_dict, _ = symVariables(sys_obj)
                    
                    # Insert symbolic variables into the system equations
                    f = trans.reset.f(vars_dict['x'], vars_dict['u'])
                    
                    # Display equations
                    if isinstance(f, (list, tuple, np.ndarray)):
                        for i, eq in enumerate(f):
                            output_lines.append(f"  f({i+1}) = {eq}")
                    else:
                        output_lines.append(f"  f(1) = {f}")
                    output_lines.append("")
                except:
                    # Silently fail if symbolic display fails
                    pass
    
    # MATLAB: display target
    # if isempty(trans.target)
    target_is_empty = isinstance(trans.target, np.ndarray) and trans.target.size == 0
    
    if target_is_empty:
        # MATLAB: empty transition object
        output_lines.append("Target location: (none)")
    elif isinstance(trans.target, (int, float, np.integer, np.floating)) or \
         (isinstance(trans.target, np.ndarray) and trans.target.size == 1):
        # MATLAB: ordinary transition (e.g., in hybrid automaton)
        target_val = trans.target if isinstance(trans.target, (int, float)) else trans.target.item()
        output_lines.append(f"Target location: {target_val}")
    else:
        # MATLAB: parallel hybrid automaton
        target_str = ", ".join(str(t) for t in np.array(trans.target).flatten())
        output_lines.append(f"Target locations: {target_str}")
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    # MATLAB: display synchronization label
    # if ~isempty(trans.syncLabel)
    if hasattr(trans, 'syncLabel') and trans.syncLabel:
        output_lines.append(f"Synchronization label: '{trans.syncLabel}'")
    else:
        output_lines.append("Synchronization label: (none)")
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    return "\n".join(output_lines)


def _display_matrix_vector(mat: Any, mat_name: str) -> list:
    """
    Display a matrix or vector with a label
    
    Args:
        mat: Matrix or vector to display
        mat_name: Name/label for the matrix
    
    Returns:
        List of output lines
    """
    output_lines = []
    if mat is None:
        return output_lines
    if isinstance(mat, np.ndarray):
        if mat.size == 0:
            output_lines.append(f"  {mat_name}: []")
        else:
            output_lines.append(f"  {mat_name}:")
            # Format matrix nicely
            if mat.ndim == 1:
                mat = mat.reshape(-1, 1)
            for i in range(mat.shape[0]):
                row_str = "    " + " ".join(f"{val:8.4f}" if isinstance(val, (int, float, np.number)) else str(val) 
                                           for val in mat[i, :])
                output_lines.append(row_str)
    else:
        output_lines.append(f"  {mat_name}: {mat}")
    return output_lines


def display(trans: Any, var_name: str = None) -> None:
    """
    Display a transition object (prints to stdout)
    
    Args:
        trans: Transition object
        var_name: Optional variable name (for display)
    """
    print(display_(trans, var_name), end='')
