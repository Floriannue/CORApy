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
import sys


def display(trans: Any, var_name: str = None) -> None:
    """
    Display a transition object
    
    Args:
        trans: Transition object
        var_name: Optional variable name (for display)
    """
    # MATLAB: fprintf(newline);
    print()
    
    # MATLAB: disp([inputname(1), ' =']);
    if var_name is None:
        var_name = 'trans'
    print(f"{var_name} =")
    
    # MATLAB: fprintf(newline);
    print()
    
    # MATLAB: if length(trans) > 1
    # For now, handle single objects (not arrays)
    # TODO: Handle arrays if needed
    
    # MATLAB: display guard set
    # if (isnumeric(trans.guard) && isempty(trans.guard)) ...
    #         || (~isnumeric(trans.guard) && ~representsa_(trans.guard,'emptySet',eps))
    guard_is_numeric_empty = isinstance(trans.guard, (int, float, np.number)) or \
                             (isinstance(trans.guard, np.ndarray) and trans.guard.size == 0)
    
    if guard_is_numeric_empty:
        print("Guard set: (none)")
    else:
        # Check if guard represents empty set
        guard_is_empty_set = False
        if hasattr(trans.guard, 'representsa_'):
            try:
                guard_is_empty_set = trans.guard.representsa_('emptySet', np.finfo(float).eps)
            except:
                pass
        
        if guard_is_empty_set:
            print("Guard set: (none)")
        else:
            print("Guard set:")
            if hasattr(trans.guard, 'display'):
                trans.guard.display()
            else:
                print(trans.guard)
    
    # MATLAB: fprintf(newline);
    print()
    
    # MATLAB: display reset function
    # if isnumeric(trans.reset) && isempty(trans.reset)
    reset_is_numeric_empty = isinstance(trans.reset, (int, float, np.number)) or \
                             (isinstance(trans.reset, np.ndarray) and trans.reset.size == 0)
    
    if reset_is_numeric_empty:
        # MATLAB: no reset function (empty transition)
        print("Reset function: (none)")
    else:
        # Check if reset is LinearReset
        from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
        if isinstance(trans.reset, LinearReset):
            # MATLAB: linear reset function
            print("Reset function: Ax + Bu + c")
            
            # MATLAB: display state matrix
            _display_matrix_vector(trans.reset.A, "A")
            
            # MATLAB: display input matrix
            _display_matrix_vector(trans.reset.B, "B")
            
            # MATLAB: display constant offset
            _display_matrix_vector(trans.reset.c, "c")
        else:
            # Check if reset is NonlinearReset (has nonlinearReset field or is NonlinearReset)
            from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
            if isinstance(trans.reset, NonlinearReset) or \
               (hasattr(trans.reset, 'nonlinearReset') and trans.reset.nonlinearReset):
                # MATLAB: nonlinear reset function
                print("Reset function: nonlinear")
                
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
                            print(f"  f({i+1}) = {eq}")
                    else:
                        print(f"  f(1) = {f}")
                    print()
                except:
                    # Silently fail if symbolic display fails
                    pass
    
    # MATLAB: display target
    # if isempty(trans.target)
    target_is_empty = isinstance(trans.target, np.ndarray) and trans.target.size == 0
    
    if target_is_empty:
        # MATLAB: empty transition object
        print("Target location: (none)")
    elif isinstance(trans.target, (int, float, np.integer, np.floating)) or \
         (isinstance(trans.target, np.ndarray) and trans.target.size == 1):
        # MATLAB: ordinary transition (e.g., in hybrid automaton)
        target_val = trans.target if isinstance(trans.target, (int, float)) else trans.target.item()
        print(f"Target location: {target_val}")
    else:
        # MATLAB: parallel hybrid automaton
        target_str = ", ".join(str(t) for t in np.array(trans.target).flatten())
        print(f"Target locations: {target_str}")
    
    # MATLAB: fprintf(newline);
    print()
    
    # MATLAB: display synchronization label
    # if ~isempty(trans.syncLabel)
    if hasattr(trans, 'syncLabel') and trans.syncLabel:
        print(f"Synchronization label: '{trans.syncLabel}'")
    else:
        print("Synchronization label: (none)")
    
    # MATLAB: fprintf(newline);
    print()


def _display_matrix_vector(mat: Any, mat_name: str) -> None:
    """
    Display a matrix or vector with a label
    
    Args:
        mat: Matrix or vector to display
        mat_name: Name/label for the matrix
    """
    if isinstance(mat, np.ndarray):
        if mat.size == 0:
            print(f"  {mat_name}: []")
        else:
            print(f"  {mat_name}:")
            print(f"    {mat}")
    else:
        print(f"  {mat_name}: {mat}")

