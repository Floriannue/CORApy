"""
bracketSubs - substitute 'L' and 'R' by opening/closing parenthesis

Syntax:
    str = bracketSubs(str)

Inputs:
    str - string

Outputs:
    str - string

Example:
    str = 'xL1R';
    str = bracketSubs(str);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       ???
Written:       ---
Last update:   01-May-2020 (MW, added header)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""


def bracketSubs(str_val: str) -> str:
    """
    Substitute 'L' and 'R' by opening/closing parenthesis
    
    Args:
        str_val: string
        
    Returns:
        str_val: string with 'L' replaced by '(' and 'R' replaced by ')'
    """
    
    # generate left and right brackets
    # MATLAB: str = strrep(str,'L','(');
    # MATLAB: str = strrep(str,'R',')');
    str_val = str_val.replace('L', '(')
    str_val = str_val.replace('R', ')')
    
    return str_val

