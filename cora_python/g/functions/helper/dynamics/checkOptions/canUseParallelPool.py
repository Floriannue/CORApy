"""
canUseParallelPool - checks if parallel pool can be used

Syntax:
    canUse = canUseParallelPool()

Inputs:
    -

Outputs:
    canUse - boolean indicating if parallel pool can be used

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       ---
Written:       ---
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import multiprocessing


def canUseParallelPool() -> bool:
    """
    Checks if parallel pool can be used
    
    Returns:
        canUse: boolean indicating if parallel pool can be used
    """
    
    try:
        # Check if multiprocessing is available and has more than 1 CPU
        # MATLAB: checks if parallel pool exists and is active
        # Python equivalent: check if multiprocessing is available
        cpu_count = multiprocessing.cpu_count()
        return cpu_count > 1
    except:
        return False

