from typing import List, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def assertNarginConstructor(numValidArgs: Union[List[int], List[float]], n_in: int) -> bool:
    """
    assertNarginConstructor - asserts if the number of input arguments
    matches any entry in a given list of admissible values

    Syntax:
       assertNarginConstructor(numValidArgs,n_in)
       res = assertNarginConstructor(numValidArgs,n_in)

    Inputs:
       numValidArgs - ordered list of admissible number of input arguments
       n_in - number of input arguments

    Outputs:
       res - true if assertion is successful

    Example:
       assertNarginConstructor([0,1,4,5],1);

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: assert, assertLoop, assertThrowsAs

    Authors:       Mark Wetzlinger
    Written:       15-October-2024
    Last update:   ---
    Last revision: ---
    """

    # different between finite admissible list and infinite
    if numValidArgs and numValidArgs[-1] == float('inf'):
        # if list of admissible numbers of input arguments ends in Inf, any
        # value between the penultimate value and Inf is also allowed
        if n_in not in numValidArgs[:-1] and n_in < numValidArgs[-2]:
            # MATLAB's throwAsCaller is equivalent to raising an exception directly
            raise CORAerror('CORA:numInputArgsConstructor', numValidArgs)
    else:
        # no Inf -> must be any of the given values
        if n_in not in numValidArgs:
            raise CORAerror('CORA:numInputArgsConstructor', numValidArgs)

    return True 