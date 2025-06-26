from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def cartProd_(self, *varargin):
    """
    cartProd_ - computes the Cartesian product of two sets
    (internal use, see also contSet/cartProd)

    Syntax:
        S = cartProd_(S1, S2)

    Inputs:
        S1 - contSet object
        S2 - contSet object

    Outputs:
        res - contSet object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror("CORA:noops", self, *varargin) 