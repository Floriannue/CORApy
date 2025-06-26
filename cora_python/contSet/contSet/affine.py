from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def affine(self, *varargin):
    """
    affine - conversion to affine objects

    Syntax:
        aff = affine(S)

    Inputs:
        S - contSet object

    Outputs:
        aff - affine object

    Example:
        I = interval([2,1],[3,4])
        aff = affine(I)
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 