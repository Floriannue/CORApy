from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def fullspace(self, *varargin):
    """
    conversion to fullspace objects

    Syntax:
        fs = fullspace(S)

    Inputs:
        S - contSet object

    Outputs:
        fs - fullspace object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror("CORA:noops", self) 