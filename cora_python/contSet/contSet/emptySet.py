from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def emptySet(self, *varargin):
    """
    conversion to emptySet objects

    Syntax:
        O = emptySet(S)

    Inputs:
        S - contSet object

    Outputs:
        O - emptySet object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror("CORA:noops", self) 