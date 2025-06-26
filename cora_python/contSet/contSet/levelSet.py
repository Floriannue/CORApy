from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def levelSet(self, *varargin):
    """
    conversion to levelSet objects

    Syntax:
        ls = levelSet(S)

    Inputs:
        S - contSet object

    Outputs:
        ls - levelSet object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 