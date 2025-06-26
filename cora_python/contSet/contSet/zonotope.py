from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def zonotope(self, *varargin):
    """
    conversion to zonotope objects

    Syntax:
        Z = zonotope(S)

    Inputs:
        S - contSet object

    Outputs:
        Z - zonotope object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 