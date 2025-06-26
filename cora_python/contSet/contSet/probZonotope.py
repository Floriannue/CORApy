from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def probZonotope(self, *varargin):
    """
    conversion to probZonotope objects

    Syntax:
        probZ = probZonotope(S)

    Inputs:
        S - contSet object

    Outputs:
        probZ - probZonotope object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 