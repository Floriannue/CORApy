from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def polytope(self, *varargin):
    """
    conversion to polytope objects

    Syntax:
        P = polytope(S)

    Inputs:
        S - contSet object

    Outputs:
        P - polytope object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 