from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def capsule(self, *varargin):
    """
    conversion to capsule objects

    Syntax:
        C = capsule(S)

    Inputs:
        S - contSet object

    Outputs:
        C - capsule object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 